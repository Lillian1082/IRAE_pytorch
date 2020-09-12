import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from torchvision.datasets import MNIST
from datasets.celeba import CelebA

import numpy as np
from tensorboardX import SummaryWriter

import os
import time
import math
import argparse
import pprint

# --------------------
# Model component layers
# --------------------

class Actnorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1,3,1,1)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x):
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, False) + 1e-6).view_as(self.bias)
            self.initialized += 1

        z = (x - self.bias) / self.scale
        # logdet = - self.scale.abs().log().sum() * x.shape[2] * x.shape[3]
        # print('Actnorm z', z.shape)
        return z #, logdet

    def inverse(self, z):
        return z * self.scale + self.bias, self.scale.abs().log().sum() * z.shape[2] * z.shape[3]


class Invertible1x1Conv(nn.Module):
    """ Invertible 1x1 convolution layer; cf Glow section 3.2 """
    def __init__(self, n_channels=3, lu_factorize=False):
        # super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        # self.lu_factorize = lu_factorize
        #
        # # initiaize a 1x1 convolution weight matrix
        # w = torch.randn(n_channels, n_channels)
        # w = torch.qr(w)[0]  # note: nn.init.orthogonal_ returns orth matrices with dets +/- 1 which complicates the inverse call below
        #
        # if lu_factorize:
        #     # compute LU factorization
        #     p, l, u = torch.btriunpack(*w.unsqueeze(0).btrifact())
        #     # initialize model parameters
        #     self.p, self.l, self.u = nn.Parameter(p.squeeze()), nn.Parameter(l.squeeze()), nn.Parameter(u.squeeze())
        #     s = self.u.diag()
        #     self.log_s = nn.Parameter(s.abs().log())
        #     self.register_buffer('sign_s', s.sign())  # note: not optimizing the sign; det W remains the same sign
        #     self.register_buffer('l_mask', torch.tril(torch.ones_like(self.l), -1))  # store mask to compute LU in forward/inverse pass
        # else:
        #     self.w = nn.Parameter(w)

    def forward(self, x):
        # print('invertible conv x', x.shape)
        B, C, H, W = x.shape
        output = self.conv(x)
        return output
        # B,C,H,W = x.shape
        # if self.lu_factorize:
        #     l = self.l * self.l_mask + torch.eye(C).to(self.l.device)
        #     u = self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp())
        #     self.w = self.p @ l @ u
        #     logdet = self.log_s.sum() * H * W
        # else:
        #     logdet = torch.slogdet(self.w)[-1] * H * W
        #
        # return F.conv2d(x, self.w.view(C,C,1,1)), logdet

    def inverse(self, z):
        B,C,H,W = z.shape
        if self.lu_factorize:
            l = torch.inverse(self.l * self.l_mask + torch.eye(C).to(self.l.device))
            u = torch.inverse(self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp()))
            w_inv = u @ l @ self.p.inverse()
            logdet = - self.log_s.sum() * H * W
        else:
            w_inv = self.w.inverse()
            logdet = - torch.slogdet(self.w)[-1] * H * W

        return F.conv2d(z, w_inv.view(C,C,1,1)), logdet

class Conv(nn.Module):
    """ Invertible 1x1 convolution layer; cf Glow section 3.2 """
    def __init__(self, n_channels=3):
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        output = self.conv(x)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.relu1(self.bn1(self.conv1(x)))
        out = F.relu(self.conv2(residual) + x) # addition Structure 6-1 elf.bn2
        return out

class AffineCoupling(nn.Module):
    """ Affine coupling layer; cf Glow section 3.3; RealNVP figure 2 """
    def __init__(self, n_channels, width):
        super().__init__()
        # network layers;
        # per realnvp, network splits input, operates on half of it, and returns shift and scale of dim = half the input channels
        self.conv1 = nn.Conv2d(n_channels//2, width, kernel_size=3, padding=1, bias=False)  # input is split along channel dim
        self.actnorm1 = Actnorm(param_dim=(1, width, 1, 1))
        self.conv2 = nn.Conv2d(width, width, kernel_size=1, padding=1, bias=False)
        self.actnorm2 = Actnorm(param_dim=(1, width, 1, 1))
        self.conv3 = nn.Conv2d(width, n_channels, kernel_size=3)            # output is split into scale and shift components
        self.log_scale_factor = nn.Parameter(torch.zeros(n_channels,1,1))   # learned scale (cf RealNVP sec 4.1 / Glow official code

        # initialize last convolution with zeros, such that each affine coupling layer performs an identity function
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()

    def forward(self, x):
        # print('affine x', x.shape)
        x_a, x_b = x.chunk(2, 1)  # split along channel dim
        # print('affine x_a', x_a.shape, x_b.shape)
        h = F.relu(self.actnorm1(self.conv1(x_b))) #[0]
        h = F.relu(self.actnorm2(self.conv2(h))) #[0]
        h = self.conv3(h) * self.log_scale_factor.exp()
        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels
        s = torch.sigmoid(s + 2.)  # at initalization, s is 0 and sigmoid(2) is near identity

        z_a = s * x_a + t
        z_b = x_b
        z = torch.cat([z_a, z_b], dim=1)  # concat along channel dim

        # logdet = s.log().sum([1, 2, 3])

        return z #, logdet

    def inverse(self, z):
        z_a, z_b = z.chunk(2, 1)  # split along channel dim

        h = F.relu(self.actnorm1(self.conv1(z_b))[0])
        h = F.relu(self.actnorm2(self.conv2(h))[0])
        h = self.conv3(h)  * self.log_scale_factor.exp()
        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels
        s = torch.sigmoid(s + 2.)

        x_a = (z_a - t) / s
        x_b = z_b
        x = torch.cat([x_a, x_b], dim=1)  # concat along channel dim

        logdet = - s.log().sum([1, 2, 3])

        return x, logdet


class Squeeze(nn.Module):
    """ RealNVP squeezing operation layer (cf RealNVP section 3.6; Glow figure 2b):
    For each channel, it divides the image into subsquares of shape 2 × 2 × c, then reshapes them into subsquares of
    shape 1 × 1 × 4c. The squeezing operation transforms an s × s × c tensor into an s × s × 4c tensor """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C, H//2, 2, W//2, 2)   # factor spatial dim
        x = x.permute(0, 1, 3, 5, 2, 4)         # transpose to (B, C, 2, 2, H//2, W//2)
        x = x.reshape(B, 4*C, H//2, W//2)       # aggregate spatial dim factors into channels
        return x

    def inverse(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C//4, 2, 2, H, W)      # factor channel dim
        x = x.permute(0, 1, 4, 2, 5, 3)         # transpose to (B, C//4, H, 2, W, 2)
        x = x.reshape(B, C//4, 2*H, 2*W)        # aggregate channel dim factors into spatial dims
        return x


class Split(nn.Module):
    """ Split layer; cf Glow figure 2 / RealNVP figure 4b
    Based on RealNVP multi-scale architecture: splits an input in half along the channel dim; half the vars are
    directly modeled as Gaussians while the other half undergo further transformations (cf RealNVP figure 4b).
    """
    def __init__(self, n_channels):
        super().__init__()
        self.gaussianize = Gaussianize(n_channels//2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # split input along channel dim
        # z2, logdet = self.gaussianize(x1, x2)
        z2 = self.gaussianize(x1, x2)
        return x1, z2 #, logdet

    def inverse(self, x1, z2):
        x2, logdet = self.gaussianize.inverse(x1, z2)
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim
        return x, logdet


class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Conv2d(n_channels, 2*n_channels, kernel_size=3, padding=1)  # computes the parameters of Gaussian
        self.log_scale_factor = nn.Parameter(torch.zeros(2*n_channels,1,1))       # learned scale (cf RealNVP sec 4.1 / Glow official code
        # initialize to identity
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1, x2):
        # print('x1', x1.shape)
        h = self.net(x1) * self.log_scale_factor.exp()  # use x1 to model x2 as Gaussians; learnable scale
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]          # split along channel dims
        z2 = (x2 - m) * torch.exp(-logs)                # center and scale; log prob is computed at the model forward
        # logdet = - logs.sum([1,2,3])
        return z2 #, logdet

    def inverse(self, x1, z2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]
        x2 = m + z2 * torch.exp(logs)
        # logdet = logs.sum([1,2,3])
        return x2 #, logdet


class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        logdet = - math.log(256) * x[0].numel() # processing each image dim from [0, 255] to [0,1]; per RealNVP sec 4.1 taken into account
        return x - 0.5, logdet                  # center x at 0

    def inverse(self, x):
        logdet = math.log(256) * x[0].numel()
        return x + 0.5, logdet

# --------------------
# Container layers
# --------------------

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def __init__(self, *args, **kwargs):
        self.checkpoint_grads = kwargs.pop('checkpoint_grads', None)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        sum_logdets = 0.
        for module in self:
            # x, logdet = module(x) if not self.checkpoint_grads else checkpoint(module, x)
            x = module(x) if not self.checkpoint_grads else checkpoint(module, x)
            # sum_logdets = sum_logdets + logdet
        return x #, sum_logdets

    def inverse(self, z):
        sum_logdets = 0.
        for module in reversed(self):
            z, logdet = module.inverse(z)
            sum_logdets = sum_logdets + logdet
        return z, sum_logdets

class FlowStep(FlowSequential): # Conv version
    """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """
    def __init__(self, n_channels, width, lu_factorize=False):
        super().__init__(#Actnorm(param_dim=(1,n_channels,1,1)),
                         # Invertible1x1Conv(n_channels), #, lu_factorize
                         nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                         nn.BatchNorm2d(n_channels),
                         nn.ReLU(),
                         nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                         nn.BatchNorm2d(n_channels),
                         nn.ReLU(),
                         nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                         nn.BatchNorm2d(n_channels),
                         nn.ReLU(),
                         nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                         AffineCoupling(n_channels, width))

# class FlowStep(FlowSequential): # Conv version
#     """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """
#     def __init__(self, n_channels, width, lu_factorize=False):
#         super().__init__(Actnorm(param_dim=(1,n_channels,1,1)),
#                          # Invertible1x1Conv(n_channels), #, lu_factorize
#                          nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
#                          nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
#                          nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
#                          nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
#                          AffineCoupling(n_channels, width))

# class FlowStep(FlowSequential): # Res version
#     """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """
#     def __init__(self, n_channels, width, lu_factorize=False):
#         super().__init__(Actnorm(param_dim=(1,n_channels,1,1)),
#                          # Invertible1x1Conv(n_channels), #, lu_factorize
#                          ResidualBlock(n_channels),
#                          AffineCoupling(n_channels, width))

class FlowLevel(nn.Module):
    """ One depth level of Glow flow (Squeeze -> FlowStep x K -> Split); cf Glow figure 2b """
    def __init__(self, n_channels, width, depth, checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        # network layers
        self.squeeze = Squeeze()
        # self.flowsteps = [FlowStep(4*n_channels, width, lu_factorize) for _ in range(depth)]
        self.flowsteps = FlowSequential(*[FlowStep(4*n_channels, width, lu_factorize) for _ in range(depth)], checkpoint_grads=checkpoint_grads)
        self.split = Split(n_channels)

    def forward(self, x):
        # print('x 0', x.shape)
        x = self.squeeze(x)
        # print('x 1', x.shape)
        x = self.flowsteps(x)
        # x, logdet_flowsteps = self.flowsteps(x)
        # print('x', x.shape)
        # x = self.squeeze.inverse(x)
        # print('x', x.shape)
        x1, z2 = self.split(x) #logdet_split
        # x1 = self.squeeze.inverse(x1)
        # z2 = self.squeeze.inverse(z2)
        # print('x1', x1.shape, 'z2', z2.shape)
        # logdet = logdet_flowsteps + logdet_split
        return x1, z2 #, logdet

    def inverse(self, x1, z2):
        x, logdet_split = self.split.inverse(x1, z2)
        x, logdet_flowsteps = self.flowsteps.inverse(x)
        x = self.squeeze.inverse(x)
        logdet = logdet_flowsteps + logdet_split
        return x, logdet
