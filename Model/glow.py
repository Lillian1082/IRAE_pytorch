# This version of Glow is modified for GDN_I
"""
Glow: Generative Flow with Invertible 1x1 Convolutions
arXiv:1807.03039v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.checkpoint import checkpoint


best_eval_logprob = float('-inf')

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
        return z

    def inverse(self, z):
        # print('Actnorm:', torch.isnan(z).sum())
        return z * self.scale + self.bias


class Invertible1x1Conv(nn.Module):
    """ Invertible 1x1 convolution layer; cf Glow section 3.2 """
    def __init__(self, n_channels=3, lu_factorize=False):
        super().__init__()
        self.lu_factorize = lu_factorize

        # initiaize a 1x1 convolution weight matrix
        w = torch.randn(n_channels, n_channels)
        w = torch.qr(w)[0]  # note: nn.init.orthogonal_ returns orth matrices with dets +/- 1 which complicates the inverse call below

        if lu_factorize:
            # compute LU factorization
            p, l, u = torch.btriunpack(*w.unsqueeze(0).btrifact())
            # initialize model parameters
            self.p, self.l, self.u = nn.Parameter(p.squeeze()), nn.Parameter(l.squeeze()), nn.Parameter(u.squeeze())
            s = self.u.diag()
            self.log_s = nn.Parameter(s.abs().log())
            self.register_buffer('sign_s', s.sign())  # note: not optimizing the sign; det W remains the same sign
            self.register_buffer('l_mask', torch.tril(torch.ones_like(self.l), -1))  # store mask to compute LU in forward/inverse pass
        else:
            self.w = nn.Parameter(w)

    def forward(self, x):
        B,C,H,W = x.shape
        if self.lu_factorize:
            l = self.l * self.l_mask + torch.eye(C).to(self.l.device)
            u = self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp())
            self.w = self.p @ l @ u

        return F.conv2d(x, self.w.view(C,C,1,1)) #, logdet

    def inverse(self, z):
        B,C,H,W = z.shape
        if self.lu_factorize:
            l = torch.inverse(self.l * self.l_mask + torch.eye(C).to(self.l.device))
            u = torch.inverse(self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp()))
            w_inv = u @ l @ self.p.inverse()
        else:
            w_inv = self.w.inverse()
        # print('Invertible Conv:', torch.isnan(z).sum(), z)
        # print('w_inv', torch.isnan(w_inv).sum(), w_inv)
        output = F.conv2d(z, w_inv.view(C, C, 1, 1))
        # print('Invertible Conv output:', torch.isnan(output).sum(), output)
        return output


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
        x_a, x_b = x.chunk(2, 1)  # split along channel dim

        h = F.relu(self.actnorm1(self.conv1(x_b)))
        h = F.relu(self.actnorm2(self.conv2(h)))
        h = self.conv3(h) * self.log_scale_factor.exp()
        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels
        s = torch.sigmoid(s + 2.)  # at initalization, s is 0 and sigmoid(2) is near identity

        z_a = s * x_a + t
        z_a = torch.clamp(z_a, -1e+9, 1e+9) # added
        # z_a = F.tanh(z_a)
        z_b = x_b
        z = torch.cat([z_a, z_b], dim=1)  # concat along channel dim

        return z

    def inverse(self, z):
        # print('Affine Coupling input:', torch.isnan(z).sum(),z)
        z_a, z_b = z.chunk(2, 1)  # split along channel dim
        # print('Affine Coupling 0:', torch.isnan(z_a).sum(), torch.isnan(z_b).sum())
        h = F.relu(self.actnorm1(self.conv1(z_b)))
        # print('Affine Coupling 1:', torch.isnan(h).sum(), h)
        h = F.relu(self.actnorm2(self.conv2(h)))
        # print('Affine Coupling 2:', torch.isnan(h).sum(), h)
        h = self.conv3(h) * self.log_scale_factor.exp()
        # print('Affine Coupling 3:', torch.isnan(h).sum(), h)
        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels
        # print('Affine Coupling s before:', torch.isnan(s).sum(), s)
        # eps = torch.tensor(1e-8).cuda()
        s = torch.sigmoid(s + 2.) #+eps
        # print(s>0)
        # print((s is not 0))
        # print(s!=0)
        # s = torch.where(s!=0, s, eps)
        # print('Affine Coupling s:', torch.isnan(s).sum(), s)

        x_a = (z_a - t) / s
        # print('Affine Coupling x_a:', torch.isnan(x_a).sum(), x_a)
        x_a = torch.clamp(x_a, -1e+9, 1e+9)
        # x_a = F.tanh(x_a)
        x_b = z_b
        # print('Affine Coupling x_b:', torch.isnan(x_b).sum(), x_b)
        x = torch.cat([x_a, x_b], dim=1)  # concat along channel dim

        # print('Affine Coupling:', torch.isnan(x).sum(), x)
        return x


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
        return x1, x2 #, z2, logdet

    def inverse(self, x1, x2): #z2
        # print('Split input:', torch.isnan(x1).sum(), torch.isnan(z2).sum())
        # x2, logdet = self.gaussianize.inverse(x1, z2)
        # print('Split output:', torch.isnan(x2).sum())
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim
        return x #, logdet


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
        h = self.net(x1) * self.log_scale_factor.exp()  # use x1 to model x2 as Gaussians; learnable scale
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]          # split along channel dims
        z2 = (x2 - m) * torch.exp(-logs)                # center and scale; log prob is computed at the model forward
        logdet = - logs.sum([1,2,3])
        return z2, logdet

    def inverse(self, x1, z2):
        # print('x1', torch.isnan(x1).sum())
        # z2 = torch.clamp(z2, -1., 1.)
        # print('z2', torch.isnan(z2).sum(), z2)
        h = self.net(x1)
        # for name, parameter in self.net.named_parameters():
        #     print('params', name, parameter) #torch.isnan(parameter).sum()
        # h = torch.clamp(h, 0., 1.)
        h = h * self.log_scale_factor.exp()
        # print('self.log_scale_factor', self.log_scale_factor, self.log_scale_factor.exp())
        # print('h', torch.isnan(h).sum())

        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]
        x2 = m + z2 * torch.exp(logs)
        # print('logs', torch.isnan(logs).sum(), logs)
        # print('torch.exp(logs)', torch.isnan(torch.exp(logs)).sum(), torch.exp(logs))
        # print('z2 * torch.exp(logs)', torch.isnan(z2 * torch.exp(logs)).sum(), z2 * torch.exp(logs))
        # print('x2', torch.isnan(x2).sum())
        logdet = logs.sum([1,2,3])
        return x2, logdet


class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x - 0.5                  # center x at 0

    def inverse(self, x):
        return x + 0.5

# --------------------
# Container layers
# --------------------

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def __init__(self, *args, **kwargs):
        self.checkpoint_grads = kwargs.pop('checkpoint_grads', None)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        count = 0
        for module in self:
            x = module(x) if not self.checkpoint_grads else checkpoint(module, x)
            # print('FlowSequential Module count:%d'%count, x)
            count += 1
        return x

    def inverse(self, z):
        for module in reversed(self):
            # print('module:', module)
            z = module.inverse(z)
            # print('FlowSequential:', torch.isnan(z).sum())
            if torch.isnan(z).sum()>0:
                print('z', torch.isnan(z).sum())
                for name, parameter in module.named_parameters():
                    print('params', name, parameter)
                assert 0
        return z


class FlowStep(FlowSequential):
    """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """
    def __init__(self, n_channels, width, lu_factorize=False):
        super().__init__(Actnorm(param_dim=(1,n_channels,1,1)),
                         Invertible1x1Conv(n_channels, lu_factorize),
                         AffineCoupling(n_channels, width))


class FlowLevel(nn.Module):
    """ One depth level of Glow flow (Squeeze -> FlowStep x K -> Split); cf Glow figure 2b """
    def __init__(self, n_channels, width, depth, checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        # network layers
        self.squeeze = Squeeze()
        self.flowsteps = FlowSequential(*[FlowStep(4*n_channels, width, lu_factorize) for _ in range(depth)], checkpoint_grads=checkpoint_grads)
        self.split = Split(4*n_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.flowsteps(x)
        x1, z2 = self.split(x)
        return x1, z2

    def inverse(self, x1, z2):
        x = self.split.inverse(x1, z2) #, logdet_split
        # print('split:', torch.isnan(x).sum(), torch.isnan(x1).sum(), torch.isnan(z2).sum())
        x = self.flowsteps.inverse(x)
        # print('flowsteps:', torch.isnan(x).sum())
        x = self.squeeze.inverse(x)
        return x


# --------------------
# Model
# --------------------

class Glow(nn.Module):
    """ Glow multi-scale architecture with depth of flow K and number of levels L; cf Glow figure 2; section 3"""
    def __init__(self, width, depth, n_levels, input_dims=(3,32,32), checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        # calculate output dims
        in_channels, H, W = input_dims
        out_channels = int(in_channels * 4**(n_levels+1) / 2**n_levels)  # each Squeeze results in 4x in_channels (cf RealNVP section 3.6); each Split in 1/2x in_channels
        out_HW = int(H / 2**(n_levels+1))                                # each Squeeze is 1/2x HW dim (cf RealNVP section 3.6)
        self.output_dims = out_channels, out_HW, out_HW

        # preprocess images
        self.preprocess = Preprocess()

        # network layers cf Glow figure 2b: (Squeeze -> FlowStep x depth -> Split) x n_levels -> Squeeze -> FlowStep x depth
        self.flowlevels = nn.ModuleList([FlowLevel(in_channels * 2**i, width, depth, checkpoint_grads, lu_factorize) for i in range(n_levels)])
        self.squeeze = Squeeze()

        # base distribution of the flow
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))

    def forward(self, x):
        x = self.preprocess(x)
        zs = []
        for m in self.flowlevels:
            x, z = m(x)
            zs.append(z)
        zs.append(x)
        return zs

    def inverse(self, zs=None, batch_size=None, z_std=1.):
        x = zs[-1]
        if zs is None:  # if no random numbers are passed, generate new from the base distribution
            assert batch_size is not None, 'Must either specify batch_size or pass a batch of z random numbers.'
            zs = [z_std * self.base_dist.sample((batch_size, *self.output_dims)).squeeze()]
        for i, m in enumerate(reversed(self.flowlevels)):
            z = z_std * (self.base_dist.sample(x.shape).squeeze() if len(zs)==1 else zs[-i-2])  # if no z's are passed, generate new random numbers from the base dist
            x = m.inverse(x, z)
        x = self.preprocess.inverse(x)
        return x

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)
