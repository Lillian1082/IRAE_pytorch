import cv2
import torch
# import torch.distributions as D
# import pyro.distributions as D
import math
# from Model.UNet import *
from Model.DnCNN import *
from Model.glow import Glow

class IRAE(nn.Module): # Only Encoder and Decoder; No Transformation
    def __init__(self, args):
        super().__init__()
        self.args = args
        in_channels = 3 # Greyscale 1; Color 3
        input_dims = (in_channels, args.cropSize, args.cropSize)
        self.out_channels = int(in_channels * 4 ** args.n_levels / 2 ** args.n_levels)

        self.block1 = Glow(args.width, args.depth, args.n_levels, input_dims, args.checkpoint_grads)
        self.block2 = Glow(args.width, args.depth, args.n_levels, input_dims, args.checkpoint_grads)

    def forward(self, noisy, bits_per_pixel=False):
        # print('noisy:', noisy.shape)
        zs = self.block1(noisy)
        # for i in range(len(zs)):
        #     print(i, torch.isnan(zs[i]).sum())
        # for name, parameter in self.block1.named_parameters():
        #     print('params block1', name, parameter) #torch.isnan(parameter).sum()
        # for name, parameter in self.block2.named_parameters():
        #     print('params block2', name, parameter) #torch.isnan(parameter).sum()

        if torch.isnan(zs[-1]).sum() > 0:
            print('zs[-1]', torch.isnan(zs[-1]).sum())
            for name, parameter in self.block1.named_parameters():
                print('params block1', name, parameter) # torch.isnan(parameter).sum()
            assert 0

        output = self.block2.inverse(zs=zs, batch_size=self.args.batch_size, z_std=1.)
        return output

    def inverse(self, clean):
        zs = self.block2(clean)
        output = self.block1.inverse(zs=zs, batch_size=self.args.batch_size, z_std=1.)
        return output