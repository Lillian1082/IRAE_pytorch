import math
import os
import argparse
import pprint

import torchvision

import cv2
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from Model.DnCNN import DnCNN
from Model.ARCNN import ARCNN
from Model.FFDNet import FFDNet
# from Model.new_GDN import GDN
from Model.GDN import GAE, GDN_I, GDN_II, GDN_IV, GAE_woSplit, GAE_woSplit_sk  #GDN, Parallel_GDN1, Parallel_GDN2
# from Model.glow import Glow
# from Model.glow_SamDim import Glow
# from Model.glow import GDN
from math import log10
import random
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from utilsData import Dataset_Grad
from utils import batchPSNR, batchSSIM, addNoise, weights_init_onetoone, plot_grad_flow, random_bbox, mask_image
from torchvision.datasets import MNIST
from datasets.celeba import CelebA
from datasets.flower import Flower
from datasets.bird import Bird
from datasets.RICE import RICE_I, RICE_II
# from datasets.moons import MOONS
import matplotlib.pyplot as plt
from Model.UNet import UNet

os.environ["CUDADEVICE_ORDER"] = "PCIBUS_ID"

parser = argparse.ArgumentParser()

# action
parser.add_argument('--archi', default='', type=str, help='Network Architecture.')
parser.add_argument('--train', action='store_true', help='Train a flow.') #store_true
# parser.add_argument('--evaluate', action='store_false', help='Evaluate a flow.') #store_true
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--denoise', action='store_true', help='Denoise an image.')
parser.add_argument('--learn_residual', action='store_true', help='Whether learn residual for denoising.')
parser.add_argument('--inpainting', action='store_true', help='Whether it is an inpainting task.')
parser.add_argument("--inpainting_mode", type=str, default='center', help="center or irregular")
parser.add_argument("--mask_shape", type=int, default=[128, 128], help="shape of the mask")
parser.add_argument('--decompression', action='store_true', help='Whether it is a decompression task.')
parser.add_argument("--decompression_level", type=str, default='X10', help="adjustable decompression level")
parser.add_argument('--visualize', action='store_true', help='Visualize manipulated attribures.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--restore_checkpoint', type=str, help='Path to model to restore.')
parser.add_argument('--restore_optimizer', type=str, help='Path to model to restore.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
# paths and reporting
parser.add_argument('--data_dir', default='/mnt/disks/data/', help='Location of datasets.')
parser.add_argument("--train_dataPath", type=str, default='data/SIDD/SIDD_train', help='path of training files to process') #data/BSDS500_new/train
parser.add_argument("--test_dataPath", type=str, default='data/SIDD/SIDD_val', help='path of validation files to process') #data/BSDS500_new/CBSD68
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0].split('/')[-1])) # os.path.splitext(__file__)[0]
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
parser.add_argument('--log_interval', type=int, default=1, help='How often to show loss statistics and save samples.')
parser.add_argument('--save_interval', type=int, default=50, help='How often to save during training.')
parser.add_argument('--eval_interval', type=int, default=1, help='Number of epochs to eval model and save model checkpoint.')
parser.add_argument('--radius', type=int, default=2, help='The radius for Gaussian Blur')
parser.add_argument('--noise_estimate', type=bool, default=0, help='whether to estimate the sigma map.')
# data
parser.add_argument('--dataset', type=str, help='Which dataset to use.')
parser.add_argument("--noise_level", type=float, default=15, help="adjustable noise level")
parser.add_argument("--noiseIntL", nargs=2, type=int, default=[0.01, 55], \
                    help="Noise training interval for FFDNet")
parser.add_argument("--noise_mode", type=str, default='S', help="mode of AWGN noise")
# parser.add_argument("--outf", type=str, default="logs/try", help='path of log files')
# parser.add_argument('--crop', default=False, type=bool, help='crop patches?')
parser.add_argument('--cropSize', default=64, type=int,  help='crop patches? training images crop size')
parser.add_argument('--real', default=0, type=bool, help='Real Dataset?')
# parser.add_argument('--seed', default=0, type= int, help='seed of all random')
parser.add_argument('--randomCount', default=10, type= int, help='the number of patches got from each patch')
parser.add_argument('--augment', default=True, type= bool, help='whether to apply data augmentation to it')
# model parameters
parser.add_argument('--depth', type=int, default=10, help='Depth of the network (cf Glow figure 2).')
parser.add_argument('--n_levels', type=int, default=1, help='Number of levels of the network (cf Glow figure 2).')
parser.add_argument('--num_layers', type=int, default=2, help='Number of conv layers of the skip-connection.')
parser.add_argument('--width', type=int, default=64, help='Dimension of the hidden layers.')
parser.add_argument('--z_std', type=float, help='Pass specific standard devition during generation/sampling.')

# training params
parser.add_argument('--batch_size', type=int, default=40, help='Training batch size.')
parser.add_argument('--batch_size_init', type=int, default=256, help='Batch size for the data dependent initialization.')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--n_epochs_warmup', type=int, default=1, help='Number of warmup epochs for linear learning rate annealing.')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')

parser.add_argument('--mannual_set_lr', action='store_true', help='Whether use the loaded lr.')

parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument('--gamma', type=float, default=0.5,
                    help="Decaying rate for the learning rate, (default: 0.5)")
parser.add_argument('--mini_data_size', type=int, default=None, help='Train only on this number of datapoints.')
parser.add_argument('--grad_norm_clip', default=50, type=float, help='Clip gradients during training.')

parser.add_argument('--checkpoint_grads', action='store_false', default=False, help='Whether to use gradient checkpointing in forward pass.')
parser.add_argument('--n_bits', default=5, type=int, help='Number of bits for input images.')
# distributed training params
parser.add_argument('--distributed', action='store_false', default=False, help='Whether to use DistributedDataParallels on multiple machines and GPUs.')
parser.add_argument('--world_size', type=int, default=2, help='Number of nodes for distributed training.')
# parser.add_argument('--local_rank', type=int, help='When provided, run model on this cuda device. When None, used by torch.distributed.launch utility to manage multi-GPU training.')
parser.add_argument('--device_ids', type=int, help='When provided, run model on this cuda device. When None, used by torch.distributed.launch utility to manage multi-GPU training.')
# visualize
parser.add_argument('--vis_img', type=str, help='Path to image file to manipulate attributes and visualize.')
parser.add_argument('--vis_attrs', nargs='+', type=int, help='Which attribute to manipulate.')
parser.add_argument('--vis_alphas', nargs='+', type=float, help='Step size on the manipulation direction.')


best_eval_logprob = float('-inf')

####################################################################
args = parser.parse_args()
torch.backends.cudnn.enabled = False # will make the speed slow, cudnn could speedup the training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# --------------------
# Train and evaluate
# --------------------
# mse_criterion = nn.MSELoss()
mse_criterion = nn.L1Loss()

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def fetch_dataloader(args, train=True, data_dependent_init=False):
    args.input_dims = {'mnist': (3, 32, 32), 'celeba': (3, 64, 64), 'flower': (3, 128, 128), 'bird': (3, 128, 128), 'moons': (32, 32), \
                       'RICE1': (3, 256, 256), 'RICE2': (3, 256, 256)}[args.dataset] # 'celeba': (3,64,64) for image denoising
    transforms = {'mnist': T.Compose([T.Pad(2),                                         # image to 32x32 same as CIFAR
                                      T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # random shifts to fill the padded pixels
                                      T.ToTensor(),
                                      T.Lambda(lambda t: t + torch.rand_like(t)/2**8),  # dequantize
                                      T.Lambda(lambda t: t.expand(3,-1,-1))]),          # expand to 3 channels

                  'celeba': T.Compose([T.CenterCrop(148),  # RealNVP preprocessing; image denoising
                                       T.Resize(64), # for image inpainting, mask shape 32*32
                                       # T.RandomHorizontalFlip(),
                                       # T.RandomVerticalFlip(),
                                       # T.RandomRotation(90),
                                       # T.RandomRotation(180),
                                       # T.RandomRotation(270),
                                       T.Lambda(lambda im: np.array(im, dtype=np.float32)),                     # to numpy
                                       T.Lambda(lambda x: np.floor(x / 2**(8 - args.n_bits)) / 2**args.n_bits), # lower bits
                                       T.ToTensor(),  # note: if input to this transform is uint8, it divides by 255 and returns float
                                       T.Lambda(lambda t: t + torch.rand_like(t) / 2**args.n_bits)]), # dequantize

                  # 'celeba': T.Compose([T.CenterCrop(148),  # RealNVP preprocessing; image inpainting
                  #                      T.Resize((256, 256)),  # for image inpainting, mask shape 32*32
                  #                      T.Lambda(lambda im: np.array(im, dtype=np.float32)),  # to numpy
                  #                      T.Lambda(lambda x: np.floor(x / 2 ** (8 - args.n_bits)) / 2 ** args.n_bits),
                  #                      # lower bits
                  #                      T.ToTensor(),
                  #                      # note: if input to this transform is uint8, it divides by 255 and returns float
                  #                      T.Lambda(lambda t: t + torch.rand_like(t) / 2 ** args.n_bits)]),  # dequantize

                  'flower': T.Compose([T.CenterCrop(400),  # RealNVP preprocessing
                                       T.Resize(128),
                                       # T.RandomHorizontalFlip(),
                                       # T.RandomVerticalFlip(),
                                       # T.RandomRotation(90),
                                       # T.RandomRotation(180),
                                       # T.RandomRotation(270),
                                       T.Lambda(lambda im: np.array(im, dtype=np.float32)),  # to numpy
                                       T.Lambda(lambda x: np.floor(x / 2 ** (8 - args.n_bits)) / 2 ** args.n_bits),# lower bits
                                       T.ToTensor(),
                                       # note: if input to this transform is uint8, it divides by 255 and returns float
                                       T.Lambda(lambda t: t + torch.rand_like(t) / 2 ** args.n_bits)]), # dequantize

                  'bird': T.Compose([T.CenterCrop(300),  # RealNVP preprocessing
                                       T.Resize(128),
                                       # T.RandomHorizontalFlip(),
                                       # T.RandomVerticalFlip(),
                                       # T.RandomRotation(90),
                                       # T.RandomRotation(180),
                                       # T.RandomRotation(270),
                                       T.Lambda(lambda im: np.array(im, dtype=np.float32)),  # to numpy
                                       T.Lambda(
                                         lambda x: np.floor(x / 2 ** (8 - args.n_bits)) / 2 ** args.n_bits),
                                       # lower bits
                                       T.ToTensor(),
                                       # note: if input to this transform is uint8, it divides by 255 and returns float
                                       T.Lambda(lambda t: t + torch.rand_like(t) / 2 ** args.n_bits)]),# dequantize

                  'RICE1': T.Compose([T.Resize(256),
                                     T.Lambda(lambda im: np.array(im, dtype=np.float32)),  # to numpy
                                     T.Lambda(lambda x: np.floor(x / 2 ** (8 - args.n_bits)) / 2 ** args.n_bits),
                                     # lower bits
                                     T.ToTensor(),
                                     # note: if input to this transform is uint8, it divides by 255 and returns float
                                     T.Lambda(lambda t: t + torch.rand_like(t) / 2 ** args.n_bits)]),  # dequantize

                  'RICE2': T.Compose([T.Resize(256),
                                      T.Lambda(lambda im: np.array(im, dtype=np.float32)),  # to numpy
                                      T.Lambda(lambda x: np.floor(x / 2 ** (8 - args.n_bits)) / 2 ** args.n_bits),
                                      # lower bits
                                      T.ToTensor(),
                                      # note: if input to this transform is uint8, it divides by 255 and returns float
                                      T.Lambda(lambda t: t + torch.rand_like(t) / 2 ** args.n_bits)]),  # dequantize

                  'moons': T.ToTensor()
                  }[args.dataset]

    dataset = {'mnist': MNIST, 'celeba': CelebA, 'flower': Flower, 'bird': Bird, 'RICE1': RICE_I, 'RICE2': RICE_II}[args.dataset] #, 'moons': MOONS

    # Information Print
    print('Train:', train)
    print('Denoise:', args.denoise)
    print('Learn residual:', args.learn_residual)
    print('Inpainting:', args.inpainting)

    if train:
        dataset = dataset(root=args.data_dir, args=args, train=train, transform=transforms) # , mask_transform=mask_transform
    else:
        dataset = dataset(root=args.data_dir, args=args, train=train, transform=transforms) # , mask_transform=mask_transform; mini_data_size=500 is for CelebA inpainting

    if args.mini_data_size:
        dataset.data = dataset.data[:args.mini_data_size]

    # load sampler and dataloader
    if args.distributed and train is True and not data_dependent_init:  # distributed training; but exclude initialization
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    batch_size = args.batch_size_init if data_dependent_init else args.batch_size  # if data dependent init use init batch size
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device.type is 'cuda' else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler, **kwargs) # shuffle=(sampler is None) for train

@torch.no_grad()
def data_dependent_init(model, args):
    # set up an iterator with batch size = batch_size_init and run through model
    # dataloader = fetch_dataloader(args, train=True, data_dependent_init=True)
    DDataset = Dataset_Grad(args.train_dataPath, randomCount=args.randomCount, augment=True, cropPatch=True, # opt.augment, opt.crop, opt.real
                             cropSize=args.cropSize, real=1, noiseLevel=args.noiseLevel, radius=args.radius, noise_estimate=args.noise_estimate)
    batch_size = args.batch_size_init #if data_dependent_init else args.batch_size
    dataloader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
    model(next(iter(dataloader))[0].requires_grad_(True if args.checkpoint_grads else False).to(args.device))
    del dataloader
    return True


def train_epoch(model, dataloader, optimizer, writer, epoch, last_epoch, args):
    model.train()

    tic = time.time()
    Avg_PSNR_input = 0
    Avg_PSNR_output = 0


    if epoch == args.milestone:
        last_epoch = epoch
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 10
    # elif (epoch - last_epoch) == 20: #(epoch - last_epoch) == 20: (epoch - last_epoch) == 10 for JPEG Decompression
    #     last_epoch = epoch
    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = param_group["lr"] / 5
    for param_group in optimizer.param_groups:
        print('lr', param_group["lr"], 'last_epoch', last_epoch)
        if (param_group["lr"] < 1e-6):
            assert 0


    for i, (clean, corrupted) in enumerate(dataloader):
        args.step += args.world_size

        # warmup learning rate
        if epoch <= args.n_epochs_warmup:
            optimizer.param_groups[0]['lr'] = args.lr * min(1, args.step / (len(dataloader) * args.world_size * args.n_epochs_warmup))

        if args.archi=='FFDNet':
            noise = torch.zeros(clean.size())
            if args.noise_mode=='B':
                stdn = np.random.uniform(args.noiseIntL[0], args.noiseIntL[1], \
                                         size=noise.size()[0])
            elif args.noise_mode=='S':
                # stdn = np.ones(noise.size()[0])*args.noise_level # For ICONIP
                stdn = np.ones(noise.size()[0]) * np.sqrt(args.noise_level) # For Remote Sensing

            for nx in range(noise.size()[0]):
                sizen = noise[0, :, :, :].size()
                # noise[nx, :, :, :] = torch.FloatTensor(sizen).normal_(mean=0, std=stdn[nx]/255) # For ICONIP
                noise[nx, :, :, :] = torch.FloatTensor(sizen).normal_(mean=0, std=stdn[nx]) # For Remote Sensing

            corrupted = clean + noise
            # Create input Variables
            clean = Variable(clean.cuda())
            corrupted = Variable(corrupted.cuda())
            stdn_var = Variable(torch.cuda.FloatTensor(stdn))
            output = model(corrupted, stdn_var)
        else:
            clean, corrupted = clean.cuda(), corrupted.cuda()
            output = model(corrupted)

        if args.learn_residual:
            loss = mse_criterion(output, corrupted - clean)
        else: # learn clean image
            loss = mse_criterion(output, clean)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
        optimizer.step()

        with torch.no_grad():
            psnr_input = batchPSNR(corrupted, clean, 1.)

            if args.learn_residual: # learn noise for denoise tasks
                psnr = batchPSNR(corrupted - output, clean, 1.)
                output = torch.clamp(corrupted - output, 0., 1.)
            else: # learn clean image
                psnr = batchPSNR(output, clean, 1.) # learn clean image
                output = torch.clamp(output, 0., 1.)  # output clean image

            psnr_output = batchPSNR(output, clean, 1.)

            Avg_PSNR_input += psnr_input
            Avg_PSNR_output += psnr_output

        # report stats
        if i % args.log_interval == 0:
            et = time.time() - tic              # elapsed time
            tt = len(dataloader) * et / (i+1)   # total time per epoch
            print('Epoch: [{}/{}][{}/{}]\tStep: {}\tTime: elapsed {:.0f}m{:02.0f}s / total {:.0f}m{:02.0f}s\tLoss {:.4f}\tLR {:.9f}\t'.format(
                  epoch, args.start_epoch + args.n_epochs, i+1, len(dataloader), args.step, et//60, et%60, tt//60, tt%60, loss.item(), optimizer.param_groups[0]['lr']))
            print('psnr_input:%.4f'%psnr_input, 'psnr_output:%.4f'%psnr, 'psnr_output_clipped:%.4f'%psnr_output)

    Avg_PSNR_input /= (i+1)
    Avg_PSNR_output /= (i+1)
    print('Epoch:%d'%epoch, 'PSNR Input:%.4f'%Avg_PSNR_input, 'PSNR Ouput:%.4f'%Avg_PSNR_output)
    return last_epoch

@torch.no_grad()
def evaluate(model, dataloader, args):
    model.eval()
    print('Evaluating ...\r') #, end='\r'

    logprobs = []
    for x,y in dataloader:
        x = x.to(args.device)
        logprobs.append(model.log_prob(x, bits_per_pixel=True))
    logprobs = torch.cat(logprobs, dim=0).to(args.device)
    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.std(0) / math.sqrt(len(dataloader.dataset))
    return logprob_mean, logprob_std

# @torch.no_grad()
# def generate(model, n_samples, z_stds):
#     model.eval()
#     print('Generating ...\r') #, end='\r'
#
#     samples = []
#     for z_std in z_stds:
#         sample, _ = model.inverse(batch_size=n_samples, z_std=z_std)
#         log_probs = model.log_prob(sample, bits_per_pixel=True)
#         samples.append(sample[log_probs.argsort().flip(0)])  # sort by log_prob; flip high (left) to low (right)
#     return torch.cat(samples,0)

@torch.no_grad()
def generate(model, test_dataloader):
    model.eval()
    avg_psnr_input = 0
    avg_psnr_output = 0
    avg_ssim_output = 0

    avg_time = 0
    for i, (clean, corrupted) in enumerate(test_dataloader):
        t0 = time.time()

        clean, corrupted = clean.cuda(), corrupted.cuda()
        # output = model(corrupted)
        output = model.inverse(clean)

        psnr_input = batchPSNR(corrupted, clean, 1.)
        output = torch.clamp(output, 0., 1.)
        psnr_output = batchPSNR(output, clean, 1.)

        print('psnr_input:', psnr_input, 'psnr_output', psnr_output)
        torchvision.utils.save_image(clean, 'GAE_clean_%d.jpg'%i)
        # torchvision.utils.save_image(output, 'GAE_output.jpg')
        torchvision.utils.save_image(corrupted, 'GAE_corrupted_%d.jpg'%i)

        if i==10:
            assert 0

    #     t1 = time.time() - t0
    #     avg_time += t1
    #
    #     psnr_input = batchPSNR(corrupted, clean, 1.)
    #
    #     if args.learn_residual:  # learn noise for denoise tasks
    #         psnr = batchPSNR(corrupted - output, clean, 1.)
    #         output = torch.clamp(corrupted - output, 0., 1.)
    #     else:  # learn clean image
    #         psnr = batchPSNR(output, clean, 1.)  # learn clean image
    #         output = torch.clamp(output, 0., 1.)  # output clean image
    #
    #     psnr_output = batchPSNR(output, clean, 1.)
    #     score_ssim = batchSSIM(output, clean, win_size=3, multichannel=True)
    #
    #     print('[{}/{}]'.format(i+1, len(test_dataloader)), 'psnr_input:%.4f'%psnr_input, 'psnr_out:%.4f'%psnr, 'psnr_output:%.4f'%psnr_output)
    #     avg_psnr_input += psnr_input
    #     avg_psnr_output += psnr_output
    #     avg_ssim_output += score_ssim
    #
    #     # image = make_grid(clean.cpu(), nrow=args.batch_size, pad_value=1)
    #     # denoised_image = make_grid(denoised_image.cpu(), nrow=args.batch_size, pad_value=1)
    #     # save_image(image, os.path.join(args.output_dir,
    #     #                            'gt_image_{}.png'.format(i)))
    #     # save_image(denoised_image, os.path.join(args.output_dir,
    #     #                                         'denoised_image_{}.png'.format(i)))
    #     # assert 0
    #
    # avg_psnr_input /= (i + 1)
    # avg_psnr_output /= (i + 1)
    # avg_ssim_output /= (i + 1)
    # avg_time /= (i + 1)
    # print('PSNR input:%.4f'%avg_psnr_input, 'PSNR output:%.4f'%avg_psnr_output,  'SSIM output:%.4f'%avg_ssim_output, 'Avg_Time:%.4f'%avg_time)

@torch.no_grad()
def denoise(model, test_dataloader, writer, epoch, Best_PSNR, best_epoch, args):
    model.eval()
    avg_psnr_input = 0
    avg_psnr_output = 0
    avg_ssim_output = 0

    avg_time = 0
    for i, (clean, corrupted) in enumerate(test_dataloader):
        t0 = time.time()

        if args.archi=='FFDNet':
            noise = torch.zeros(clean.size())
            if args.noise_mode=='B':
                stdn = np.random.uniform(args.noiseIntL[0], args.noiseIntL[1], \
                                         size=noise.size()[0])
            elif args.noise_mode=='S':
                stdn = np.ones(noise.size()[0])*args.noise_level # For ICONIP
                # stdn = np.ones(noise.size()[0]) * np.sqrt(args.noise_level) # For Remote Sensing

            for nx in range(noise.size()[0]):
                sizen = noise[0, :, :, :].size()
                noise[nx, :, :, :] = torch.FloatTensor(sizen).normal_(mean=0, std=stdn[nx]/255) # For ICONIP
                # noise[nx, :, :, :] = torch.FloatTensor(sizen).normal_(mean=0, std=stdn[nx]) # For Remote Sensing

            corrupted = clean + noise
            # Create input Variables
            clean = Variable(clean.cuda())
            corrupted = Variable(corrupted.cuda())
            stdn_var = Variable(torch.cuda.FloatTensor(stdn))
            output = model(corrupted, stdn_var)
        else:
            clean, corrupted = clean.cuda(), corrupted.cuda()
            output = model(corrupted)

        t1 = time.time() - t0
        avg_time += t1

        psnr_input = batchPSNR(corrupted, clean, 1.)

        if args.learn_residual:  # learn noise for denoise tasks
            psnr = batchPSNR(corrupted - output, clean, 1.)
            output = torch.clamp(corrupted - output, 0., 1.)
        else:  # learn clean image
            psnr = batchPSNR(output, clean, 1.)  # learn clean image
            output = torch.clamp(output, 0., 1.)  # output clean image

        psnr_output = batchPSNR(output, clean, 1.)
        score_ssim = batchSSIM(output, clean, win_size=3, multichannel=True)

        print('Epoch:%d'%epoch, '[{}/{}]'.format(i+1, len(test_dataloader)), 'psnr_input:%.4f'%psnr_input, 'psnr_out:%.4f'%psnr, 'psnr_output:%.4f'%psnr_output)
        avg_psnr_input += psnr_input
        avg_psnr_output += psnr_output
        avg_ssim_output += score_ssim

        # image = make_grid(clean.cpu(), nrow=args.batch_size, pad_value=1)
        # noisy_image = make_grid(corrupted.cpu(), nrow=args.batch_size, pad_value=1)
        # denoised_image = make_grid(output.cpu(), nrow=args.batch_size, pad_value=1)
        # print(args.output_dir)
        # if args.denoise:
        #     save_image(image, os.path.join(args.output_dir,
        #                             args.dataset+'_gt_image.png')) # 'gt_image_{}.png'.format(i)
        #     save_image(noisy_image, os.path.join(args.output_dir,
        #                             args.dataset+'_'+args.noise_mode+str(args.noise_level)+'_corrupted_image.png')) # 'corrupted_image_{}.png'.format(i)
        #     save_image(denoised_image, os.path.join(args.output_dir,
        #                             args.dataset+'_'+args.noise_mode+str(args.noise_level)+'_'+args.archi+'_denoised_image.png')) # 'denoised_image_{}.png'.format(i)
        # if args.decompression:
        #     image = image[[2, 1, 0], :, :]
        #     noisy_image = noisy_image[[2, 1, 0], :, :]
        #     denoised_image = denoised_image[[2, 1, 0], :, :]
        #     save_image(image, os.path.join(args.output_dir,'Decompression_gt_image_%d.jpg'%i))
        #     save_image(noisy_image, os.path.join(args.output_dir, 'Decompression_corrupted_image_%d.jpg'%i))
        #     save_image(denoised_image, os.path.join(args.output_dir, 'Decompression_'+args.archi+'_image_%d.jpg'%i))

        # if i==10:
        #     assert 0

    avg_psnr_input /= (i + 1)
    avg_psnr_output /= (i + 1)
    avg_ssim_output /= (i + 1)
    avg_time /= (i + 1)
    print('Denoise Epoch:%d'%epoch, 'PSNR input:%.4f'%avg_psnr_input, 'PSNR output:%.4f'%avg_psnr_output,  'SSIM output:%.4f'%avg_ssim_output, 'Avg_Time:%.4f'%avg_time)

    # write stats and save checkpoints
    if avg_psnr_output > Best_PSNR:
        Best_PSNR = avg_psnr_output
        best_epoch = epoch
        print('Saving...')
        torch.save({'epoch': epoch,
                    'global_step': args.step,
                    'state_dict': model.state_dict()},
                   os.path.join(args.output_dir, 'checkpoint_%d.pt'%epoch))
        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint_%d.pt'%epoch))
    return Best_PSNR, best_epoch


def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, writer, args):
    global best_eval_logprob
    Best_PSNR = 0
    last_epoch = 0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        last_epoch = train_epoch(model, train_dataloader, optimizer, writer, epoch, last_epoch, args)
        PSNR, best_epoch = denoise(model, test_dataloader, writer, epoch, Best_PSNR, best_epoch, args)
        Best_PSNR = PSNR
        last_epoch = max(last_epoch, best_epoch)


# --------------------
# Visualizations
# --------------------

def encode_dataset(model, dataloader):
    model.eval()

    zs = []
    attrs = []
    for i, (x,y) in enumerate(dataloader):
        print('Encoding [{}/{}]'.format(i+1, len(dataloader))) #, end='\r'
        x = x.to(args.device)
        zs_i, _ = model(x)
        zs.append(torch.cat([z.flatten(1) for z in zs_i], dim=1))
        attrs.append(y)

    zs = torch.cat(zs, dim=0)
    attrs = torch.cat(attrs, dim=0)
    print('Encoding completed.')
    return zs, attrs

def compute_dz(zs, attrs, idx):
    """ for a given attribute idx, compute the mean for all encoded z's corresponding to the positive and negative attribute """
    z_pos = [zs[i] for i in range(len(zs)) if attrs[i][idx] == +1]
    z_neg = [zs[i] for i in range(len(zs)) if attrs[i][idx] == -1]
    # dz = z_pos - z_neg; where z_pos is mean of all encoded datapoints where attr is present;
    return torch.stack(z_pos).mean(0) - torch.stack(z_neg).mean(0)   # out tensor of shape (flattened zs dim,)

def get_manipulators(zs, attrs):
    """ compute dz (= z_pos - z_neg) for each attribute """
    print('Extracting manipulators... ') #, end=' '
    dzs = 1.6 * torch.stack([compute_dz(zs, attrs, i) for i in range(attrs.shape[1])], dim=0)  # compute dz for each attribute official code multiplies by 1.6 scalar here
    print('Completed.')
    return dzs  # out (n_attributes, flattened zs dim)

def manipulate(model, z, dz, z_std, alpha):
    # 1. record incoming shapes
    z_dims   = [z_.squeeze().shape   for z_ in z]
    z_numels = [z_.numel() for z_ in z]
    # 2. flatten z into a vector and manipulate by alpha in the direction of dz
    z = torch.cat([z_.flatten(1) for z_ in z], dim=1).to(dz.device)
    z = z + dz * torch.tensor(alpha).float().view(-1,1).to(dz.device)  # out (n_alphas, flattened zs dim)
    # 3. reshape back to z shapes from each level of the model
    zs = [z_.view((len(alpha), *dim)) for z_, dim in zip(z.split(z_numels, dim=1), z_dims)]
    # 4. decode
    return model.inverse(zs, z_std=z_std)[0]

def load_manipulators(model, args):
    # construct dataloader with limited number of images
    args.mini_data_size = 30000
    # load z manipulators for each attribute
    if os.path.exists(os.path.join(args.output_dir, 'z_manipulate.pt')):
        z_manipulate = torch.load(os.path.join(args.output_dir, 'z_manipulate.pt'), map_location=args.device)
    else:
        # encode dataset, compute manipulators, store zs, attributes, and dzs
        dataloader = fetch_dataloader(args, train=True)
        zs, attrs = encode_dataset(model, dataloader)
        z_manipulate = get_manipulators(zs, attrs)
        torch.save(zs, os.path.join(args.output_dir, 'zs.pt'))
        torch.save(attrs, os.path.join(args.output_dir, 'attrs.pt'))
        torch.save(z_manipulate, os.path.join(args.output_dir, 'z_manipulate.pt'))
    return z_manipulate

@torch.no_grad()
def visualize(model, args, attrs=None, alphas=None, img_path=None, n_examples=1):
    """ manipulate an input image along a given attribute """
    dataset = fetch_dataloader(args, train=False).dataset  # pull the dataset to access transforms and attrs
    # if no attrs passed, manipulate all of them
    if not attrs:
        attrs = list(range(len(dataset.attr_names)))
    # if image is passed, manipulate only the image
    if img_path:
        from PIL import Image
        img = Image.open(img_path)
        x = dataset.transform(img)  # transform image to tensor and encode
    else:  # take first n_examples from the dataset
        x, _ = dataset[0]
    z, _ = model(x.unsqueeze(0).to(args.device))
    # get manipulors
    z_manipulate = load_manipulators(model, args)
    # decode the varied attributes
    dec_x =[]
    for attr_idx in attrs:
        dec_x.append(manipulate(model, z, z_manipulate[attr_idx].unsqueeze(0), args.z_std, alphas))
    return torch.stack(dec_x).cpu()


# --------------------
# Main
# --------------------
if __name__ == '__main__':
    args = parser.parse_args()
    args.step = 0  # global step
    print(os.path.splitext(__file__)[0].split('/')[-1])
    print('output_dir:', args.output_dir)

    writer = None  # init as None in case of multiprocessing; only main process performs write ops
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data; sets args.input_dims needed for setting up the model
    Dataset_List = ['celeba', 'flower', 'bird'] # Synthetic Datasets for Denoising; Impainting; JPEG decompression; , 'RICE1', 'RICE2'
    if args.dataset in Dataset_List:
        if args.denoise:
            args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else os.path.join(args.output_dir, args.archi + '_' + 'Denoise_'+ args.dataset + '_' + args.noise_mode + '_' + str(int(args.noise_level)) + '_' + time.strftime('%Y-%m-%d_%H-%M-%S',time.gmtime()))
        elif args.inpainting:
            args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else os.path.join(args.output_dir, args.archi + '_' + 'Inpainting_'+ args.inpainting_mode + '_' + args.dataset + '_' + time.strftime('%Y-%m-%d_%H-%M-%S',time.gmtime()))

        train_dataloader = fetch_dataloader(args, train=True)
        test_dataloader = fetch_dataloader(args, train=False)

    else: # Real Datasets; JPEG Decompression
        if args.denoise:
            args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else os.path.join(args.output_dir, args.archi + '_' + 'Denoise_'+args.train_dataPath.split('/')[-1] + '_' + time.strftime('%Y-%m-%d_%H-%M-%S',time.gmtime()))
        elif args.decompression:
            args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else os.path.join(args.output_dir, args.archi + '_' +'Decompression_'+ args.train_dataPath.split('/')[-1] + '_' + args.decompression_level + '_' + time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
        DDataset = Dataset_Grad(args.train_dataPath, randomCount=args.randomCount, augment=False, cropPatch=True, \
                                 cropSize=args.cropSize, real=0, denoise=args.denoise, decompression=args.decompression, \
                                decompression_level=args.decompression_level, noiseLevel=args.noise_level, radius=args.radius, \
                                noise_estimate=args.noise_estimate, dataset=args.dataset)
        train_dataloader = DataLoader(dataset=DDataset, num_workers=0, drop_last=True, batch_size=args.batch_size, shuffle=True)
        print('Train set:', len(train_dataloader))
        VDataset = Dataset_Grad(args.test_dataPath, randomCount=1, augment=0, cropPatch=0, \
                                 cropSize=args.cropSize, real=0, denoise=args.denoise, decompression=args.decompression, \
                                decompression_level=args.decompression_level, noiseLevel=args.noise_level, radius=args.radius, \
                                noise_estimate=args.noise_estimate,  dataset=args.dataset)
        test_dataloader = DataLoader(dataset=VDataset, num_workers=0, batch_size=args.batch_size//2, shuffle=False)

    print('output_dir:', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # load model
    if args.archi=='DnCNN':
        model = DnCNN(channels=3, num_of_layers=40)
    elif args.archi == 'FFDNet':
        model = FFDNet(num_input_channels=3)
    elif args.archi == 'ARCNN':
        model = ARCNN()
    elif args.archi=='Unet':
        model = UNet(in_channels=3, out_channels=3, depth=4, wf=64, slope=0.2)
    elif args.archi == 'IRAE':
        model = IRAE(args)
    print('Net archi:', args.archi)

    device_ids = list(range(args.device_ids))
    print('device_ids', device_ids)

    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

    # For GAE only
    # model.inverse = model.module.inverse

    # Calculate the model size
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('params:', params)

    # load optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # args.milestones = [10, 20, 25, 30, 35, 40, 45, 50]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)

    if args.restore_checkpoint:
        model_checkpoint = torch.load(args.restore_checkpoint, map_location=args.device)
        # model.load_state_dict(model_checkpoint['state_dict'])
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in model_checkpoint['state_dict'].items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        optimizer.load_state_dict(torch.load(args.restore_optimizer, map_location=args.device))
        if args.mannual_set_lr:
            for param_group in optimizer.param_groups: # Don't use the pretrained learining rate
                param_group["lr"] = args.lr

    # setup writer and outputs
    # writer = SummaryWriter(log_dir = args.output_dir)

    # save settings
    config = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__)) + \
             'Num trainable params: {:,.0f}\n\n'.format(sum(p.numel() for p in model.parameters())) + \
             'Model:\n{}'.format(model)
    config_path = os.path.join(args.output_dir, 'config.txt')
    # writer.add_text('model_config', config)
    if not os.path.exists(config_path):
        with open(config_path, 'a') as f:
            print(config, file=f)

    if args.train:
        print('args.train', args.train)
        # train_and_evaluate(model, model1, model2, train_dataloader, test_dataloader, optimizer, writer, args)
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, writer, args)

    if args.denoise:
        print('args.denoise', args.denoise)
        denoise(model, test_dataloader, writer, 0, 0, 0, args)

    if args.decompression:
        print('args.decompression', args.decompression)
        denoise(model, test_dataloader, writer, 0, 0, 0, args)

    if args.generate:
        print('args.generate', args.generate)
        generate(model, test_dataloader)
    # writer.close()