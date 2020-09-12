import csv
import torch
import h5py
import os
import os.path
import numpy as np
import random
import torch
import cv2
import glob
import math
import scipy.misc
import torch.utils.data as udata
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from os import listdir
from os.path import join
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
# from skimage.measure.simple_metrics import _assert_compatible, _as_floats
from skimage.measure.simple_metrics import compare_psnr
from skimage.util.dtype import dtype_range
from skimage.measure import compare_ssim as ssim
import scipy.stats as ss
# from pybm3d.bm3d import bm3d
# from models.models import *
# import pyro.distributions as D
import torch.nn.functional as F

from torch.autograd import Function as autoF
from scipy.special import gammaln

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

class LogGamma(autoF):
    '''
    Implement of the logarithm of gamma Function.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            input_np = input.detach().cpu().numpy()
        else:
            input_np = input.detach().numpy()
        out = gammaln(input_np)
        out = torch.from_numpy(out).to(device=input.device).type(dtype=input.dtype)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.digamma(input) * grad_output

        return grad_input

# estimate the sigma map for caculating the belta
def sigma_estimate(im_noisy, im_gt, win, sigma_spatial):
    # im_noisy, im_gt = im_noisy.cpu().numpy(), im_gt.cpu().numpy()
    noise2 = (im_noisy - im_gt) ** 2
    # print('noise2', noise2.shape)
    sigma2_map_est = cv2.GaussianBlur(noise2, (win, win), sigma_spatial)
    sigma2_map_est = sigma2_map_est.astype(np.float32)
    sigma2_map_est = np.where(sigma2_map_est < 1e-10, 1e-10, sigma2_map_est)
    if sigma2_map_est.ndim == 2:
        sigma2_map_est = sigma2_map_est[:, :, np.newaxis]
    return sigma2_map_est

# def addNoise(img, sigma=12, ifInhomogeneous=False, mu=0):
#     h, w, c = img.shape
#     noise = np.random.normal(mu, sigma/255, (h, w, c))
#     noise = np.transpose(noise, (2, 0, 1))
#     if ifInhomogeneous:
#         noise = np.multiply(noise,levelOfNoiseDistribution(img))
#     if (img[1]==img[0]).all():
#         noise[1], noise[2] = noise[0], noise[0]
#     noise = np.transpose(noise, (1, 2, 0))
#     return noise + img

# def addNoise(img, sigma=12, mode='S', mu=0):
#     # print('img.shape', img.shape)
#     if mode == 'S':
#         noise = np.random.normal(mu, sigma/255, img.shape)
#     if mode == 'B':
#         noiseL_B = [0, 55]
#         stdN = np.random.uniform(noiseL_B[0], noiseL_B[1])
#         # print('stdN', stdN)
#         noise = np.random.normal(mu, stdN / 255, img.shape)
#     # if len(img.shape)==4:
#     #     noise = np.transpose(noise, (2, 0, 1, 3))
#     #     if (img[1]==img[0]).all():
#     #         noise[1], noise[2] = noise[0], noise[0]
#     #     noise = np.transpose(noise, (1, 2, 0, 3))
#     #
#     # if len(img.shape)==3:
#     #     noise = np.transpose(noise, (2, 0, 1))
#     #     if (img[1]==img[0]).all():
#     #         noise[1], noise[2] = noise[0], noise[0]
#     #     noise = np.transpose(noise, (1, 2, 0))
#     return noise + img

def addNoise(img, sigma=12, mode='S', mu=0):
    if mode == 'S':
        noise = torch.normal(mean=torch.tensor(mu), std=float(sigma)/255, size=img.shape)
    if mode == 'B':
        noiseL_B = [1, 55]
        sigma = torch.randint(low=noiseL_B[0], high=noiseL_B[1], size=(1,))
        noise = torch.normal(mean=torch.tensor(mu), std=float(sigma)/255, size=img.shape)
    return noise + img

def addNoise_RemoteSensing(img, sigma=0.001, mode='S', mu=0):
    if mode == 'S':
        noise = np.random.normal(mu, np.sqrt(sigma), size=img.shape)
    if mode == 'B':
        noiseL_B = [1, 55]
        sigma = np.random.randint(low=noiseL_B[0], high=noiseL_B[1], size=(1,))
        noise = np.random.normal(mu, np.sqrt(sigma), size=img.shape)
    return noise + img

def levelOfNoiseDistribution(img):
    h, w, _ = img.shape
    xi, yi = np.linspace(-w/2, w/2, w), np.linspace(-h/2, h/2, h)
    xi, yi = np.meshgrid(xi, yi)
    distribution = noiseSpatialDistributionFunction(xi, yi)/8.72
    #np.savetxt('this.txt', distribution)
    return distribution

def noiseSpatialDistributionFunction(xi, yi):
    f = np.exp(-0.0016 * (pow(xi, 2) + pow(yi, 2)) + 3)
    return f

def display_transform():
    return Compose([
        ToPILImage(),
        #Resize(400),
        #CenterCrop(400),
        ToTensor()
    ])

def weights_init_kaiming(m):
    # classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d): #classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear): #classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d): #classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def weights_init_onetoone(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, -100)
    elif isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 1)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def completePathImportFile(dataPath):
    files = glob.glob(os.path.join(dataPath, '*.jpg'))
    files.sort()
    return files
    
def tensorVariableNumpyImage(img):
    img = normalize(np.float32(img[:2]))
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, 0)
    img = torch.Tensor(img)
    imgCuda = Variable(img.cuda())
    return imgCuda


def dataAug(image, mode):
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    return out

def normalize(data):
    #normScale = float(1/(data.max()-data.min()))
    #x = data * normScale
    return data/255.

def patchCrop(img, scales, cropSize, stride=1):
    endc = img.shape[2]
    imgPatNum = patchCount(img, scales, cropSize, stride)
    Y = np.zeros(shape=(cropSize, cropSize, endc, imgPatNum))#, np.float32
    for k in range(len(scales)):
        img = cv2.resize(img, (int(img.shape[1]*scales[k]), int(img.shape[0]*scales[k])), interpolation=cv2.INTER_CUBIC)
        endw = img.shape[0]
        endh = img.shape[1]
        col_n = (endw - stride*2)//cropSize
        row_n = (endh - stride*2)//cropSize

        for i in range(col_n):
            for j in range(row_n):
                patch = img[stride+cropSize*i:stride+cropSize*(i+1), stride+cropSize*j:stride+cropSize*(j+1),:]
                Y[:,:,:,k] = patch
                k = k + 1
    return Y

def patchCount(img, scales, cropSize, stride=1):
    imgPatNum = 0
    for k in range(len(scales)):
        img = cv2.resize(img, (int(img.shape[1]*scales[k]), int(img.shape[0]*scales[k])), interpolation=cv2.INTER_CUBIC)
        endw = img.shape[0]
        endh = img.shape[1]
        col_n = (endw - stride*2)//cropSize
        row_n = (endh - stride*2)//cropSize
        imgPatNum += row_n * col_n
    return imgPatNum

def imgSaveTensor(img, outf, imgName):
    imgClamp = torch.clamp(img, 0., 1.)
    img= imgClamp[0].cpu()
    img= img.detach().numpy().astype(np.float32)*255
    img = np.transpose(img, (1, 2, 0))
    cv2.imwrite(os.path.join(outf, imgName), img)
    return imgClamp

def displayImage(I_A, I_B, I_C):
    fig = plt.figure()
    I_A = I_A[0,:,:].cpu()
    I_A = I_A[0].numpy().astype(np.float32)
    I_B= I_B[0,:,:].cpu()
    I_B= I_B[0].numpy().astype(np.float32)
    I_C= I_C[0,:,:].cpu()
    I_C= I_C[0].numpy().astype(np.float32)

    ax = plt.subplot("131")
    ax.imshow(I_A, cmap='gray')
    ax.set_title("Ground truth")

    ax = plt.subplot("132")
    ax.imshow(I_B, cmap='gray')
    ax.set_title("Input")

    ax = plt.subplot("133")
    ax.imshow(I_C, cmap='gray')
    ax.set_title("Model output")
    plt.show()

    return None


def batchPSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    # print('Img.shape[0]', Img.shape[0])
    for i in range(Img.shape[0]):
        tmp = compare_psnr(Iclean[i], Img[i], data_range=data_range)
        # print('%d_PSNR'%i, '%.4f'%tmp)
        PSNR += tmp
    # print('PSNR：%.4f'%PSNR, 'shape:%d'%Img.shape[0])
    return (PSNR/Img.shape[0])

def batchSSIM(img, imclean, win_size, multichannel=True):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    # print('Img.shape[0]', Img.shape[0])
    for i in range(Img.shape[0]):
        tmp = ssim(Iclean[i], Img[i], win_size=win_size, multichannel=multichannel)
        SSIM += tmp
    return (SSIM/Img.shape[0])

def img_gradient(img):
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    a = torch.from_numpy(a).float().unsqueeze(0)
    a = torch.stack((a, a, a))
    conv1.weight = nn.Parameter(a, requires_grad=False)
    conv1 = conv1.cuda()
    G_x = conv1(img)

    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    b = torch.from_numpy(b).float().unsqueeze(0)
    b = torch.stack((b, b, b))
    conv2.weight = nn.Parameter(b, requires_grad=False)
    conv2 = conv2.cuda()
    G_y = conv2(img)

    return G_x, G_y

def img_gradient_total(img):
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    a = torch.from_numpy(a).float().unsqueeze(0)
    a = torch.stack((a, a, a))
    conv1.weight = nn.Parameter(a, requires_grad=False)
    conv1 = conv1.cuda()
    G_x = conv1(img)

    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    b = torch.from_numpy(b).float().unsqueeze(0)
    b = torch.stack((b, b, b))
    conv2.weight = nn.Parameter(b, requires_grad=False)
    conv2 = conv2.cuda()
    G_y = conv2(img)

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
    return G

def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf

def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0,bb]
    return sigma


def PatchSNR(gt, noise):
    var_p = gt.var()
    var_n = noise.var()
    if var_n < 1e-10:
        var_n = 1e-10
    out = np.sqrt(var_p/var_n)
    return out

def batch_PatchSNR(imclean, noise):
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    Inoise = noise.data.cpu().numpy().astype(np.float32)
    patch_snr = np.zeros(Iclean.shape[0])

    for i in range(Iclean.shape[0]):
        patch_snr[i] = PatchSNR(Iclean[i], Inoise[i])
        # print('%d_PatchSNR'%i, '%.4f'%patch_snr[i])
    patch_snr = np.where((patch_snr>0.45), 0.5, 0)
    return torch.tensor(patch_snr, dtype=torch.float).cuda()

def PatchSNR_gradient_loss(imgClear, imgNoisy, imgDenoised):
    imgDiff = imgNoisy - imgClear
    patch_snr = batch_PatchSNR(imgClear, imgDiff)

    clear_x, clear_y = img_gradient(imgClear)
    x_grad, y_grad = img_gradient(imgDenoised)

    horizontal_loss = torch.abs(x_grad - clear_x).view(x_grad.shape[0], -1).sum(1)
    vertical_loss = torch.abs(y_grad - clear_y).view(y_grad.shape[0], -1).sum(1)
    gradient_loss = patch_snr * (horizontal_loss + vertical_loss)
    gradient_loss = gradient_loss.sum(0)
    return gradient_loss

def Gradient_Loss(imgClear, imgDenoised, Loss_criterion):
    clear_x, clear_y = img_gradient(imgClear)
    x_grad, y_grad = img_gradient(imgDenoised)
    gradient_loss = Loss_criterion(x_grad, clear_x) + Loss_criterion(y_grad, clear_y)
    return gradient_loss

def compare_psnr(im_true, im_test, data_range=None):
    """ Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    im_true : ndarray
        Ground-truth image.
    im_test : ndarray
        Test image.
    data_range : int
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.

    Returns
    -------
    psnr : float
        The PSNR metric.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """
    # _assert_compatible(im_true, im_test)

    if data_range is None:
        if im_true.dtype != im_test.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "im_true.")
        dmin, dmax = dtype_range[im_true.dtype.type]
        true_min, true_max = np.min(im_true), np.max(im_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "im_true has intensity values outside the range expected for "
                "its data type.  Please manually specify the data_range")
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    # im_true, im_test = _as_floats(im_true, im_test)

    # print('im_true - im_test', im_true - im_test)
    err = np.mean(np.square(im_true - im_test), dtype=np.float64)
    # print('err', err)
    if err < 1e-8:
        err = 1e-8
    # print('err', err)
    return 10 * np.log10((data_range ** 2) / err)

def generate_mask(clean_image, eps):
    grad = img_gradient_total(clean_image)
    grad = grad.cpu().numpy()
    # print('grad', grad.shape, grad.min(), grad.max())
    tmp = np.argwhere(grad>=eps)
    # print('tmp', tmp, len(tmp))
    tmp_1 = np.copy(tmp)
    tmp_1[:, 3] = tmp_1[:, 3]+1
    # print('tmp_1', tmp_1)
    # print('tmp', tmp)
    tmp_2 = np.copy(tmp)
    tmp_2[:, 3] = tmp_2[:, 3]-1
    # print('tmp_2', tmp_2)

    new = np.vstack((tmp_1, tmp, tmp_2))
    # print('new', new.shape)

    new_1 = np.copy(new)
    new_1[:, 2] = new_1[:, 2] + 1

    new_2 = np.copy(new)
    new_2[:, 2] = new_2[:, 2] - 1

    # print('new_1', new_1)
    # print('new_2', new_2)

    new_total = np.vstack((new_1, new, new_2))
    # print('new_total', new_total.shape)

    width = grad.shape[2]
    height = grad.shape[3]
    # print('shape', grad.shape)
    new_total[:, 2] = np.clip(new_total[:, 2], 0, width-1)
    new_total[:, 3] = np.clip(new_total[:, 3], 0, height-1)

    # print('new_total', new_total, new_total.min(), new_total.max())

    mask = np.zeros_like(grad)
    mask[new_total[:,0], new_total[:,1], new_total[:,2], new_total[:,3]]= 1.
    # mask = np.where((grad>=eps), 1., 0.) # texture
    complement_mask = np.ones_like(mask)-mask # smooth
    # print('mask', mask.shape, mask.sum(), mask)
    # print('complement_mask', complement_mask.shape, complement_mask.sum(), complement_mask)
    return mask, complement_mask

# For image impainting

def mask_image(image, bboxes, mask_type='hole'):
    _, height, width = image.shape # batch_size,
    max_delta_h, max_delta_w = [32, 32] #config['max_delta_shape']
    mask = bbox2mask(bboxes, height, width, max_delta_h, max_delta_w)
    if image.is_cuda:
        mask = mask.cuda()

    if mask_type == 'hole':
        result = image * (1. - mask) + mask
    elif mask_type == 'mosaic':
        # TODO: Matching the mosaic patch size and the mask size
        mosaic_unit_size = 12 # config['mosaic_unit_size']
        downsampled_image = F.interpolate(x, scale_factor=1. / mosaic_unit_size, mode='nearest')
        upsampled_image = F.interpolate(downsampled_image, size=(height, width), mode='nearest')
        result = upsampled_image * mask + image * (1. - mask)
    else:
        raise NotImplementedError('Not implemented mask type.')
    return result, mask

# def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
# ''' with batch_size version'''
#     batch_size = bboxes.size(0)
#     mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
#     for i in range(batch_size):
#         bbox = bboxes[i]
#         delta_h = np.random.randint(max_delta_h // 2 + 1)
#         delta_w = np.random.randint(max_delta_w // 2 + 1)
#         mask[i, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.
#     return mask

def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
    ''' without batch_size version'''
    mask = torch.zeros((3, height, width), dtype=torch.float32)
    bbox = bboxes[0]
    # print(bbox, bbox[3])
    delta_h = np.random.randint(max_delta_h // 2 + 1)
    delta_w = np.random.randint(max_delta_w // 2 + 1)
    mask[:, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.
    return mask

def random_bbox(image, args, mask_batch_same=True):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including img

    Returns:
        tuple: (top, left, height, width)

    """
    _, img_height, img_width = image.shape # batch_size,
    h, w = args.mask_shape
    margin_height, margin_width = [0, 0]
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []
    if mask_batch_same:
        t = np.random.randint(margin_height, maxt)
        l = np.random.randint(margin_width, maxl)
        bbox_list.append((t, l, h, w))
        bbox_list = bbox_list # * batch_size
    else:
        # for i in range(batch_size):
        t = np.random.randint(margin_height, maxt)
        l = np.random.randint(margin_width, maxl)
        bbox_list.append((t, l, h, w))

    return torch.tensor(bbox_list, dtype=torch.int64)