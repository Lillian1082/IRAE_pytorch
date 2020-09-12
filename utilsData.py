import os.path
from utils import *

class Dataset_Grad(Dataset):
    def __init__(self, path, **kwargs):
        super(Dataset_Grad, self).__init__()
        self.real = kwargs['real'] # Real or Synthetic DataSet
        self.denoise = kwargs['denoise']
        self.decompression = kwargs['decompression']
        self.decompression_level = kwargs['decompression_level']
        self.crop = kwargs['cropPatch']
        self.cropSize = kwargs['cropSize']
        self.sigma_spatial = kwargs['radius']
        self.win = 2 * self.sigma_spatial + 1
        self.randomCount = kwargs['randomCount']
        self.augment = kwargs['augment']
        self.noiseLevel = kwargs['noiseLevel']
        self.noise_estimate = kwargs['noise_estimate'] # whether to estimate the sigma map

        if kwargs['dataset']=='UCMerced_LandUse':
            self.gt_folders = glob.glob(os.path.join(path, '*'))
            self.gt_files = []
            for i in range(len(self.gt_folders)):
                self.gt_files += glob.glob(os.path.join(self.gt_folders[i], '*'))
        else:
            self.gt_files = glob.glob(os.path.join(path, 'GT', '*.*'))

        self.gt_files.sort()

        if self.real:
            if self.denoise:
                self.noisy_files = glob.glob(os.path.join(path, 'Noisy', '*.*')) # Denoise
                self.noisy_files.sort()
                # self.noisy_files = glob.glob(os.path.join(path, 'Noisy_S%d'%self.noiseLevel, '*.*'))
        if self.decompression:
            self.noisy_files = glob.glob(os.path.join(path, 'JPEG', self.decompression_level, '*.*')) # Image JPEG Decompression
            self.noisy_files.sort()

        print(len(self.gt_files))
        print('real', self.real)
        print('aug', self.augment)
        print('crop', self.crop)

    def __getitem__(self, index):
        # print(index)
        # print(self.gt_files[index // self.randomCount])
        if self.gt_files[index // self.randomCount].split('/')[-1][-4:]=='.npy':
            image = np.load(self.gt_files[index // self.randomCount])
        else:
            image = cv2.imread(self.gt_files[index // self.randomCount]) # color image
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # greyscale image
            image = normalize(image)
            # image = np.expand_dims(image, axis=2)

        # gt = np.random.randint(0, 10)
        # print(gt)
        # if (gt==0):
        #     noisy_image = image
        # elif
        if self.real:
            if self.gt_files[index // self.randomCount].split('/')[-1][-4:] == '.npy':
                noisy_image = np.load(self.noisy_files[index // self.randomCount])
            else:
                noisy_image = cv2.imread(self.noisy_files[index // self.randomCount])  # color image
                # noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)  # greyscale image
                noisy_image = normalize(noisy_image)
                # noisy_image = np.expand_dims(noisy_image, axis=2) # greyscale image
        elif self.decompression:
            noisy_image = cv2.imread(self.noisy_files[index // self.randomCount])  # color image
            noisy_image = normalize(noisy_image)
        else:
            # noisy_image = addNoise(image, self.noiseLevel)
            noisy_image = addNoise_RemoteSensing(image, self.noiseLevel)
        if self.crop:
            endw, endh = image.shape[0], image.shape[1]
            assert (endw >= self.cropSize) and (endh >= self.cropSize)
            x = np.random.randint(0, endw - self.cropSize)
            y = np.random.randint(0, endh - self.cropSize)
            image = image[x:(self.cropSize + x), y:(self.cropSize + y), :]
            noisy_image = noisy_image[x:(self.cropSize + x), y:(self.cropSize + y), :]
        if self.augment:
            def _augment(img, noisy_img):
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                rot90 = random.random() < 0.5
                if hflip:
                    img = img[:, ::-1, :]
                    noisy_img = noisy_img[:, ::-1, :]
                if vflip:
                    img = img[::-1, :, :]
                    noisy_img = noisy_img[::-1, :, :]
                if rot90:
                    img = img.transpose(1, 0, 2)
                    noisy_img = noisy_img.transpose(1, 0, 2)
                return img, noisy_img
            image, noisy_image = _augment(image, noisy_image)

        if self.noise_estimate:
            sigma2_map_est = sigma_estimate(noisy_image, image, self.win, self.sigma_spatial)

        image = np.moveaxis(image, -1, 0)
        noisy_image = np.moveaxis(noisy_image, -1, 0)
        image = torch.from_numpy(np.copy(image))
        noisy_image = torch.from_numpy(np.copy(noisy_image))

        if self.noise_estimate:
            sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2, 0, 1)))
            return (image.type(torch.FloatTensor), noisy_image.type(torch.FloatTensor), sigma2_map_est)
        else:
            return (image.type(torch.FloatTensor), noisy_image.type(torch.FloatTensor))

    def __len__(self):
        return len(self.gt_files) * self.randomCount