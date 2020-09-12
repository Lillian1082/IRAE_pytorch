import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy
from scipy import io
import glob
from utils import addNoise, random_bbox, mask_image

class Flower(Dataset):
    def __init__(self, args, root, train=True, transform=None, mini_data_size=None):
        self.img_folder = 'Image'
        self.partition_file = 'setid.mat'
        # self.root = root
        self.root = os.path.join(os.path.expanduser(root), self.__class__.__name__)
        self.transform = transform
        self.args = args
        # self.files = glob.glob(os.path.join(self.root, self.img_folder,'*'))
        # self.files.sort()
        partition = scipy.io.loadmat(os.path.join(self.root, self.partition_file))

        if train:
            id_list = list(np.asarray(partition['tstid']).flatten())
        else:
            id_list = []
            id_list.append(list(np.asarray(partition['trnid']).flatten()))
            id_list.append(list(np.asarray(partition['valid']).flatten()))
            id_list = [x for xs in id_list for x in xs]

        self.data_idx = id_list
        print('data_idx', len(self.data_idx))

        # for i in range(len(self.data_idx)):
        #     file_name = 'image_'+'%05d'%(self.data_idx[i])+'.jpg'
        #     img = Image.open(os.path.join(self.root, self.img_folder,file_name))  # loads in RGB mode
        #     if self.transform is not None:
        #         img = self.transform(img)
        #         img.save(os.path.join(self.root, 'Val',file_name))
        # assert 0

        if mini_data_size != None:
            self.data_idx = self.data_idx[:mini_data_size]

    def __getitem__(self, idx):
        file_name = 'image_'+'%05d'%(self.data_idx[idx])+'.jpg' # if don't added, there might be some bug, but what we have run haven't added this line
        img = Image.open(os.path.join(self.root, self.img_folder,file_name))  # loads in RGB mode
        if self.transform is not None:
            img = self.transform(img)

        if self.args.denoise:
            corrupted = addNoise(img, sigma=self.args.noise_level, mode=self.args.noise_mode)  # AWGN
        elif self.args.inpainting:
            bboxes = random_bbox(img)
            corrupted, mask = mask_image(img, bboxes)  # hole_inpainting

        return img, corrupted

    def __len__(self):
        return len(self.data_idx)



if __name__ == '__main__':
    d = Flower('../data/Flowers')