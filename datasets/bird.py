import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy
from scipy import io
import glob
from utils import addNoise, random_bbox, mask_image

class Bird(Dataset):
    def __init__(self, args, root, train=True, transform=None, mini_data_size=None):
        self.img_folder = 'images'
        self.img_names = 'images.txt'
        self.partition_file = 'train_test_split.txt'
        # self.root = root
        self.root = os.path.join(os.path.expanduser(root), self.__class__.__name__)
        self.transform = transform
        self.args = args

        files = open(os.path.join(self.root, self.img_names), 'r')
        file_list=[]
        for file in files:
            file_name=file.split(' ')[1].split('\n')[0]
            file_list.append(file_name)
        self.file_list = file_list

        splits = open(os.path.join(self.root, self.partition_file), 'r')
        id_list=[]
        for line in splits:
            id = int(line.split(' ')[0])
            label = int(line.split(' ')[1])
            if train and label: # Train set
                id_list.append(id)
            elif (not train) and (not label): # Test set
                id_list.append(id)

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
        file_name = self.file_list[self.data_idx[idx]-1]
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