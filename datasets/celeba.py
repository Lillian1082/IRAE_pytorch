import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import addNoise, random_bbox, mask_image
import glob
import random
import torchvision

class CelebA(Dataset):
    processed_file = 'processed.pt'
    partition_file = 'Eval/list_eval_partition.txt'
    attr_file = 'Anno/list_attr_celeba.txt'
    img_folder = 'Img/img_align_celeba'
    attr_names = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young'.split()

    mask_root = 'data/irregular_holes/disocclusion_img_mask'

    def __init__(self, root, args, train=True, transform=None, mask_transform=None, mini_data_size=None):
        self.root = os.path.join(os.path.expanduser(root), self.__class__.__name__)
        self.transform = transform
        self.mask_transform = mask_transform
        self.args = args
        # check if processed
        if not os.path.exists(os.path.join(self.root, self.processed_file)):
            self._process_and_save()
        data = torch.load(os.path.join(self.root, self.processed_file))

        if train:
            self.data = data['train']
        else:
            self.data = data['val']

        if self.args.inpainting_mode == 'irregular':
            self.mask_paths = glob.glob('{:s}/*.png'.format(self.mask_root))
            self.N_mask = len(self.mask_paths)

        # print('train:', len(data['train']))
        # print('val:', len(data['val']))
        # print('test:', len(data['test']))
        #
        # assert 0

        # for idx in range(len(data['val'])):
        #     filename, attr = data['val'][idx]
        #     img = Image.open(os.path.join(self.root, self.img_folder, filename))  # loads in RGB mode
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     img.save(os.path.join(self.root,'Val/%s'%filename))
        # assert 0

        if mini_data_size != None:
            self.data = self.data[:mini_data_size]

    def __getitem__(self, idx):
        filename, attr = self.data[idx]
        img = Image.open(os.path.join(self.root, self.img_folder, filename))  # loads in RGB mode
        if self.transform is not None:
            img = self.transform(img)

        # # Glow
        # attr = torch.from_numpy(attr)
        # return img, attr

        # GAE/ GDN
        if self.args.denoise:
            corrupted = addNoise(img, sigma=self.args.noise_level, mode=self.args.noise_mode)  # AWGN
            return img, corrupted

        elif self.args.inpainting:
            if self.args.inpainting_mode == 'center': # center hole inpainting
                bboxes = random_bbox(img, args=self.args)
                corrupted, mask = mask_image(img, bboxes)
            if self.args.inpainting_mode == 'irregular': # irregular hole inpainting
                while True:
                    mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
                    mask = self.mask_transform(mask.convert('RGB'))
                    mask = mask[0]
                    mask = torch.unsqueeze(mask, 0)
                    # mask = torch.unsqueeze(mask, 1)
                    mask = mask.byte().float()
                    percent = 100*float((mask == 0).sum()) / float(torch.ones_like(mask).sum())
                    if percent >= 10 and percent < 50:
                        break

                corrupted = img * mask + (1.-mask)

            # torchvision.utils.save_image(corrupted, 'try_irregular.jpg')
            # assert 0
            return img, corrupted, mask

    def __len__(self):
        return len(self.data)

    def _process_and_save(self):
        if not os.path.exists(os.path.join(self.root, self.attr_file)):
            raise RuntimeError('Dataset attributes file not found at {}.'.format(os.path.join(self.root, self.attr_file)))
        if not os.path.exists(os.path.join(self.root, self.partition_file)):
            raise RuntimeError('Dataset evaluation partitions file not found at {}.'.format(os.path.join(self.root, self.partition_file)))
        if not os.path.isdir(os.path.join(self.root, self.img_folder)):
            raise RuntimeError('Dataset image folder not found at {}.'.format(os.path.join(self.root, self.img_folder)))

        # read attributes file: list_attr_celeba.txt
        # First Row: number of images
        # Second Row: attribute names
        # Rest of the Rows: <image_id> <attribute_labels>
        with open(os.path.join(self.root, self.attr_file), 'r') as f:
            lines = f.readlines()
        n_files = int(lines[0])
        attr = [[l.split()[0], l.split()[1:]] for l in lines[2:]]  # [image_id.jpg, <attr_labels>]

        assert len(attr) == n_files, \
                'Mismatch b/n num entries in attributes file {} and reported num files {}'.format(len(attr), n_files)

        # read partition file: list_eval_partition.txt;
        # All Rows: <image_id> <evaluation_status>
        # "0" represents training image,
        # "1" represents validation image,
        # "2" represents testing image;
        data = [[], [], []]  # train, val, test
        unmatched = 0
        with open(os.path.join(self.root, self.partition_file), 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            fname, split = line.split()
            if attr[i][0] != fname:
                unmatched += 1
                continue
            data[int(split)].append([fname, np.array(attr[i][1], dtype=np.float32)])  # [image_id.jpg, <attr_labels>] by train/val/test


        if unmatched > 0: print('Unmatched partition filenames to attribute filenames: ', unmatched)
        assert sum(len(s) for s in data) == n_files, \
                'Mismatch b/n num entries in partition {} and reported num files {}'.format(sum(len(s) for s in filenames), n_files)

        # check image folder
        filenames = os.listdir(os.path.join(self.root, self.img_folder))
        assert len(filenames) == n_files, \
                'Mismatch b/n num files in image folder {} and report num files {}'.format(len(filenames), n_files)

        # save
        data = {'train': data[0], 'val': data[1], 'test': data[2]}
        with open(os.path.join(self.root, self.processed_file), 'wb') as f:
            torch.save(data, f)



if __name__ == '__main__':
    d = CelebA('~/Data/')
    print('Length: ', len(d))
    print('Image: ', d[0][0])
    print('Attr: ', d[0][1])

    import timeit
    t = timeit.timeit('d[np.random.randint(0,len(d))]', number=1000, globals=globals())
    print('Retrieval time: ', t)

    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    n_bits = 5
    t = T.Compose([T.CenterCrop(148),  # RealNVP preprocessing
                   T.Resize(64),
                   T.Lambda(lambda im: np.array(im, dtype=np.float32)),  # to numpy
                   T.Lambda(lambda x: np.floor(x / 2**(8 - n_bits)) / 2**n_bits), # lower bits
                   T.ToTensor(),
                   T.Lambda(lambda t: t + torch.rand(t.shape)/ 2**n_bits)])                     # dequantize
    d_ = CelebA('~/Data/', transform=t)
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(np.array(d[0][0]))
    axs[1].imshow(d_[0][0].numpy().transpose(1,2,0))
    plt.show()
