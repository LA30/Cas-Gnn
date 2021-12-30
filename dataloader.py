import os

import numpy as np
import PIL.Image
from PIL import Image
#import scipy.io as sio
import torch
from torch.utils import data


class Data(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, transform=False):
        super().__init__()
        self.root = root
        self._transform = transform

        img_root = os.path.join(self.root, 'test_images')
        depth_root = os.path.join(self.root, 'test_depth')
        mask_root = os.path.join(self.root, 'test_masks')
        
        file_names = os.listdir(img_root)
        self.img_names = []
        self.names = []
        self.depth_names = []
        self.mask_names = []

        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.names.append(name[:-4])
            self.depth_names.append(
                #os.path.join(depth_root, name[:-4]+'_depth.png')        # Test RGBD135 dataset
                os.path.join(depth_root, name[:-4] + '.png')
            )
            self.mask_names.append(
                os.path.join(mask_root, name[:-4]+'.png')
            )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img_size = img.size
        # img = img.resize((256, 256))

        # load label
        mask_file = self.mask_names[index]
        mask = PIL.Image.open(mask_file)

        # load focal
        depth_file = self.depth_names[index]
        depth = PIL.Image.open(depth_file)
        # depth = depth.resize(256, 256)

        img = np.array(img, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)
        mask[mask != 0] = 1
        depth = np.array(depth, dtype=np.uint8)
        
        if self._transform:
            img, focal = self.transform(img, depth)
            return img, focal, mask, self.names[index], img_size
        else:
            return img, depth, mask, self.names[index], img_size

    def transform(self, img, depth):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        depth = depth.astype(np.float64)/255.0
        depth = torch.from_numpy(depth).float()

        return img, depth
