from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import nibabel as nib
from PIL import Image
import Data_extraction


class BasicDataset(Dataset):
    def __init__(self, imgs_dir1, imgs_dir2, masks_dir, scale=1):
        self.imgs_dir1 = imgs_dir1
        self.imgs_dir2 = imgs_dir2
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        #self.ids = imgs_dir1
        logging.info(f'Creating dataset with {len(imgs_dir1)} examples')

    def __len__(self):
        return len(self.imgs_dir1)

    @classmethod
    def preprocess(cls, pil_img, scale):
        # Comment the scale since we don't use it for the moment
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img.dataobj)

        # if len(img_nd.shape) == 2:
            # img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW Reading ,may be RGB only
        # img_trans = img_nd.transpose((2, 0, 1))
        # if img_trans.max() > 1:
            # img_trans = img_trans / 255

        return img_nd

    def __getitem__(self, i):
        #idx = self.ids[i]
        #mask_file = self.imgs_dir1
        #img_file1 = self.imgs_dir2
        #img_file2 = self.masks_dir

        #assert len(mask_file) == 1, \
            #f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        #assert len(img_file1) == 1, \
            #f'Either no image or multiple images found for the ID {idx}: {img_file1}'
        #assert len(img_file2) == 1, \
            #f'Either no image or multiple images found for the ID {idx}: {img_file2}'

        mask = nib.load(self.masks_dir[i])
        img = nib.load(self.imgs_dir1[i])
        img2 = nib.load(self.imgs_dir2[i])

        #assert img.size == mask.size, \
            #f'Image and mask {i} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        img2 = self.preprocess(img2, self.scale)
        mask = self.preprocess(mask, self.scale)
        # Appending two images   
        inimg = np.append(np.expand_dims(img,axis=0),np.expand_dims(img2,axis=0),axis=0)


        return {'image': torch.from_numpy(inimg), 'mask': torch.from_numpy(mask)}
