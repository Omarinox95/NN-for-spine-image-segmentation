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
import torchvision.transforms as Tr
from skimage import exposure
import random
from utils.augmentation import Data_Augmentation


class BasicDataset(Dataset):
    # Function that initialize all variables
    def __init__(self, imgs_dir1, imgs_dir2, masks_dir, scale=1,data_aug=Data_Augmentation()):
        self.imgs_dir1 = imgs_dir1
        self.imgs_dir2 = imgs_dir2
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        if isinstance(data_aug, Data_Augmentation):
            self.data_augmentation = data_aug
        else:
            self.data_augmentation = None
            
        logging.info(f'Creating dataset with {len(imgs_dir1)} examples')
    
    # define length of the lists 
    def __len__(self):
        return len(self.imgs_dir1)
        
    # equalizes images ( if want to use , only for display not processing) 
    def histogram_equalize(self, img):
        img_cdf, bin_centers = exposure.cumulative_distribution(img)

        return np.interp(img, bin_centers, img_cdf)
    # Function to z-score normalize 
    def z_score_standardization(self, img_npArray):
        new_img = img_npArray
        new_img[0,0,:,:] = (img_npArray[0,0,:,:] - np.mean(img_npArray[0,0,:,:])) / np.std(img_npArray[0,0,:,:])
        new_img[0,1,:,:] = (img_npArray[0,1,:,:] - np.mean(img_npArray[0,1,:,:])) / np.std(img_npArray[0,1,:,:])
        
        return new_img
        
    # Function to import and convert nifti images to array and then tensors 
    def __getitem__(self, i):
        mask_prim = nib.load(self.masks_dir[1])
        mask = nib.load(self.masks_dir[i])
        img = nib.load(self.imgs_dir1[i])
        img2 = nib.load(self.imgs_dir2[i])

        img = np.array(img.dataobj)
        img2 = np.array(img2.dataobj)
        mask = np.array(mask.dataobj)
        mask_prim = np.array(mask_prim.dataobj)
        
        ''' ---- code to calculate the weights for CEL and DL
        W,H = np.shape(mask_prim)
        total = W*H
        PB = np.sum(mask_prim==0)/total
        W_b = 1/PB
        PW = np.sum(mask_prim==1)/total
        W_wm = 1/PW
        PG = np.sum(mask_prim==2)/total
        W_gm = 1/PG
        Prob_total = (1/W_gm) + (1/W_wm) + (1/W_b)
        '''
        
        img = np.stack((img, img2), axis=0)
        img = np.expand_dims(img, axis=0)
        
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)

        img, mask = self.data_augmentation.run(img, mask)
        
        img = self.z_score_standardization(img)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
