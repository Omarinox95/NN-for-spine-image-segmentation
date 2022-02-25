import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset 
import nibabel as nib
from Data_extraction import *

logging.getLogger().setLevel(logging.INFO)

# modified to have two images as arguments
def predict_img(net,
                full_img1,
                full_img2,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    
        
    # the two images are converted to array 
    img1 = np.array(full_img1.dataobj)
    img2 = np.array(full_img2.dataobj)
    # prepare img1 as an image 
    img_show = Image.fromarray(img1)
    
    # z-score normalizartion ( same normalization as the train model) 
    img1 = (img1 - np.mean(img1)) / np.std(img1)
    img2 = (img2 - np.mean(img2)) / np.std(img2)    
    
    # Stack the two images
    img = np.stack((img1,img2),axis=0)
    
    # Convert array to tensor and takes out the bacth dimention
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        # gives the predicted segmentation of given images in 3 channels
        output = net(img)
        
        if net.n_classes > 1:
            output = output.squeeze(0)
            probs = F.softmax(output, dim=0)

        else:
            probs = torch.sigmoid(output)
            
        full_mask = probs.cpu().data.numpy().argmax(axis=0)
        
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_show.size),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        
    return full_mask

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
                        
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filename of input 1 images', required=False)
                        
    parser.add_argument('--input2', '-i2', metavar='INPUT', nargs='+',
                        help='filename of input 2 images', required=False)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    
    # prediction into JPG 
    jpg_mask = (mask*127).astype(np.uint8)
    jpg_mask = Image.fromarray(jpg_mask)
    # prediction into Nifti
    mask_uint = mask.astype(np.uint8)
    affine = np.eye(4)
    niftiObject = nib.Nifti1Image(mask_uint,affine) 
    
    
    return  niftiObject, jpg_mask
  

if __name__ == "__main__":
    args = get_args()
    
    # add lists of the images to be segmented
     
    dirpath = "/home/mialab.team02/data_mialab/test_public"
    filename = "img1.nii.gz"
    filename_2 = "img2.nii.gz"
    empty= "NONE"
    dir_img1,dir_img2,dir_mask = extract_and_match_data_walk(dirpath,
                                                 filename,
                                                 filename_2,
                                                 empty)
                                                 
    # choose the destination of the segemented image
    out_dest ="/home/mialab.team02/Team02_code/Pytorch-UNet-master/team02_predictions"
    
    net = UNet(n_channels=2, n_classes=3)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(dir_img1):
        logging.info("\nPredicting image {} ...".format(fn))
        
        # add nifti loader for our images 
        niftimg = nib.load(dir_img1[i])
        niftimg2 = nib.load(dir_img2[i])
        
        # modified to have two images as arguments 
        mask = predict_img(net=net,
                           full_img1=niftimg,
                           full_img2=niftimg2,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
   
        if not args.no_save:
        
            out_fn = dir_img1[i].replace("img1", "prediction")
            out_fn = out_fn.replace("/", "_")
            _,out_fn = out_fn.split("_su")
            out_fn = ("su"+out_fn)
            
            # defining the paths
            saveToFilePath_nift = os.path.join(out_dest, out_fn)
            saveToFilePath_PJg = os.path.join(out_dest,"prediction.jpg")

            #create folder if not created 
            if not os.path.exists(out_dest):
              os.mkdir(out_dest)
        
            result_nift, result_jpg = mask_to_image(mask)
            
            #saving images
            nib.save(result_nift, saveToFilePath_nift)
            #result_jpg.save(saveToFilePath_PJg)
            
            logging.info("Mask saved to {}".format(out_dest))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            img_nd = np.array(niftimg.dataobj)
            img = Image.fromarray(img_nd)
            plot_img_and_mask(img, mask)
            
