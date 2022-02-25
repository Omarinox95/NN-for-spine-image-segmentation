#Import libreries 
import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval import eval_net
from unet import UNet
from torch.utils.tensorboard import SummaryWriter
from utils.dataset2 import BasicDataset
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as F
# Adding the script made to extract data
from Data_extraction import extract_and_match_data_walk
# diceloss import (nair)
from dice_loss import dice_co_nair_cl

# Direction variables
dirpath = "/home/mialab.team02/data_mialab/train_public"
filename = "img1.nii.gz"
filename_2 = "img2.nii.gz"
filename_Stru = "structure.nii.gz"

#Weight variables (first calculation)

W_b = 1.0043197956488195
W_wm = 277.17293233082705
W_gm = 1442.2535211267607
W_sum = W_b + W_wm + W_gm

#Weight variables (approach two, calculated with probabilities, added up to one)
#W_b = 0.01746
#W_wm = 0.48417
#W_gm = 0.49779

weight_Array = [W_b/W_sum, W_wm/W_sum, W_gm/W_sum]
weight_tensor = torch.FloatTensor(weight_Array)

# extracts directions of image 1 , image 2 and ground truth
dir_img1,dir_img2,dir_mask = extract_and_match_data_walk(dirpath,
                                                 filename,
                                                 filename_2,
                                                 filename_Stru)
dir_checkpoint = 'checkpoints/'  # give a checkpoint direction 


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.2,
              save_cp=True,
              img_scale=1):

    # import images from list to tensor
    dataset  = BasicDataset(dir_img1, dir_img2, dir_mask, img_scale)
    # Divides the data set in two, validation and training randomly
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    
    # Optimizer 
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    # Loss function
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor.float().cuda())
        #criterion = nn.CrossEntropyLoss(weights)
        #dice_loss = dice_co_nair(true_masks, masks_pred, weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
    
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image'].squeeze(0)
                true_masks = batch['mask'].squeeze(0).squeeze(0)
                
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                masks_pred = net(imgs)
                #loss = criterion(masks_pred, true_masks)
                
                dice_loss = dice_co_nair_cl.dice_co_nair(true_masks, masks_pred, weight_tensor.float().cuda())
                loss = 0.9*criterion(masks_pred, true_masks) - 0.1*dice_loss 
                epoch_loss += loss.item()
                # Add to the tensorboard value 
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                
                # using softmax 
                seg_mask_pred = masks_pred.squeeze()
                
                softy = nn.Softmax(dim=0)
                seg_mask_pred = softy(seg_mask_pred)
                
                pbar.update(imgs.shape[0])
                global_step += 1
                # add to values and images to tensorboard every N step
                if global_step % (400) == 0:
                    
                    seg_mask_predict = seg_mask_pred.cpu().data.numpy().argmax(axis=0)
                    if np.max(seg_mask_predict) != 0:
                      seg_mask_predict = seg_mask_predict/np.max(seg_mask_predict)
                      
                    true_masks = true_masks.cpu().data.numpy()
                    if np.max(true_masks) != 0:
                      true_masks = true_masks/np.max(true_masks)

                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)
                        
                    writer.add_images('images_1', imgs[batch_size-1,0,:,:], global_step,dataformats='HW')
                    writer.add_images('images_2', imgs[batch_size-1,1,:,:], global_step,dataformats='HW')
                    
                    writer.add_images('ground_truth', true_masks[batch_size-1,:,:], global_step,dataformats='HW')
                    writer.add_images('output_truth', seg_mask_predict[:,:], global_step,dataformats='HW')
                    
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        
        writer.close()
            
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if (epoch+1) % 5 == 0:    
                torch.save(net.state_dict(),dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=60,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=2, n_classes=3)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
