import torch
from torch.autograd import Function
import torch.nn as nn
#from tensorflow.keras import backend as K


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
    

#my take on the dice loss
class dice_co_nair_cl():
      
  #generate a zero image , and add a class to this imge have a zero image(0, 1) for each class; one hot vecor encoding
  def dice_co_nair(true_m, pred_m, weight_tensor, smooth = 0.01):
    num_classes = 3
    
    true_mask = nn.functional.one_hot(true_m, num_classes = 3)
    
    softy = nn.Softmax(dim=0)
    pred_mask = softy(pred_m)
    
    dice_loss = 0
    #b, x, y ,c -> b, c, x ,y
    true_mask = true_mask.permute(0 ,3, 1, 2)
    
    for c in range(num_classes):
      true_flat = true_mask[:, c].view(-1)
      pred_flat = pred_mask[:, c].view(-1)
    
      intersection = (pred_flat * true_flat).sum()
    
      sum_true = torch.sum(true_flat)
      sum_pred = torch.sum(pred_flat)
    
    
      w = weight_tensor[c]
      dice_loss += w*(1 - ((2. * intersection + smooth) /(pred_flat.sum() + true_flat.sum() + smooth)))     
    return dice_loss

