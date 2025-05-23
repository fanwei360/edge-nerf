import torch
from kornia.losses import ssim as dssim
import lpips
from piq import ssim

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value *= valid_mask  # [n*1]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim_fn(image_pred, image_gt):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    # dssim_ = dssim(torch.tensor(image_pred), torch.tensor(image_gt), 3, reduction) # dissimilarity in [0, 1]
    # return 1-2*dssim_ # in [-1, 1]

    ssim_value = ssim(image_pred, image_gt)
    return ssim_value

def lpips_fn(image_pred, image_gt):
    lpips_model = lpips.LPIPS(net="alex")
    distance = lpips_model(image_pred, image_gt)
    return distance.item()