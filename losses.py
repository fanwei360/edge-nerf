import torch
from torch import nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        targets = targets.squeeze(0)

        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)
        return self.coef * loss


class MSELoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        targets = targets.squeeze(0)

        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)
        return self.coef * loss


class RGB_density_consistency(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs):
        rgbs_c = inputs['spacial_rgbs_coarse'].view(-1, 1)
        sigmas_c = inputs['edge_sigmas_coarse'].view(-1, 1)
        loss_total = self.loss(rgbs_c, sigmas_c)
        if 'spacial_rgbs_fine' in inputs and 'edge_sigmas_fine' in inputs:
            rgbs_f = inputs['spacial_rgbs_fine'].view(-1, 1)
            sigmas_f = inputs['edge_sigmas_fine'].view(-1, 1)
            loss_total += self.loss(rgbs_f, sigmas_f)
        return self.coef * loss_total

class Edge_density_consistency(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs):
        edge_sems = inputs['spacial_sem_ret_coarse'].view(-1, 1)
        edge_sigmas = inputs['spacial_e_density_coarse'].view(-1, 1)
        loss_total = self.loss(edge_sems, edge_sigmas)
        if 'spacial_sem_ret_fine' in inputs and 'spacial_e_density_fine' in inputs:
            edge_sems_f = inputs['spacial_sem_ret_fine'].view(-1, 1)
            edge_sigmas_f = inputs['spacial_e_density_fine'].view(-1, 1)
            loss_total += self.loss(edge_sems_f, edge_sigmas_f)
        return self.coef * loss_total

class Adaptive_MSELoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='none')

    def get_mask_mse(self, rgbs_tensor):
        # print(torch.max(rgbs_tensor), torch.min(rgbs_tensor), torch.mean(rgbs_tensor))
        thresh = 0.3  # threshold η
        num_positive = (torch.sum(rgbs_tensor > thresh)).float()    # +1 to avoid 0 (no gradient)
        num_negative = (torch.sum(rgbs_tensor <= thresh)).float()
        # print("num_positive:", num_positive, "num_negative:", num_negative)
        mask = torch.zeros_like(rgbs_tensor)

        mask[rgbs_tensor > thresh] = 1.0 * (num_negative + 1) / (num_positive + num_negative)
        mask[rgbs_tensor <= thresh] = 1.0 * (num_positive + 1) / (num_positive + num_negative)
        # print(mask, mask.shape)
        return mask

    def forward(self, inputs, targets):
        mask = self.get_mask_mse(targets)

        targets = targets.squeeze(0)
        loss_coarse = self.loss(inputs['edge_map_coarse'], targets)
        loss_total = (loss_coarse * mask).mean()

        if 'rgb_fine' in inputs:
            loss_fine = self.loss(inputs['edge_map_fine'], targets)
            loss_total += (loss_fine * mask).mean()
        return self.coef * loss_total


class Sparsity_Loss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.HuberLoss(reduction='mean', delta=0.6)

    def get_mask_ray(self, edges_tensor):
        mask = torch.zeros_like(edges_tensor)
        mask[edges_tensor <= 0.3] = 1
        return mask

    def forward(self, inputs, edges):
        mask = self.get_mask_ray(edges)
        sigmas_c = inputs['spacial_e_density_coarse']
        mask_c = mask.repeat(1, sigmas_c.shape[1])  # batch_size * n_samples  (1024 * 64)
        sigmas_c = sigmas_c.view(-1, 1)
        mask_c = mask_c.view(-1, 1)
        loss_coarse = torch.log(1 + torch.square(sigmas_c) / 0.5)
        loss_total = (loss_coarse * mask_c).mean()

        if 'spacial_e_density_fine' in inputs:
            sigmas_f = inputs['spacial_e_density_fine']
            mask_f = mask.repeat(1, sigmas_f.shape[1])
            sigmas_f = sigmas_f.view(-1, 1)
            mask_f = mask_f.view(-1, 1)
            loss_fine = torch.log(1 + torch.square(sigmas_f) / 0.5)
            loss_total += (loss_fine * mask_f).mean()

        return self.coef * loss_total

class Depth_Consistency_Loss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='none')
        # self.huber_loss = nn.HuberLoss(reduction='mean', delta=1)

    def get_mask_mse(self, rgbs_tensor):
        thresh = 0.3  # threshold η
        mask = torch.zeros_like(rgbs_tensor)
        mask[rgbs_tensor > thresh] = 1
        return mask

    def forward(self, inputs, edge_tensor):
        loss_coarse = self.loss(inputs['depth_coarse'], inputs['edge_depth_coarse'])
        loss_total = (loss_coarse * edge_tensor).mean()

        if 'rgb_fine' in inputs:
            loss_fine = self.loss(inputs['depth_fine'], inputs['edge_depth_fine'])
            loss_total += (loss_fine * edge_tensor).mean()
        return self.coef * loss_total

    def forward(self, inputs):
        loss_total = 1/self.cul_var(inputs['spacial_e_density_coarse']).mean()
        if 'spacial_e_density_fine' in inputs:
            loss_total += 1/self.cul_var(inputs['spacial_e_density_fine']).mean()
        return self.coef * loss_total

class Depth_mask_loss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, masks):
        loss = ((inputs['depth_coarse'] + inputs['depth_fine']).unsqueeze(1) * masks).mean()
        return loss * self.coef

loss_dict = {'l1': L1Loss,
             'mse': MSELoss,
             'edge_density_consistency': Edge_density_consistency,
             'adaptive_mse': Adaptive_MSELoss,
             'sparsity': Sparsity_Loss}
             # 'depth_consistency': Depth_Consistency_Loss,
             # 'depth_mask': Depth_mask_loss}
