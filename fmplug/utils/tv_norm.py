import torch
import torch.nn as nn

class tv_lp_loss(nn.Module):
    def __init__(self,pow=2):
        super(tv_lp_loss,self).__init__()
        self.pow = pow
        if self.pow == 1:
            self.loss_func = self._l1_tv_norm
        elif self.pow == 2:
            self.loss_func = self._l2_tv_norm
        else:
            self.loss_func = self._tv_norm


    def _l1_tv_norm(self,x):
        numel = x.shape[-2] * x.shape[-1]
        pixel_dif1 = x[:, :, 1:, :] - x[:, :, :-1, :]
        pixel_dif2 = x[:, :, :, 1:] - x[:, :, :, :-1]
        tv_norm = (torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2))) / numel
        # tv_norm = (torch.sum((torch.abs(pixel_dif1)) + torch.abs(pixel_dif2))) / numel
        return tv_norm

    def _l2_tv_norm(self,x):
        numel = x.shape[-2] * x.shape[-1]
        pixel_dif1 = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)
        pixel_dif2 = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
        dif1_l2 = torch.sum(pixel_dif1)
        dif2_l2 = torch.sum(pixel_dif2)
        tv_norm = torch.sqrt(dif1_l2) + torch.sqrt(dif2_l2) / numel
        # tv_norm = torch.sum(torch.sqrt(pixel_dif1 + pixel_dif2)) / numel
        return tv_norm
    
    def _tv_norm(self, x):
        numel = x.shape[-2] * x.shape[-1]
        pixel_dif1 = torch.norm(x[:, :, 1:, :] - x[:, :, :-1, :], self.pow)
        pixel_dif2 = torch.norm(x[:, :, :, 1:] - x[:, :, :, :-1], self.pow)
        dif1_l2 = torch.sum(pixel_dif1)
        dif2_l2 = torch.sum(pixel_dif2)
        tv_norm = (dif1_l2 + dif2_l2) ** (1 / self.pow) / numel
        # tv_norm = torch.sum((pixel_dif1 + pixel_dif2) ** (1 / self.pow)) / numel
        return tv_norm

    def forward(self, x):
        return self.loss_func(x)
    