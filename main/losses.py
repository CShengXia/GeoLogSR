import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SSIM1D(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM1D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        # Create a 1D Gaussian window
        self.register_buffer('window', self._create_window(window_size))
    def _create_window(self, window_size):
        # 1D Gaussian window (un-normalized)
        gauss = np.array([np.exp(-(x - window_size//2)**2 / float(window_size)) for x in range(window_size)], dtype=np.float32)
        gauss = torch.Tensor(gauss).unsqueeze(0).unsqueeze(0)
        window = gauss / gauss.sum()
        return window
    def forward(self, img1, img2):
        window = self.window.to(img1.device)
        mu1 = F.conv1d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv1d(img2, window, padding=self.window_size // 2, groups=self.channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv1d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv1d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12   = F.conv1d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=2).mean(dim=1)

class EnhancedCombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, delta=0.5):
        """
        alpha: weight for MSE loss
        beta:  weight for MAE loss
        gamma: weight for SSIM-based loss
        delta: weight for gradient-based loss
        """
        super(EnhancedCombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ssim_loss = SSIM1D(window_size=11)
    def gradient_loss(self, pred, target):
        if pred.size(2) <= 1 or target.size(2) <= 1:
            return torch.tensor(0.0, device=pred.device)
        pred_grad = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        target_grad = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        grad_mse = F.mse_loss(pred_grad, target_grad)
        if pred.size(2) > 2:
            pred_grad2 = torch.abs(pred_grad[:, :, 1:] - pred_grad[:, :, :-1])
            target_grad2 = torch.abs(target_grad[:, :, 1:] - target_grad[:, :, :-1])
            if pred_grad2.numel() > 0 and target_grad2.numel() > 0:
                grad2_mse = F.mse_loss(pred_grad2, target_grad2)
                return 0.7 * grad_mse + 0.3 * grad2_mse
        return grad_mse
    def forward(self, outputs, targets):
        # Standard losses
        mse_val = self.mse_loss(outputs, targets)
        mae_val = self.mae_loss(outputs, targets)
        ssim_val = self.ssim_loss(outputs, targets)
        ssim_loss = 1 - ssim_val
        grad_val = self.gradient_loss(outputs, targets)
        total_loss = self.alpha * mse_val + self.beta * mae_val + self.gamma * ssim_loss + self.delta * grad_val
        return total_loss, mse_val, mae_val, ssim_loss, grad_val