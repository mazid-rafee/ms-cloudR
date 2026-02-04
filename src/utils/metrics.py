import torch


def mse(pred, target):
    return torch.mean((pred - target) ** 2)


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def psnr(pred, target, max_val=1.0):
    mse_val = mse(pred, target)
    return 10 * torch.log10((max_val ** 2) / (mse_val + 1e-8))


def ssim(pred, target, max_val=1.0):
    # Global SSIM (no window), suitable for quick comparison.
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    mu_x = torch.mean(pred, dim=(-2, -1), keepdim=True)
    mu_y = torch.mean(target, dim=(-2, -1), keepdim=True)
    sigma_x = torch.mean((pred - mu_x) ** 2, dim=(-2, -1), keepdim=True)
    sigma_y = torch.mean((target - mu_y) ** 2, dim=(-2, -1), keepdim=True)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y), dim=(-2, -1), keepdim=True)
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return torch.mean(num / (den + 1e-8))
