import torch


def mse(pred, target):
    return torch.mean((pred - target) ** 2)


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))
