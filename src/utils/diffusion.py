import torch

def linear_beta_schedule(T):
    return torch.linspace(1e-4, 0.02, T)

class Diffusion:
    def __init__(self, T=1000, device="cuda"):
        self.T = T
        self.device = device

        self.betas = linear_beta_schedule(T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        a = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise
