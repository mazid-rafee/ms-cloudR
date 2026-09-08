import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t, max_period=10000.0):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        dw_channel = channels * 2
        ffn_channel = channels * 2

        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = nn.GroupNorm(1, channels)
        self.conv4 = nn.Conv2d(channels, ffn_channel, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        return x + y * self.gamma


class SFBlock(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x_opt, x_sar):
        b, c, h, w = x_opt.shape
        q = self.q(x_opt).reshape(b, self.heads, c // self.heads, h * w)
        k = self.k(x_sar).reshape(b, self.heads, c // self.heads, h * w)
        v = self.v(x_sar).reshape(b, self.heads, c // self.heads, h * w)

        q = q / math.sqrt(c // self.heads)
        attn = torch.softmax(
            torch.einsum("bhcn,bhdn->bhcd", q, k),
            dim=-1,
        )
        fused = torch.einsum("bhcn,bhcd->bhdn", v, attn).reshape(b, c, h, w)
        out = x_opt + self.proj(fused)
        out = out + self.mlp(out)
        return out


class DBCRNet(nn.Module):
    def __init__(
        self,
        in_opt=13,
        in_sar=2,
        widths=(22, 44, 88, 176),
        enc_blocks=(1, 1, 1, 28),
        dec_blocks=(1, 1, 1, 1),
        heads=(1, 1, 2, 4),
        time_dim=128,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.time_to_channels = nn.ModuleList(
            [nn.Linear(time_dim, c) for c in widths]
        )

        self.opt_stem = nn.Conv2d(in_opt, widths[0], 1)
        self.sar_stem = nn.Conv2d(in_sar, widths[0], 1)

        self.opt_enc = nn.ModuleList()
        self.sar_enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.fuse = nn.ModuleList()

        for i, c in enumerate(widths):
            self.opt_enc.append(nn.Sequential(*[NAFBlock(c) for _ in range(enc_blocks[i])]))
            self.sar_enc.append(nn.Sequential(*[NAFBlock(c) for _ in range(enc_blocks[i])]))
            self.fuse.append(SFBlock(c, heads[i]))
            if i < len(widths) - 1:
                self.downs.append(nn.Conv2d(c, widths[i + 1], 2, stride=2))

        self.mid = nn.Sequential(*[NAFBlock(widths[-1]) for _ in range(1)])

        self.ups = nn.ModuleList()
        self.opt_dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.opt_dec.append(nn.Sequential(*[NAFBlock(widths[i - 1]) for _ in range(dec_blocks[len(widths) - 1 - i])]))

        self.head = nn.Conv2d(widths[0], in_opt, 1)

    def forward(self, x_t, t, z):
        b = x_t.size(0)
        t = t.float()
        t_embed = self.time_mlp(t)

        opt = self.opt_stem(x_t)
        sar = self.sar_stem(z)

        skips = []
        for i in range(len(self.opt_enc)):
            bias = self.time_to_channels[i](t_embed).view(b, -1, 1, 1)
            opt = opt + bias
            opt = self.opt_enc[i](opt)
            sar = self.sar_enc[i](sar)
            opt = self.fuse[i](opt, sar)
            skips.append(opt)
            if i < len(self.downs):
                opt = self.downs[i](opt)
                sar = self.downs[i](sar)

        opt = self.mid(opt)

        for i, up in enumerate(self.ups):
            opt = up(opt)
            skip = skips[-(i + 2)]
            opt = opt + skip
            opt = self.opt_dec[i](opt)

        return self.head(opt)


def alpha_schedule(t, T):
    """Original DB-CR sinusoidal bridge schedule. alpha(0)=0, alpha(T)=1."""
    return torch.sin((t / T) * math.pi / 2)


def mean_reverting_alpha_schedule(t, T, rate=3.0):
    """Deterministic mean-reverting bridge schedule with exact endpoints.

    Uses normalized time s = t / T and
        alpha(t) = (1 - exp(-rate * s)) / (1 - exp(-rate))
    so that alpha(0) = 0 and alpha(T) = 1 exactly (up to floating-point).

    Args:
        t: scalar or tensor timesteps (same device as returned alpha).
        T: total diffusion steps (positive scalar).
        rate: mean-reversion rate; must be > 0.
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    t = t.float()

    rate_val = float(rate.detach().item() if torch.is_tensor(rate) else rate)
    if rate_val <= 0.0:
        raise ValueError(f"mean_reversion_rate must be > 0, got {rate_val}")

    s = t / float(T)
    rate_t = torch.as_tensor(rate_val, dtype=t.dtype, device=t.device)
    # expm1 improves accuracy for small rate; mathematically equal to
    # (1 - exp(-rate * s)) / (1 - exp(-rate)).
    return torch.expm1(-rate_t * s) / torch.expm1(-rate_t)


def get_alpha_schedule(bridge_schedule="original", mean_reversion_rate=3.0):
    """Return the alpha(t, T) callable for the selected bridge schedule."""
    name = str(bridge_schedule).lower().replace("-", "_")
    if name == "original":
        return alpha_schedule
    if name == "mean_reverting":
        rate = float(mean_reversion_rate)

        def schedule(t, T):
            return mean_reverting_alpha_schedule(t, T, rate=rate)

        return schedule
    raise ValueError(
        f"Unknown bridge_schedule '{bridge_schedule}'. "
        "Expected 'original' or 'mean_reverting'."
    )
