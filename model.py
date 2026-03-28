import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


# Snake activation

@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape # [B, C, T]
    x = x.reshape(shape[0], shape[1], -1) # [B, C, T]
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape) # [B, C, T]
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1)) # [1, C, 1] one for each channel

    def forward(self, x):
        return snake(x, self.alpha)


# Weight-normalized convolutions

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class VQ(nn.Module):
    def __init__(self, latent_ch, K=1024, codebook_dim=8):
        super().__init__()
        self.in_proj = nn.Linear(latent_ch, codebook_dim, bias=False)
        self.out_proj = nn.Linear(codebook_dim, latent_ch, bias=False)
        self.codebook = nn.Embedding(K, codebook_dim)

    def forward(self, z: torch.tensor):
        # z -> [N, C] 2d tensor flattened

        # project to low-dim codebook space
        z_e = self.in_proj(z)  # [N, codebook_dim]

        # L2 normalise for cosine similarity matching
        z_e_norm = F.normalize(z_e, dim=-1)              # [N, codebook_dim]
        cb_norm = F.normalize(self.codebook.weight, dim=-1)  # [K, codebook_dim]

        # euclidean distance between two unit vectors ~ cosine similarity
        sim = z_e_norm @ cb_norm.t()  # [N, K]

        # nearest codebook entry = highest similarity
        indices = sim.max(dim=1)[1]  # [N]

        # lookup normalised codebook entry
        z_q_norm = cb_norm[indices]  # [N, codebook_dim]

        # losses on normalised vectors
        commitment_loss = F.mse_loss(z_e_norm, z_q_norm.detach())  # push encoder direction → codebook
        codebook_loss = F.mse_loss(z_e_norm.detach(), z_q_norm)    # push codebook → encoder direction

        # STE in normalised space
        z_q_st = z_e_norm + (z_q_norm - z_e_norm).detach()

        # project back to full latent space
        z_q_out = self.out_proj(z_q_st)  # [N, latent_ch]

        return z_q_out, indices, commitment_loss, codebook_loss


class RVQ(nn.Module):
    def __init__(self, num_levels, latent_ch, K=1024, codebook_dim=8):
        super().__init__()
        self.num_levels = num_levels
        self.levels = nn.ModuleList([
            VQ(latent_ch, K=K, codebook_dim=codebook_dim) for _ in range(num_levels)
        ])

    def forward(self, z):
        # z -> [N, C] 2d flat vector
        r = z # initilise residual with z for the first level
        quantised_sum = torch.zeros_like(z)
        all_indices = []
        total_commitment_loss = 0
        total_codebook_loss = 0

        for level in self.levels:
            z_q, indices, commitment_loss, codebook_loss = level(r)
            r = r - z_q.detach()  # next level quantizes the error
            quantised_sum = quantised_sum + z_q  # accumulate: z ≈ q1 + q2 + q3 + ...
            all_indices.append(indices)
            total_commitment_loss = total_commitment_loss + commitment_loss
            total_codebook_loss = total_codebook_loss + codebook_loss

        return quantised_sum, all_indices, total_commitment_loss, total_codebook_loss


class ResidualUnit(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(ch), # [B, C, T]
            WNConv1d(ch, ch, kernel_size=7, dilation=dilation, padding=3 * dilation), # [B, C, T] sk=7, padding=3 to keep same shape
            Snake1d(ch), # [B, C, T]
            WNConv1d(ch, ch, kernel_size=1), # [B, C, T]
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.res1 = ResidualUnit(in_ch, dilation=1)
        self.res2 = ResidualUnit(in_ch, dilation=3)
        self.res3 = ResidualUnit(in_ch, dilation=9)
        self.downsample = nn.Sequential(
            Snake1d(in_ch),
            WNConv1d(in_ch, out_ch, kernel_size=2 * stride, stride=stride, padding=stride // 2),
        )

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.downsample(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.upsample = nn.Sequential(
            Snake1d(in_ch),
            WNConvTranspose1d(in_ch, out_ch, kernel_size=2 * stride, stride=stride, padding=stride // 2),
        )
        self.res1 = ResidualUnit(out_ch, dilation=1)
        self.res2 = ResidualUnit(out_ch, dilation=3)
        self.res3 = ResidualUnit(out_ch, dilation=9)

    def forward(self, x):
        x = self.upsample(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class RVQCodec(nn.Module):
    def __init__(self, in_ch=1, latent_ch=32, K=1024, num_rvq_levels=1, codebook_dim=8):
        super().__init__()
        # Encoder - [B, 1, T] → [B, D, T/128]
        # strides - 2 × 4 × 4 × 4 = 128x downsample
        self.encoder = nn.Sequential(
            WNConv1d(in_ch, 64, kernel_size=7, padding=3),              # [B, 64, T]
            EncoderBlock(64, 128, stride=2),                             # [B, 128, T/2]
            EncoderBlock(128, 256, stride=4),                            # [B, 256, T/8]
            EncoderBlock(256, 512, stride=4),                            # [B, 512, T/32]
            EncoderBlock(512, 512, stride=4),                            # [B, 512, T/128]
            Snake1d(512),
            WNConv1d(512, latent_ch, kernel_size=3, padding=1),          # [B, D, T/128]
        )
        # Decoder - [B, D, T/128] → [B, 1, T]
        # strides - 4 × 4 × 4 × 2 = 128x upsample
        self.decoder = nn.Sequential(
            WNConv1d(latent_ch, 512, kernel_size=7, padding=3),          # [B, 512, T/128]
            DecoderBlock(512, 512, stride=4),                            # [B, 512, T/32]
            DecoderBlock(512, 256, stride=4),                            # [B, 256, T/8]
            DecoderBlock(256, 128, stride=4),                            # [B, 128, T/2]
            DecoderBlock(128, 64, stride=2),                             # [B, 64, T]
            Snake1d(64),
            WNConv1d(64, in_ch, kernel_size=7, padding=3),              # [B, 1, T]
            nn.Tanh(),
        )
        self.rvq = RVQ(num_levels=num_rvq_levels, latent_ch=latent_ch, K=K, codebook_dim=codebook_dim)

    def forward(self, x: torch.tensor):
        # x -> [B, C=1, T]
        z = self.encoder(x) # [B, D, T/128]

        # flatten to 2d vector for applying rvq on channel dim
        B, C, T_128 = z.shape
        z_flat = z.permute(0, 2, 1).contiguous().view(B * T_128, C)

        # vector quantize
        z_q, all_indices, commitment_loss, codebook_loss = self.rvq(z_flat)

        # reshape back
        z_q = z_q.view(B, T_128, C).permute(0, 2, 1) # [B, C, T_128]

        x_recon = self.decoder(z_q) # [B, C=1, T]

        return x_recon, all_indices, commitment_loss, codebook_loss


if __name__ == "__main__":
    device = "cuda"
    x = torch.randn(1, 1, 8192)

    model = RVQCodec()
    print(model)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    x = x.to(device)
    model = model.to(device)

    out, _, _, _ = model(x)
    print(f"in: {x.shape} → out: {out.shape}")
