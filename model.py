import torch 
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    def __init__(self, K, latent_ch, cb_lerp=0.2):
        super().__init__()
        # codebook is a LEARNABLE parameter — optimizer updates it via codebook_loss gradients
        self.codebook = nn.Parameter(torch.randn(K, latent_ch))
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, z: torch.tensor):
        # z -> [N, C] 2d tensor flattened

        # initialize codebook from first batch of actual encoder outputs
        if not self.initialized and self.training:
            with torch.no_grad():
                K = self.codebook.shape[0]
                N = z.shape[0]
                if N >= K:
                    perm = torch.randperm(N)[:K]
                    self.codebook.copy_(z[perm].detach())
                else:
                    # not enough data points — repeat with random sampling
                    idx = torch.randint(0, N, (K,), device=z.device)
                    self.codebook.copy_(z[idx].detach())
                self.initialized.fill_(True)

        # distances from each embedding to each codebook entry
        dist = (z.detach()[:, None, :] - self.codebook[None, :, :]).pow(2).sum(dim=2)  # [N, K]

        # nearest codebook entry for each embedding
        indices = torch.argmin(dist, dim=-1)  # [N]

        # lookup quantized vectors
        z_q = self.codebook[indices]  # [N, C]

        # losses
        commitment_loss = F.mse_loss(z, z_q.detach())  # push encoder → codebook (gradient to encoder only)
        codebook_loss = F.mse_loss(z.detach(), z_q)     # push codebook → encoder (gradient to codebook only)

        # straight-through estimator
        r = z - z_q
        z_q_st = z - r.detach()

        return z_q_st, indices, commitment_loss, codebook_loss


class RVQ(nn.Module):
    def __init__(self, num_levels, K, latent_ch, cb_lerp=0.2):
        super().__init__()
        self.num_levels = num_levels
        self.levels = nn.ModuleList([
            VectorQuantize(K, latent_ch, cb_lerp=cb_lerp) for _ in range(num_levels)
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



class RVQCodec(nn.Module):
    def __init__(self, in_ch=1, latent_ch=32, K=512, num_rvq_levels=1, cb_lerp=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            # input -> [B, 1, T]
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=1, padding=3),  # [B, 32, T]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),     # [B, 64, T/2]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=4, padding=0),    # [B, 128, T/8]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=4, padding=0),   # [B, 256, T/32]
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=4, stride=4, padding=0),   # [B, 256, T/128]
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, latent_ch, kernel_size=3, stride=1, padding=1),   # [B, D, T/128] D is latent_ch
        )
        self.decoder = nn.Sequential(
            # input -> [B, D, T/128]
            nn.Conv1d(latent_ch, 256, kernel_size=3, stride=1, padding=1),   # [B, 256, T/128]
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=4, padding=0),  # [B, 256, T/32]
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=4, padding=0),   # [B, 128, T/8]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4, padding=0), # [B, 64, T/2]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1), # [B, 32, T]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, in_ch, kernel_size=7, stride=1, padding=3),   # [B, 1, T]
            nn.Tanh(), # not sigmoid as audio samples range is [-1, 1]
        )
        self.rvq = RVQ(num_levels=num_rvq_levels, K=K, latent_ch=latent_ch, cb_lerp=cb_lerp)

    def forward(self, x: torch.tensor):
        # x -> [B, C=1, T]
        z = self.encoder(x) # [B, D, T/128]

        # flatten to 2d vector for applying rvq on channel dim
        B, C, T_128 = z.shape
        z_flat = z.permute(0, 2, 1).contiguous().view(B*T_128, C)

        # vector quantize
        z_q, all_indices, commitment_loss, codebook_loss = self.rvq(z_flat)

        # reshape back
        z_q = z_q.view(B, T_128, C).permute(0, 2, 1) # [B, C, T_128]

        x_recon = self.decoder(z_q) # [B, C=1, T]

        return x_recon, all_indices, commitment_loss, codebook_loss


if __name__ == "__main__":
    device = "cuda"
    im = torch.randn(1, 1, 8192)

    model = RVQCodec()

    print(model)

    im = im.to(device)
    model = model.to(device)

    out, _, _, _ = model(im)

    print(out.shape)
