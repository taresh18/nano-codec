import torch.nn as nn
import torch.nn.functional as F
import auraloss.freq


def build_loss(cfg):
    """Build loss function from config.

    Config options for loss_type:
        "mse"       — waveform MSE only
        "stft"      — multi-resolution STFT loss only
        "mel"       — mel spectrogram loss only
        "mse+stft"  — waveform MSE + multi-resolution STFT
        "mse+mel"   — waveform MSE + mel loss
        "mse+stft+mel" — all three combined

    Config options for loss weights:
        lambda_mse:  weight for MSE loss (default 1.0)
        lambda_stft: weight for STFT loss (default 1.0)
        lambda_mel:  weight for mel loss (default 1.0)
    """
    loss_type = cfg.get('loss_type', 'mse')
    sample_rate = cfg.get('sample_rate', 16000)
    lambda_mse = cfg.get('lambda_mse', 1.0)
    lambda_stft = cfg.get('lambda_stft', 1.0)
    lambda_mel = cfg.get('lambda_mel', 1.0)

    return CodecLoss(
        loss_type=loss_type,
        sample_rate=sample_rate,
        lambda_mse=lambda_mse,
        lambda_stft=lambda_stft,
        lambda_mel=lambda_mel,
    )


class CodecLoss(nn.Module):
    def __init__(self, loss_type="mse", sample_rate=16000,
                 lambda_mse=1.0, lambda_stft=1.0, lambda_mel=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_mse = lambda_mse
        self.lambda_stft = lambda_stft
        self.lambda_mel = lambda_mel

        self.use_mse = "mse" in loss_type
        self.use_stft = "stft" in loss_type
        self.use_mel = "mel" in loss_type

        if self.use_stft:
            self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss(
                fft_sizes=[256, 512, 1024, 2048],
                hop_sizes=[64, 128, 256, 512],
                win_lengths=[256, 512, 1024, 2048],
            )

        if self.use_mel:
            # multiple mel losses at different scales
            # win_length must equal fft_size (default in auraloss uses 1024 which breaks for small fft)
            self.mel_losses = nn.ModuleList([
                auraloss.freq.MelSTFTLoss(sample_rate=sample_rate, fft_size=256, hop_size=64, win_length=256, n_mels=32),
                auraloss.freq.MelSTFTLoss(sample_rate=sample_rate, fft_size=512, hop_size=128, win_length=512, n_mels=64),
                auraloss.freq.MelSTFTLoss(sample_rate=sample_rate, fft_size=1024, hop_size=256, win_length=1024, n_mels=128),
                auraloss.freq.MelSTFTLoss(sample_rate=sample_rate, fft_size=2048, hop_size=512, win_length=2048, n_mels=128),
            ])

    def forward(self, x, x_recon):
        """Returns (total_loss, loss_dict) for logging."""
        losses = {}
        total = 0

        if self.use_mse:
            mse = F.mse_loss(x_recon, x)
            losses['mse'] = mse.item()
            total = total + self.lambda_mse * mse

        if self.use_stft:
            stft = self.stft_loss(x_recon, x)
            losses['stft'] = stft.item()
            total = total + self.lambda_stft * stft

        if self.use_mel:
            mel = sum(m(x_recon, x) for m in self.mel_losses) / len(self.mel_losses)
            losses['mel'] = mel.item()
            total = total + self.lambda_mel * mel

        losses['recon_total'] = total.item()
        return total, losses
