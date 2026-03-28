import torch.nn as nn
import torch.nn.functional as F
import auraloss.freq


def build_loss(cfg):
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
            # 7-scale mel loss matching
            mel_scales = [
                (32,  5),     # fft=32,   mels=5
                (64,  10),    # fft=64,   mels=10
                (128, 20),    # fft=128,  mels=20
                (256, 40),    # fft=256,  mels=40
                (512, 80),    # fft=512,  mels=80
                (1024, 160),  # fft=1024, mels=160
                (2048, 320),  # fft=2048, mels=320
            ]
            self.mel_losses = nn.ModuleList([
                auraloss.freq.MelSTFTLoss(
                    sample_rate=sample_rate,
                    fft_size=fft,
                    hop_size=fft // 4,
                    win_length=fft,
                    n_mels=mels,
                )
                for fft, mels in mel_scales
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
