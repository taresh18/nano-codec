import os
import json
import time
import logging

import torch
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def save_loss_plot(recon_losses, commit_losses, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(recon_losses) + 1)
    ax.plot(epochs, recon_losses, label='Reconstruction', alpha=0.7, marker='o')
    ax.plot(epochs, commit_losses, label='Commitment', alpha=0.7, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Audio Codec Training')
    ax.legend()
    ax.set_yscale('log')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def save_checkpoint(model, optimiser, scheduler, epoch, losses, path, is_compiled=False):
    state = model.state_dict()
    # strip torch.compile prefix if present
    if is_compiled:
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    torch.save({
        'epoch': epoch,
        'model_state_dict': state,
        'optimiser_state_dict': optimiser.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'losses': losses,
    }, path)


def load_checkpoint(path, model, optimiser=None, scheduler=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimiser is not None and 'optimiser_state_dict' in ckpt:
        optimiser.load_state_dict(ckpt['optimiser_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt.get('losses', {})


def setup_exp_dir(base_dir, exp_name):
    """Create experiment directory structure:
        base_dir/exp_name/
            checkpoints/
            samples/
    """
    exp_dir = os.path.join(base_dir, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    samples_dir = os.path.join(exp_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    return exp_dir, ckpt_dir, samples_dir


def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


class Profiler:
    """Profile first N batches if enabled, otherwise no-op."""
    def __init__(self, enabled=False, num_batches=5):
        self.enabled = enabled
        self.num_batches = num_batches
        self.marks = {}

    def should_profile(self, batch_idx):
        return self.enabled and batch_idx < self.num_batches

    def mark(self, name, batch_idx):
        if self.should_profile(batch_idx):
            torch.cuda.synchronize()
            self.marks[name] = time.time()

    def report(self, batch_idx):
        if not self.should_profile(batch_idx):
            return
        m = self.marks
        log.info(
            f"  [profile] batch {batch_idx}: "
            f"data={m['fwd']-m['start']:.3f}s "
            f"fwd={m['loss']-m['fwd']:.3f}s "
            f"loss={m['bwd']-m['loss']:.3f}s "
            f"bwd={m['end']-m['bwd']:.3f}s "
            f"total={m['end']-m['start']:.3f}s"
        )
        self.marks.clear()


class MetricsTracker:
    """Accumulate batch metrics and compute epoch averages."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.recon = []
        self.commit = []
        self.cb = []

    def update(self, recon_details, commitment_loss, codebook_loss):
        self.recon.append(recon_details['recon_total'])
        self.commit.append(commitment_loss.item())
        self.cb.append(codebook_loss.item())

    def averages(self):
        return (
            sum(self.recon) / len(self.recon),
            sum(self.commit) / len(self.commit),
            sum(self.cb) / len(self.cb),
        )


class WandbLogger:
    """Handle wandb logging. No-op if disabled."""
    def __init__(self, enabled, log_interval):
        self.enabled = enabled
        self.log_interval = log_interval

    def log_batch(self, batch_idx, recon_details, commitment_loss, codebook_loss, raw_total):
        if not self.enabled or batch_idx % self.log_interval != 0:
            return
        import wandb
        log_dict = {
            "batch/recon_loss": recon_details['recon_total'],
            "batch/commit_loss": commitment_loss.item(),
            "batch/codebook_loss": codebook_loss.item(),
            "batch/total_loss": raw_total.item(),
        }
        for k, v in recon_details.items():
            if k != 'recon_total':
                log_dict[f"batch/{k}_loss"] = v
        wandb.log(log_dict)

    def log_epoch(self, epoch, avg_recon, avg_commit, avg_cb, lr):
        if not self.enabled:
            return
        import wandb
        wandb.log({
            "epoch/recon_loss": avg_recon,
            "epoch/commit_loss": avg_commit,
            "epoch/codebook_loss": avg_cb,
            "epoch/lr": lr,
            "epoch": epoch,
        })

    def log_audio(self, epoch, idx, orig_path, recon_path, sample_rate):
        if not self.enabled:
            return
        import wandb
        wandb.log({
            f"audio/sample_{idx}_original": wandb.Audio(orig_path, sample_rate=sample_rate, caption=f"original s{idx} e{epoch}"),
            f"audio/sample_{idx}_recon": wandb.Audio(recon_path, sample_rate=sample_rate, caption=f"recon s{idx} e{epoch}"),
            "epoch": epoch,
        })


def reconstruct_samples(model, dataset, eval_indices, device, use_amp, epoch, samples_dir, sample_rate, logger):
    """Reconstruct eval samples and save as wav files."""
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(eval_indices):
            sample = dataset[idx].unsqueeze(0).to(device)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                recon, _, _, _ = model(sample)
            recon = recon[..., :sample.shape[-1]]

            orig_path = os.path.join(samples_dir, f"e{epoch:03d}_s{i}_original.wav")
            recon_path = os.path.join(samples_dir, f"e{epoch:03d}_s{i}_recon.wav")
            sf.write(orig_path, sample[0, 0].cpu().float().numpy(), sample_rate)
            sf.write(recon_path, recon[0, 0].cpu().float().numpy(), sample_rate)
            logger.log_audio(epoch, i, orig_path, recon_path, sample_rate)
