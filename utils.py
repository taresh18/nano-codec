import os
import json

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
