import os
import sys
import logging
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

import torch
import soundfile as sf
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb

from model import RVQCodec
from loader import AudioChunkDataset
from loss import build_loss
from utils import save_loss_plot, save_checkpoint, load_checkpoint, setup_exp_dir, save_config

torch.manual_seed(67)

log = logging.getLogger(__name__)


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )



def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train(cfg, model, loader, dataset, optimiser, scheduler, criterion, device, exp_dir, ckpt_dir, samples_dir, start_epoch, epoch_recon_losses, epoch_commit_losses, best_recon_loss):
    # fixed eval samples for consistent reconstruction comparison
    num_eval_samples = cfg.get('num_eval_samples', 4)
    eval_indices = torch.randperm(len(dataset))[:num_eval_samples].tolist()
    log.info(f"eval sample indices: {eval_indices}")

    for epoch in range(start_epoch, cfg['num_epochs']):
        model.train()
        batch_recon = []
        batch_commit = []

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch_idx, x in enumerate(pbar):
            x = x.to(device)
            x_recon, indices, commitment_loss, codebook_loss = model(x)
            x_recon = x_recon[..., :x.shape[-1]]

            recon_loss, recon_details = criterion(x, x_recon)
            total_loss = recon_loss + cfg['beta'] * commitment_loss + codebook_loss

            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()

            batch_recon.append(recon_details['recon_total'])
            batch_commit.append(commitment_loss.item())

            pbar.set_postfix(recon=f"{recon_details['recon_total']:.5f}", commit=f"{commitment_loss.item():.5f}")

            log_dict = {
                "batch/recon_loss": recon_details['recon_total'],
                "batch/commit_loss": commitment_loss.item(),
                "batch/total_loss": total_loss.item(),
            }
            # log individual loss components
            for k, v in recon_details.items():
                if k != 'recon_total':
                    log_dict[f"batch/{k}_loss"] = v
            wandb.log(log_dict)

        scheduler.step()

        avg_recon = sum(batch_recon) / len(batch_recon)
        avg_commit = sum(batch_commit) / len(batch_commit)
        epoch_recon_losses.append(avg_recon)
        epoch_commit_losses.append(avg_commit)
        log.info(f"epoch {epoch}: recon={avg_recon:.5f}, commit={avg_commit:.5f}, lr={scheduler.get_last_lr()[0]:.6f}")

        wandb.log({
            "epoch/recon_loss": avg_recon,
            "epoch/commit_loss": avg_commit,
            "epoch/lr": scheduler.get_last_lr()[0],
            "epoch": epoch,
        })

        # save last checkpoint
        losses = {'recon': epoch_recon_losses, 'commit': epoch_commit_losses}
        save_checkpoint(model, optimiser, scheduler, epoch, losses, os.path.join(ckpt_dir, "last.pt"))

        # save best checkpoint + reconstruct samples
        if avg_recon < best_recon_loss:
            best_recon_loss = avg_recon
            save_checkpoint(model, optimiser, scheduler, epoch, losses, os.path.join(ckpt_dir, "best.pt"))
            log.info(f"  new best recon: {best_recon_loss:.5f}")

            # reconstruct eval samples
            model.eval()
            with torch.no_grad():
                for i, idx in enumerate(eval_indices):
                    sample = dataset[idx].unsqueeze(0).to(device)  # [1, 1, chunk_size]
                    recon, _, _, _ = model(sample)
                    recon = recon[..., :sample.shape[-1]]

                    orig_path = os.path.join(samples_dir, f"e{epoch:03d}_s{i}_original.wav")
                    recon_path = os.path.join(samples_dir, f"e{epoch:03d}_s{i}_recon.wav")
                    sf.write(orig_path, sample[0, 0].cpu().numpy(), cfg['sample_rate'])
                    sf.write(recon_path, recon[0, 0].cpu().numpy(), cfg['sample_rate'])

                    wandb.log({
                        f"audio/sample_{i}_original": wandb.Audio(orig_path, sample_rate=cfg['sample_rate'], caption=f"original s{i} e{epoch}"),
                        f"audio/sample_{i}_recon": wandb.Audio(recon_path, sample_rate=cfg['sample_rate'], caption=f"recon s{i} e{epoch}"),
                        "epoch": epoch,
                    })

        # save loss plot
        save_loss_plot(epoch_recon_losses, epoch_commit_losses, os.path.join(exp_dir, "losses.png"))


def main():
    cfg = load_config('config.yaml')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = input("experiment name: ").strip()
    cfg['exp_name'] = exp_name

    # setup experiment directory
    exp_dir, ckpt_dir, samples_dir = setup_exp_dir(cfg['output_dir'], exp_name)
    setup_logging(os.path.join(exp_dir, "train.log"))
    save_config(cfg, os.path.join(exp_dir, "config.json"))

    # wandb
    wandb.init(project=cfg['wandb_project'], name=exp_name, config=cfg)

    # data
    dataset = AudioChunkDataset(
        root=cfg['data_dir'],
        chunk_size=cfg['chunk_size'],
        sample_rate=cfg['sample_rate'],
        url=cfg['librispeech_url'],
        max_chunks=cfg.get('max_chunks'),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
    )

    # model
    model = RVQCodec(
        in_ch=1,
        latent_ch=cfg['latent_dim'],
        K=cfg['codebook_size'],
        num_rvq_levels=cfg['num_rvq_levels'],
        cb_lerp=cfg['cb_lerp'],
    )
    model = model.to(device)
    log.info(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    optimiser = AdamW(model.parameters(), lr=cfg['lr'])
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg['num_epochs'], eta_min=cfg['lr_min'])

    # loss function
    criterion = build_loss(cfg).to(device)

    # tracking
    epoch_recon_losses = []
    epoch_commit_losses = []
    start_epoch = 0
    best_recon_loss = float('inf')

    # resume from checkpoint
    if cfg['resume']:
        last_ckpt = os.path.join(ckpt_dir, "last.pt")
        if os.path.exists(last_ckpt):
            start_epoch, losses = load_checkpoint(last_ckpt, model, optimiser, scheduler, device)
            start_epoch += 1
            epoch_recon_losses = losses.get('recon', [])
            epoch_commit_losses = losses.get('commit', [])
            best_recon_loss = min(epoch_recon_losses) if epoch_recon_losses else float('inf')
            log.info(f"resumed from epoch {start_epoch}")
        else:
            log.info("no checkpoint found, starting fresh")

    # train
    train(cfg, model, loader, dataset, optimiser, scheduler, criterion, device, exp_dir, ckpt_dir, samples_dir,
          start_epoch, epoch_recon_losses, epoch_commit_losses, best_recon_loss)

    wandb.finish()
    log.info(f"done. outputs in {exp_dir}/")


if __name__ == "__main__":
    main()
