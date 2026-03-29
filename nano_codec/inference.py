"""
nano-codec inference: reconstruct audio through the codec.

Usage:
    python inference.py --input input.wav --output reconstructed.wav

Downloads model weights from HuggingFace on first run.
"""

import argparse
import torch
import soundfile as sf
import torchaudio
import yaml
from huggingface_hub import hf_hub_download
from model import RVQCodec

REPO_ID = "taresh18/nano-codec"


def load_model(device="cpu"):
    model_path = hf_hub_download(REPO_ID, "model.pt")
    config_path = hf_hub_download(REPO_ID, "config.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = RVQCodec(
        in_ch=1,
        latent_ch=cfg['latent_dim'],
        K=cfg['codebook_size'],
        num_rvq_levels=cfg['num_rvq_levels'],
        codebook_dim=cfg.get('codebook_dim', 8),
    )

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    return model, cfg


def reconstruct(model, audio_path, output_path, sample_rate=16000, chunk_size=16384, device="cpu"):
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio).unsqueeze(0)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    waveform = waveform / waveform.abs().max().clamp(min=1e-8)

    total_samples = waveform.shape[1]
    pad_len = (chunk_size - total_samples % chunk_size) % chunk_size
    if pad_len > 0:
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    recon_chunks = []
    with torch.no_grad():
        for start in range(0, waveform.shape[1], chunk_size):
            chunk = waveform[:, start:start + chunk_size].unsqueeze(0).to(device)
            recon, _, _, _ = model(chunk)
            recon = recon[..., :chunk_size]
            recon_chunks.append(recon.cpu())

    recon_full = torch.cat(recon_chunks, dim=-1)
    recon_full = recon_full[0, :, :total_samples]

    sf.write(output_path, recon_full[0].float().numpy(), sample_rate)
    print(f"saved: {output_path} ({total_samples / sample_rate:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="nano-codec inference")
    parser.add_argument("--input", required=True, help="input wav file")
    parser.add_argument("--output", default="reconstructed.wav", help="output wav file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, cfg = load_model(device=args.device)
    reconstruct(
        model,
        args.input,
        args.output,
        sample_rate=cfg['sample_rate'],
        chunk_size=cfg['chunk_size'],
        device=args.device,
    )


if __name__ == "__main__":
    main()
