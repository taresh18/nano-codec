import torch
import torchaudio
import soundfile as sf

from model import RVQCodec


def main():
    checkpoint_path = "outputs/run5/checkpoints/best.pt"
    input_audio = "/workspace/data/LibriSpeech/train-clean-100/19/198/19-198-0001.flac"
    output_dir = "outputs/generate"
    sample_rate = 16000
    chunk_size = 16384

    # model config - whats used during training
    latent_dim = 64
    codebook_size = 1024
    num_rvq_levels = 8
    codebook_dim = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = RVQCodec(
        in_ch=1,
        latent_ch=latent_dim,
        K=codebook_size,
        num_rvq_levels=num_rvq_levels,
        codebook_dim=codebook_dim,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    # strip torch.compile prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"loaded checkpoint from {checkpoint_path} (epoch {ckpt['epoch']})")

    # load audio
    waveform, sr = torchaudio.load(input_audio)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # stereo → mono
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    waveform = waveform / waveform.abs().max().clamp(min=1e-8)  # normalize

    print(f"input: {input_audio}")
    print(f"  shape: {waveform.shape}, duration: {waveform.shape[1]/sample_rate:.2f}s")

    # reconstruct in chunks
    import os
    os.makedirs(output_dir, exist_ok=True)

    # pad to multiple of chunk_size
    total_samples = waveform.shape[1]
    pad_len = (chunk_size - total_samples % chunk_size) % chunk_size
    if pad_len > 0:
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    recon_chunks = []
    with torch.no_grad():
        for start in range(0, waveform.shape[1], chunk_size):
            chunk = waveform[:, start:start+chunk_size].unsqueeze(0).to(device)  # [1, 1, chunk_size]
            recon, _, _, _ = model(chunk)
            recon = recon[..., :chunk_size]
            recon_chunks.append(recon.cpu())

    recon_full = torch.cat(recon_chunks, dim=-1)
    recon_full = recon_full[0, :, :total_samples]  # trim padding, [1, total_samples]

    # save
    orig_path = os.path.join(output_dir, "original.wav")
    recon_path = os.path.join(output_dir, "reconstructed.wav")

    sf.write(orig_path, waveform[0, :total_samples].numpy(), sample_rate)
    sf.write(recon_path, recon_full[0].float().numpy(), sample_rate)

    print(f"saved: {orig_path}")
    print(f"saved: {recon_path}")


if __name__ == "__main__":
    main()
