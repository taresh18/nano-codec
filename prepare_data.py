"""Pre-process LibriSpeech into fixed-size chunks saved as a single .pt file.

Usage:
    python prepare_data.py

Reads config.yaml for sample_rate, chunk_size, data_dir, librispeech_url.
Outputs: {data_dir}/chunks_{sample_rate}_{chunk_size}.pt
"""

import os
import logging
import sys
from tqdm import tqdm

import torch
import torchaudio
import soundfile as sf
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    data_dir = cfg['data_dir']
    sample_rate = cfg['sample_rate']
    chunk_size = cfg['chunk_size']
    url = cfg['librispeech_url']

    # download/extract
    log.info(f"loading LibriSpeech ({url})...")
    torchaudio.datasets.LIBRISPEECH(root=data_dir, url=url, download=True)

    # find all .flac files
    data_path = os.path.join(data_dir, "LibriSpeech", url)
    flac_files = []
    for dirpath, _, filenames in os.walk(data_path):
        for f in filenames:
            if f.endswith(".flac"):
                flac_files.append(os.path.join(dirpath, f))
    flac_files.sort()
    log.info(f"found {len(flac_files)} audio files")

    # chunk all audio
    chunks = []
    for path in tqdm(flac_files, desc="Processing"):
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        waveform = torch.from_numpy(audio).unsqueeze(0)

        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        # normalize
        waveform = waveform / waveform.abs().max().clamp(min=1e-8)

        # split into fixed-size chunks
        num_samples = waveform.shape[1]
        for start in range(0, num_samples - chunk_size + 1, chunk_size):
            chunk = waveform[:, start:start + chunk_size]
            chunks.append(chunk)

    # stack into single tensor
    all_chunks = torch.stack(chunks)  # [N, 1, chunk_size]
    log.info(f"total chunks: {len(chunks)} ({len(chunks) * chunk_size / sample_rate / 3600:.1f} hours)")
    log.info(f"tensor shape: {all_chunks.shape}, size: {all_chunks.element_size() * all_chunks.nelement() / 1e9:.2f} GB")

    # save
    out_path = os.path.join(data_dir, f"chunks_{sample_rate}_{chunk_size}.pt")
    torch.save(all_chunks, out_path)
    log.info(f"saved to {out_path}")


if __name__ == "__main__":
    main()
