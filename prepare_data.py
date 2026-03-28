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

    # process and save in shards to avoid OOM
    shard_size = 10000  # chunks per shard
    out_dir = os.path.join(data_dir, f"chunks_{sample_rate}_{chunk_size}")
    os.makedirs(out_dir, exist_ok=True)

    shard_chunks = []
    shard_idx = 0
    total_chunks = 0

    def save_shard():
        nonlocal shard_chunks, shard_idx
        if not shard_chunks:
            return
        tensor = torch.stack(shard_chunks)  # [N, 1, chunk_size]
        path = os.path.join(out_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(tensor, path)
        log.info(f"saved shard {shard_idx}: {tensor.shape}")
        shard_idx += 1
        shard_chunks.clear()

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
            shard_chunks.append(chunk)
            total_chunks += 1

            if len(shard_chunks) >= shard_size:
                save_shard()

    # save remaining chunks
    save_shard()

    log.info(f"total chunks: {total_chunks} ({total_chunks * chunk_size / sample_rate / 3600:.1f} hours)")
    log.info(f"saved {shard_idx} shards to {out_dir}/")


if __name__ == "__main__":
    main()
