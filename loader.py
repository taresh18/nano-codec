import os
import logging
import random
from tqdm import tqdm

import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class AudioChunkDataset(Dataset):
    """Lazy-loading audio dataset. Stores file paths + chunk offsets, loads on demand.

    Only scans files and computes chunk boundaries at init (fast, no audio loaded).
    Actual audio is read from disk in __getitem__ (one chunk at a time).
    """

    def __init__(self, root, chunk_size=16384, sample_rate=16000, url="train-clean-100", max_chunks=None):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.chunks = []  # list of (file_path, start_sample) tuples

        log.info(f"loading LibriSpeech ({url})...")
        torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=True)

        # find all .flac files
        data_path = os.path.join(root, "LibriSpeech", url)
        flac_files = []
        for dirpath, _, filenames in os.walk(data_path):
            for f in filenames:
                if f.endswith(".flac"):
                    flac_files.append(os.path.join(dirpath, f))
        flac_files.sort()

        # scan files for duration and compute chunk boundaries (no audio loaded)
        log.info(f"found {len(flac_files)} audio files, indexing chunks...")
        for path in tqdm(flac_files, desc="Indexing"):
            info = sf.info(path)
            num_samples = int(info.frames * sample_rate / info.samplerate)  # after resampling
            for start in range(0, num_samples - chunk_size + 1, chunk_size):
                self.chunks.append((path, start))

        if max_chunks is not None:
            random.shuffle(self.chunks)
            self.chunks = self.chunks[:max_chunks]

        log.info(f"total chunks: {len(self.chunks)} ({len(self.chunks) * chunk_size / sample_rate / 3600:.1f} hours)")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        path, start = self.chunks[idx]

        # load just the chunk we need
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, num_samples]

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # normalize
        waveform = waveform / waveform.abs().max().clamp(min=1e-8)

        # extract chunk
        chunk = waveform[:, start:start + self.chunk_size]  # [1, chunk_size]
        return chunk
