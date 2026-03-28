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

    def __init__(self, root, chunk_size=16384, sample_rate=16000, url="train-clean-100",
                 max_chunks=None, streaming=False):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.streaming = streaming

        if streaming:
            self._init_streaming(root, url, max_chunks)
        else:
            self._init_in_memory(root, sample_rate, chunk_size, max_chunks)

    def _init_in_memory(self, root, sample_rate, chunk_size, max_chunks):
        """Load pre-processed chunks from shards directory or single .pt file into RAM."""
        shard_dir = os.path.join(root, f"chunks_{sample_rate}_{chunk_size}")
        single_pt = os.path.join(root, f"chunks_{sample_rate}_{chunk_size}.pt")

        if os.path.isdir(shard_dir):
            # load from shards
            shard_files = sorted([
                os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".pt")
            ])
            if not shard_files:
                raise FileNotFoundError(f"no .pt shards found in {shard_dir}")

            log.info(f"loading {len(shard_files)} shards from {shard_dir}...")
            shards = []
            loaded = 0
            for sf_path in tqdm(shard_files, desc="Loading shards"):
                shard = torch.load(sf_path, weights_only=True)
                if max_chunks is not None and loaded + shard.shape[0] > max_chunks:
                    shard = shard[:max_chunks - loaded]
                shards.append(shard)
                loaded += shard.shape[0]
                if max_chunks is not None and loaded >= max_chunks:
                    break
            self.data = torch.cat(shards, dim=0)  # [N, 1, chunk_size]

        else:
            raise FileNotFoundError(
                f"neither {shard_dir}/ nor {single_pt} found. Run `python prepare_data.py` first."
            )

        if max_chunks is not None and self.data.shape[0] > max_chunks:
            perm = torch.randperm(self.data.shape[0])[:max_chunks]
            self.data = self.data[perm]

        log.info(f"loaded {self.data.shape[0]} chunks ({self.data.shape[0] * chunk_size / sample_rate / 3600:.1f} hours), "
                 f"{self.data.element_size() * self.data.nelement() / 1e9:.2f} GB in RAM")

    def _init_streaming(self, root, url, max_chunks):
        """Index chunk boundaries for on-the-fly loading."""
        self.chunks = []

        log.info(f"loading LibriSpeech ({url})...")
        torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=True)

        data_path = os.path.join(root, "LibriSpeech", url)
        flac_files = []
        for dirpath, _, filenames in os.walk(data_path):
            for f in filenames:
                if f.endswith(".flac"):
                    flac_files.append(os.path.join(dirpath, f))
        flac_files.sort()

        log.info(f"found {len(flac_files)} audio files, indexing chunks...")
        for path in tqdm(flac_files, desc="Indexing"):
            info = sf.info(path)
            num_samples = int(info.frames * self.sample_rate / info.samplerate)
            for start in range(0, num_samples - self.chunk_size + 1, self.chunk_size):
                self.chunks.append((path, start))

        if max_chunks is not None:
            random.shuffle(self.chunks)
            self.chunks = self.chunks[:max_chunks]

        log.info(f"indexed {len(self.chunks)} chunks ({len(self.chunks) * self.chunk_size / self.sample_rate / 3600:.1f} hours)")

    def __len__(self):
        if self.streaming:
            return len(self.chunks)
        return self.data.shape[0]

    def __getitem__(self, idx):
        if not self.streaming:
            return self.data[idx]  # [1, chunk_size] — direct tensor lookup, very fast

        # streaming - read from disk
        path, start = self.chunks[idx]
        info = sf.info(path)
        orig_sr = info.samplerate
        orig_start = int(start * orig_sr / self.sample_rate)
        orig_frames = int(self.chunk_size * orig_sr / self.sample_rate) + 256
        orig_stop = min(orig_start + orig_frames, info.frames)

        audio, sr = sf.read(path, start=orig_start, stop=orig_stop, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        waveform = torch.from_numpy(audio).unsqueeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        waveform = waveform / waveform.abs().max().clamp(min=1e-8)
        chunk = waveform[:, :self.chunk_size]
        return chunk
