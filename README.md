# nano-codec 🔊

A minimal neural audio codec. 16kHz mono • 128x compression • 10.2 kbps • 24M parameters.

Trained on LibriSpeech train-clean-100 (~100 hours) for ~180k steps.

📝 [Blog Post]() — in-depth walkthrough of the architecture, training, and lessons learned

🤗 [Model Weights](https://huggingface.co/taresh18/nano-codec) — pretrained model on HuggingFace

💻 [GitHub](https://github.com/taresh18/nano-codec) — full training and inference code

## 🏗️ Architecture

![nano-codec architecture](assets/arch.png)

Inspired by [DAC](https://arxiv.org/abs/2306.06546) (Descript Audio Codec). Strided convolutional encoder, 8-level RVQ with factorized L2-normalized codebooks, mirror decoder.

## 🎧 Samples

| Original | Reconstructed |
|----------|---------------|
| https://github.com/user-attachments/assets/0509a99e-7538-4218-9769-874cf2dc57ff | https://github.com/user-attachments/assets/b11e9da3-5ac9-458e-921a-e1009a01da35 |
| https://github.com/user-attachments/assets/a9a883f7-bb27-4a3f-a92b-a0e857a76879 | https://github.com/user-attachments/assets/5e2792ab-dbf4-4b3f-9d21-d1de3b290a3e |
| https://github.com/user-attachments/assets/07153b3a-d6d9-452b-acad-a286d3c2f894 | https://github.com/user-attachments/assets/3152e68e-28c9-434b-875d-cb2f05419a7d |
| https://github.com/user-attachments/assets/912ca8a2-594d-4ffc-8494-70f15de733ff | https://github.com/user-attachments/assets/0809b359-793a-4989-af17-14082e8049b6 |

![mel spectrogram comparison](assets/aud_8_mel.png)

## 🏃 Quick Start

**1. Clone & Install**
```bash
git clone https://github.com/taresh18/nano-codec.git
cd nano-codec
uv sync
```

**2. Reconstruct Audio**
```bash
cd nano_codec
python inference.py --input audio.wav --output reconstructed.wav
```
Downloads model weights from HuggingFace on first run. Resamples to 16kHz if needed.

**3. Train Your Own**
```bash
cd nano_codec
python prepare_data.py    # download LibriSpeech, chunk into shards
python train.py           # config in configs/config.yaml
```

## 🏗️ Project Structure

```
nano-codec/
├── configs/
│   └── config.yaml           # Training & model config
├── nano_codec/
│   ├── model.py              # RVQCodec, VQ, RVQ, encoder/decoder
│   ├── loss.py               # Multi-scale spectral losses
│   ├── loader.py             # Dataset loading (in-memory + streaming)
│   ├── train.py              # Training loop
│   ├── inference.py          # Reconstruct audio from trained model
│   ├── prepare_data.py       # Preprocess LibriSpeech into chunks
│   └── utils.py              # Checkpointing, logging, profiling
└── assets/                   # Audio samples, images
```

## 📚 References

- [Audio Codec Explainer (Kyutai)](https://kyutai.org/codec-explainer)
- [High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546)
- [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
