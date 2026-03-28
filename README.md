# nano-codec

A minimal neural audio codec built from scratch for learning.

16kHz mono audio. 128x temporal compression

## Architecture

- **Encoder**: 4-block CNN with Snake activations, weight normalization, dilated residuals (128x downsample)
- **Quantizer**: 8-level Residual Vector Quantization with factorized codebooks
- **Decoder**: Mirror of encoder with transposed convolutions (128x upsample)
- **Loss**: Multi-scale Mel spectrogram + Multi-resolution STFT + MSE

Trained on LibriSpeech train-clean-100 (~100 hours) at 16kHz.

## Usage

```bash
# prepare data
python prepare_data.py

# train
python train.py

# reconstruct audio
python generate.py
```

## Model

<!-- TODO: add link after training -->
Pretrained weights: [HuggingFace]()

## Samples

<!-- TODO: add audio samples after training -->
| Original | Reconstructed |
|----------|--------------|
| | |

## References

- [Audio Codec Explainer (Kyutai)](https://kyutai.org/codec-explainer)
- [High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546)
- [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
