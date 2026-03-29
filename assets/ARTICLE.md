# I Built a Neural Audio Codec

Most modern TTS models share the same core component: a neural audio codec. The codec learns to compress audio into discrete tokens and reconstruct it back. At inference time, a language model predicts these tokens from text, and the codec's decoder turns them into a waveform. The TTS model never touches raw audio directly. It just predicts tokens. The codec handles everything about actually producing sound.

I work on real-time voice agents so understanding this piece was essential. I built [nano-codec](https://github.com/taresh18/nano-codec) to learn how it works from the ground up, a 24M parameter codec trained on 100 hours of speech. This article walks through the architecture, the loss functions, and the surprisingly difficult problem of getting the codebook to train stably.

## What a Neural Codec Does

![One second of speech at 16kHz: 16,000 numbers](waveform.png)

Audio is a sequence of air pressure measurements sampled thousands of times per second. At 16kHz that's 16,000 numbers per second. Thirty seconds is 480,000 numbers. Working directly with sequences this long is difficult for any generative model.

This is the fundamental reason codecs exist. A neural codec has an encoder that squeezes the waveform into a short sequence of latent vectors and a decoder that reconstructs it back.


### Vector Quantization

There's a catch though. The language model downstream needs discrete tokens from a fixed vocabulary. That's how it was trained, predicting the next token from a set of possibilities. Continuous vectors don't fit that paradigm. So between the encoder and decoder, we quantize - snap each continuous latent vector to the nearest entry in a learned codebook. The codebook index becomes the token. This is vector quantization (VQ).

This works, but it comes at a cost. Snapping to the nearest codebook entry throws away everything that didn't exactly match. The gap between the original vector and its nearest entry is quantization error, and it shows up directly as audio artifacts. There's also a training challenge - "find nearest entry" isn't differentiable, so gradients can't flow through it during backprop. A trick called the straight-through estimator bridges this where the forward pass uses the quantized value, but the backward pass pretends quantization didn't happen and copies the decoder's gradients directly to the encoder.

### Residual Vector Quantization

A single codebook with 1024 entries isn't expressive enough for high-quality audio. Making the codebook bigger helps, but searching millions of entries gets expensive.

RVQ takes a different approach. It stacks multiple small codebooks, each one encoding the error left by the previous one. The first codebook captures the broad structure, second captures what the first missed, and the third captures what both missed. Each level chips away at the quantization error. Eight codebooks of 1024 entries each can represent 1024^8 possible combinations which is enough to capture the subtle differences between speakers, pitches, and phonetic contexts that a single codebook wouldn't handle.

For deeper background on VQ and RVQ, [Kyutai's codec explainer](https://kyutai.org/codec-explainer) is excellent. I also built a [VQ-VAE on images](https://github.com/taresh18/vq-vae) to get visual intuition for codebook dynamics.

## The Architecture

The model architecture is inspired by [DAC](https://arxiv.org/abs/2306.06546) (Descript Audio Codec). An Autoencoder composed of a strided convolutional encoder, RVQ in the middle, and a mirror decoder.

![nano-codec architecture](arch.png)

### Encoder

Each encoder block does two things - look around, then compress. The "look around" part matters because a strided convolution is about to throw away most of the time axis. If each position can only see a few neighboring samples, it has no idea what's worth keeping.

Three dilated residual units (dilations 1, 3, 9) handle this. The dilations spread the convolution's reach without adding parameters. By the time the strided conv fires, each position has about 5ms of surrounding context which is enough to capture meaningful audio structure. Four blocks with strides 2, 4, 4, 4 give the total 128x compression, producing a sequence of 64-dimensional latent vectors that feed the quantizer.

### Snake Activation

Every activation in the network is Snake:

```
snake(x) = x + (1/alpha) * sin(alpha * x)^2
```

Audio oscillates. That's what sound *is*. ReLU zeros out negative activations, which is a problem when the intermediate representations still carry oscillatory structure from the audio. Snake adds a learnable periodic component on top of the identity. The `alpha` parameter (one per channel, learnable) controls the frequency of this periodic component. The idea is that different channels can specialize for different time scales, from slow pitch-level patterns to fast consonant textures.

### The Quantizer

The quantizer module consists of eight RVQ levels, each with 1024 codebook entries. To quantize a latent vector, we need to find which codebook entry is closest to it, which can be computed using Euclidean distance between the encoder's continuous output `z` and each codebook entry `c` and picking the smallest. The squared distance expands as:

$$||z - c||^2 = ||z||^2 + ||c||^2 - 2(z \cdot c)$$

Computing this against all 1024 codebook entries in 64 dimensions is expensive. Two design choices help to mitigate this:

**Factorized codebooks.** Each level projects the 64-dim latent down to 8 dimensions before searching, then projects back after. The codebook lives in this smaller 8-dim space, making distance computation 8x cheaper. This projection also turns out to be critical for training stability (more on that shortly).

**L2 normalization.** Codebook entries with large magnitudes "attract" most embeddings regardless of direction, and entries with small magnitudes go unused. Normalizing both the projected embedding and codebook entries to unit length puts every entry on equal footing, competing purely on direction. It also simplifies the math - when `||z|| = ||c|| = 1`, the distance formula reduces to:

$$||z - c||^2 = 2 - 2(z \cdot c)$$

Minimizing this distance becomes maximizing the dot product i.e., cosine similarity which is just a single matmul.

The resulting bitrate is 128 frames/sec × 8 codebooks × 10 bits per codebook (log2(1024)) = 10.2 kbps. For comparison, MP3 runs at 128 kbps, so nano-codec compresses about 12x more aggressively.

## Loss Functions

With the architecture defined, the next question is what to optimize. I started with MSE and found it nearly useless for audio. A one-sample time shift which is completely inaudible to humans, still produces massive MSE. The loss function needs to compare what the audio *sounds like*, not what the waveform *looks like*.

**Multi-resolution STFT loss** gets us partway there. STFT breaks audio into overlapping windows and decomposes each window into its frequency components, essentially asking "what frequencies are present and how loud are they?" at each moment in time. There's a tradeoff in window size - short windows (256 samples) give precise timing but blur frequencies whereas long windows (2048 samples) give precise frequencies but blur timing. Rather than picking one, the loss computes STFT at four window sizes (256, 512, 1024, 2048) for both the original and reconstructed audio, and compares all of them.

**Multi-scale mel loss** goes a step further. It applies the same STFT decomposition but then passes the result through a mel filterbank which is a set of triangular filters spaced according to how humans perceive pitch. Low frequencies get more resolution (we're very sensitive to the difference between 100Hz and 200Hz), high frequencies get less (8000Hz and 8100Hz sound nearly identical).

The total reconstruction loss is a weighted combination of all three, with MSE kept at a low weight as a regularizer. On adding mel loss, reconstructions got noticeably sharper immediately. My guess is that without it, the STFT loss was treating all frequencies equally, so the model spent capacity getting high-frequency details right that humans barely notice, while under-optimizing the low-frequency content that we're most sensitive to.


## The Codebook Problem

Encoder and decoder were straightforward CNN engineering, but the codebook was the real challenge.

Encoder updates via gradient descent every batch, moving its outputs through latent space. The codebook needs to track where those outputs are, placing entries close to them. If the codebook can't keep up, the gap grows, quantization error increases, and training falls apart.

I started with **EMA updates** where each codebook entry maintains a running average of the embeddings assigned to it, slowly drifting toward their mean. The encoder, meanwhile, takes full gradient steps which results in the encoder outrunning the codebook. The distance between encoder outputs and their nearest codebook entries (commitment loss) explodes early during training:

![Commitment loss exploding with EMA updates](commit_loss_explode.png)

What actually helped was making the codebook an `nn.Parameter` and training it with two explicit loss terms:

```python
commitment_loss = F.mse_loss(z, z_q.detach())   # push encoder toward codebook (gradients to encoder only)
codebook_loss   = F.mse_loss(z.detach(), z_q)    # push codebook toward encoder (gradients to codebook only)
```

The `.detach()` controls which side receives gradients. The encoder and codebook each get pulled toward the other, but through separate loss terms. The optimizer handles both at the same learning rate with the same momentum.

Even with this, training was unstable when quantizing in the full 64-dimensional latent space. Most codebook entries were far from any encoder output and received sparse, noisy gradients.

This is where factorized codebooks from earlier paid off. In 8 dimensions, 1024 entries can cover the space much more densely. Entries get assigned more consistently, gradients are cleaner, and codebook utilization improved significantly.

![Stable commitment loss with gradient-based updates + factorized codebooks](commit_loss.png)

## Training

I trained the model on LibriSpeech clean-100 (~100 hours of clean audio), chunked into 1-second segments at 16kHz for 180k steps on a single 3090, taking ~20 hours.

First run took about an hour per epoch, which upon profiling showed that backward pass was 74% of batch time. The spectral losses which I assumed were the bottleneck contributed only 0.005 seconds per batch. A combination of `torch.compile`, AMP mixed precision, and TF32 matmul precision brought the epoch time down to 25 minutes, fast enough to iterate on codebook experiments without waiting overnight.
![Reconstruction loss over training](recon_loss.png)

The reconstruction loss drops steeply in the first few hundred steps then gradually levels off. Most of the learning happens early. The long tail is the model squeezing out incremental improvements in fine detail. By 180k steps the loss had mostly plateaued, suggesting more training time alone wouldn't help much without architectural changes.

One gotcha worth mentioning separately - spectral losses must run in float32. I initially had them inside `torch.amp.autocast` and got NaN losses. FFTs accumulate tiny rounding errors across hundreds of frequency bins and in float16 those compound fast. Moving loss computation outside autocast fixed it.

## Results

Some samples comparing original audio with what the codec reconstructs:

Original: <audio src="aud_6_original.wav"> Reconstructed: <audio src="aud_6_recon.wav">

Original: <audio src="aud_8_original.wav"> Reconstructed: <audio src="aud_8_recon.wav">

The codec produces recognizable speech. You can understand what's being said and speaker identity is roughly preserved. Surprisingly decent for 20 hours of training on a small dataset, but not close to production quality yet.

The mel spectrograms tell the story more precisely:

![aud_8 mel spectrogram comparison](aud_8_mel.png)
![aud_6 mel spectrogram comparison](aud_6_mel.png)

aud_8 is the better reconstruction. The low-frequency harmonic patterns (bottom half of the spectrogram, where pitch and formants live) match closely between original and reconstructed. You can see the horizontal stripes lining up. The upper frequencies lose some definition but the overall structure holds.

aud_6 is where the codec struggles more. The high-frequency content (top half) is noticeably blurrier in the reconstruction. Sharp features in the original get smeared, and that's what you hear as the muffled quality. Consonants like 's' and 't' live in those upper frequencies, which is why they suffer most. This sample also has more energy and dynamic range, which the codec has a harder time preserving.

The early RVQ levels carry most of the intelligibility. Decode with only the first 4 levels and the speech is still understandable but sounds rough. The later levels add warmth and detail.

## Things to Explore Next

The muffled quality points to a clear set of improvements. An **adversarial discriminator** would likely be the single biggest win. It teaches the decoder what real audio sounds like, capturing sharp consonant attacks and natural texture that spectral losses miss.

**Splitting the codebook** into parallel branches for content and acoustics would let each focus on one job. Right now the serial RVQ forces each level to balance both. A parallel split, one branch for what's being said, one for how it sounds, removes that compromise.

Making the **decoder causal** would enable streaming. The decoder starts producing audio the moment first tokens arrive instead of waiting for the complete input. For voice agents where every millisecond of latency matters, this is essential.

The trained model is available on [HuggingFace](https://huggingface.co/taresh18/nano-codec) and the full training code is on [GitHub](https://github.com/taresh18/nano-codec).

## References

- [Audio Codec Explainer (Kyutai)](https://kyutai.org/codec-explainer)
- [High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546)
- [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
