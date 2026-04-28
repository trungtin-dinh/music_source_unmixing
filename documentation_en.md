## Table of Contents

1. [The Music Source Separation Problem](#1-the-music-source-separation-problem)
2. [Classical Approaches: ICA and NMF](#2-classical-approaches-ica-and-nmf)
3. [The Short-Time Fourier Transform for Audio Analysis](#3-the-short-time-fourier-transform-for-audio-analysis)
4. [The Real-Valued FFT and the Magnitude Spectrum](#4-the-real-valued-fft-and-the-magnitude-spectrum)
5. [Deep Learning for Source Separation: Overview](#5-deep-learning-for-source-separation-overview)
6. [HTDemucs: a Hybrid Time-Frequency Architecture](#6-htdemucs-a-hybrid-time-frequency-architecture)
7. [MDX-Net: a Frequency-Domain U-Net](#7-mdx-net-a-frequency-domain-u-net)
8. [Training Objectives for Source Separation](#8-training-objectives-for-source-separation)
9. [Gain Control and Stem Remixing](#9-gain-control-and-stem-remixing)
10. [Peak Normalisation After Mixing](#10-peak-normalisation-after-mixing)

---

## 1. The Music Source Separation Problem

Music source separation is a specific instance of the **Blind Source Separation (BSS)** problem. A recorded mixture $x(t)$ is the superposition of $J$ latent source signals $s_j(t)$:

$$
x(t) = \sum_{j=1}^{J} s_j(t)
$$

where in the music context the four canonical sources are **vocals**, **drums**, **bass**, and **other** (melodic instruments). The term "blind" reflects that neither the mixing process nor the individual sources are observed — only $x(t)$ is available at inference time.

This problem is fundamentally **underdetermined**: with a single mixture channel and four sources, the system has far more unknowns than equations at every time instant. Recovery is only possible by exploiting additional structure — statistical properties of the sources, their spectral characteristics, or learned priors encoded by a neural network.

In the stereo case, the mixture has two channels $(x_L(t), x_R(t))$, which provides some spatial cues (inter-channel level and phase differences), but the problem remains underdetermined since there are still more sources than channels. Modern deep learning methods sidestep the underdetermined formulation entirely by treating separation as a **supervised regression problem**: given a large dataset of isolated stems and their mixtures, a network is trained to map a mixture directly to its constituent sources.

---

## 2. Classical Approaches: ICA and NMF

Understanding classical methods contextualises the advance that deep learning represents.

### 2.1 Independent Component Analysis

**Independent Component Analysis (ICA)** assumes that the sources $s_j$ are mutually statistically independent and non-Gaussian. Given a multichannel mixture $\mathbf{x}(t) = A\,\mathbf{s}(t)$ where $A$ is an unknown mixing matrix, ICA estimates the **demixing matrix** $W = A^{-1}$ by maximising a measure of statistical independence among the recovered signals $\hat{\mathbf{s}}(t) = W\mathbf{x}(t)$.

The canonical objective is to maximise **non-Gaussianity** of the separated signals — by the Central Limit Theorem, a sum of independent variables is more Gaussian than any of its components individually, so reversing the mixing requires maximising non-Gaussianity. This is measured by negentropy or by kurtosis. Applied per frequency bin in the STFT domain, this yields **Frequency-Domain ICA (FDICA)**.

ICA suffers from two fundamental indeterminacies: a **permutation ambiguity** (the ordering of estimated sources across frequency bins is arbitrary) and a **scale ambiguity** (each source is recoverable only up to a scalar factor). More critically, ICA requires **at least as many microphones as sources**, and it assumes a static linear mixing model — neither of which holds for a single recorded stereo track.

### 2.2 Non-negative Matrix Factorisation

**Non-negative Matrix Factorisation (NMF)** operates on the **magnitude spectrogram** $V \in \mathbb{R}_{\geq 0}^{F \times T}$, where $F$ is the number of frequency bins and $T$ is the number of time frames. NMF seeks:

$$
V \approx W H, \quad W \in \mathbb{R}_{\geq 0}^{F \times K},\; H \in \mathbb{R}_{\geq 0}^{K \times T}
$$

where $K$ is the number of latent components. The columns of $W$ are **spectral templates** (frequency profiles of elementary sources) and the rows of $H$ are their **temporal activations**. Non-negativity is enforced as a physically motivated prior: magnitude spectra are inherently non-negative.

The factorisation is typically found by minimising the **Kullback-Leibler divergence**:

$$
D_\text{KL}(V \,\|\, WH) = \sum_{f,t} \left[ V_{ft} \log \frac{V_{ft}}{(WH)_{ft}} - V_{ft} + (WH)_{ft} \right]
$$

via multiplicative update rules that maintain non-negativity. NMF can separate **harmonic** (pitched) from **percussive** (broadband, impulsive) components by exploiting differences in the structure of their spectral templates. However, NMF is a shallow, memoryless model with no temporal context, and its performance on realistic polyphonic music is substantially below that of deep learning models.

---

## 3. The Short-Time Fourier Transform for Audio Analysis

The **Short-Time Fourier Transform (STFT)** is the central analysis tool for both the spectrogram display and the frequency-domain processing branch of the neural network. For a discrete-time signal $x[n]$, the STFT is:

$$
X[m, k] = \sum_{n=-\infty}^{+\infty} x[n]\, w[n - m H]\, e^{-j 2\pi k n / N}
$$

where $w[\cdot]$ is the analysis window of length $N$ (here $N = 2048$), $H$ is the hop size in samples (here $H = 512$), $m$ is the frame index, and $k \in \{0, 1, \ldots, N/2\}$ is the frequency bin index.

### 3.1 The Hann Window

The **Hann window** applied here is:

$$
w[n] = \frac{1}{2}\!\left(1 - \cos\frac{2\pi n}{N-1}\right), \quad n = 0, 1, \ldots, N-1
$$

It achieves a good trade-off between **main-lobe width** (frequency resolution) and **side-lobe attenuation** (spectral leakage suppression), with side-lobes decaying at $-18\,\text{dB/octave}$. This is important for music, where a strong spectral component (e.g. a bass note at 80 Hz) must not mask weaker nearby components (e.g. a melody note at 100 Hz) through spectral leakage.

The **overlap** between successive frames is $(N - H)/N = 75\%$. This high overlap satisfies the **Constant Overlap-Add (COLA)** condition required for perfect reconstruction in the Overlap-Add (OLA) synthesis framework. In practice, the Hann window is COLA-compliant at both 50% and 75% overlap, ensuring that summing windowed output frames produces a flat reconstruction envelope.

### 3.2 The Spectrogram Display

The displayed spectrogram uses the **log-compressed magnitude**:

$$
\tilde{X}[m, k] = \log\!\left(1 + |X[m, k]|\right)
$$

The $\log(1 + \cdot)$ transform serves two purposes. First, it avoids the numerical singularity of $\log(0)$ at silent frequency bins without requiring an explicit noise floor. Second, it compresses the large dynamic range of music: the ratio between the loudest and quietest components can exceed $10^6$ in amplitude ($120\,\text{dB}$), which would make a linear-scale spectrogram visually uninterpretable.

The **frequency axis** spans $[0, f_s/2]$ with $N/2 + 1 = 1025$ uniformly spaced bins, giving a frequency resolution of $\Delta f = f_s / N$. At $f_s = 44100\,\text{Hz}$, this yields $\Delta f \approx 21.5\,\text{Hz/bin}$. The **time axis** step is $\Delta t = H / f_s \approx 11.6\,\text{ms}$, giving approximately 86 frames per second.

One subtlety: `torch.stft` is called with `center=True`, which symmetrically pads the signal by $N/2$ samples before framing. This ensures that frame $m = 0$ is centred on sample $n = 0$, improving boundary alignment. The displayed time axis uses the approximation $t_m = m \cdot H / f_s$; the true centred time is $t_m + N/(2f_s)$, a fixed offset of $\approx 23\,\text{ms}$ that does not affect perceptual interpretation.

---

## 4. The Real-Valued FFT and the Magnitude Spectrum

The **real-valued FFT** (`numpy.fft.rfft`) exploits the Hermitian symmetry of the DFT of a real signal: $X[N - k] = X^*[k]$. For a signal of length $N$, only the first $\lfloor N/2 \rfloor + 1$ coefficients are independent, so `rfft` returns a vector of that length, halving computation and storage compared to the full DFT.

The **normalised magnitude spectrum** is:

$$
|S[k]| = \frac{|\text{rfft}(x)[k]|}{N}, \quad k = 0, 1, \ldots, \left\lfloor \frac{N}{2} \right\rfloor
$$

Division by $N$ converts raw DFT coefficients to the **amplitude spectrum** in the same physical units as the input. For a pure tone $x[n] = A \cos(2\pi f_0 n / f_s)$, the peak at bin $k_0 = \text{round}(f_0 N / f_s)$ has height $A/2$ in the one-sided spectrum (the factor of 2 comes from combining the two symmetric DFT bins). The **frequency axis** is:

$$
f_k = \frac{k \cdot f_s}{N}, \quad k = 0, 1, \ldots, \left\lfloor \frac{N}{2} \right\rfloor
$$

equivalently computed by `numpy.fft.rfftfreq(N, d=1/f_s)`.

### 4.1 Display Downsampling

For a long audio file the rfft spectrum contains hundreds of thousands of bins, producing an unresponsive interactive plot. The curve is **decimated** to at most 5000 points by retaining every $\lceil N_\text{bins} / 5000 \rceil$-th sample. This is a **stride decimation without prior anti-aliasing filtering**, which introduces **aliasing in the display**: spectral peaks that fall between retained samples may be missed or distorted in the plot. This is a display-only approximation — the underlying separation and remixing operate on the full-resolution signal throughout.

---

## 5. Deep Learning for Source Separation: Overview

Modern neural source separation is formulated as **supervised mask estimation**. A neural network $f_\theta$ takes the mixture spectrogram $X[m,k]$ (or the raw waveform) and predicts a **soft mask** $\mathcal{M}_j[m,k] \in [0,1]$ per source $j$:

$$
\hat{S}_j[m, k] = \mathcal{M}_j[m, k] \cdot X[m, k]
$$

The masks collectively distribute the mixture energy among sources, with $\sum_j \mathcal{M}_j[m,k] \leq 1$ at each time-frequency bin. The estimated time-domain source is recovered by inverse STFT.

Early deep methods operated exclusively in the **STFT magnitude domain**, reusing the mixture phase for reconstruction. This **phase approximation** introduces **musical noise** — random-phase artefacts audible as a metallic shimmer. The evolution towards **waveform-domain models** (Demucs v1–v3) addressed this by processing the raw waveform directly, implicitly estimating phase. The most recent generation — **hybrid models** (HTDemucs) — processes both representations simultaneously, combining their complementary strengths.

---

## 6. HTDemucs: a Hybrid Time-Frequency Architecture

**HTDemucs** (Rouard et al., 2023) is built on a **dual-branch encoder-decoder** that processes the mixture simultaneously in the **time domain** and the **STFT frequency domain**, fusing information at the bottleneck through a **cross-domain transformer**.

### 6.1 The Encoder-Decoder U-Net Backbone

Each branch follows a **U-Net** topology (Ronneberger et al., 2015): a contracting **encoder** followed by an expansive **decoder**, linked by **skip connections** that carry feature maps from each encoder level to the corresponding decoder level.

**Encoder.** Each layer $\ell$ applies a strided convolution with stride $S_\ell$, downsampling the input and expanding the receptive field:

$$
\mathbf{h}_\ell = \sigma\!\left(W_\ell^{(\text{enc})} * \mathbf{h}_{\ell-1}\right)
$$

where $\sigma$ is a non-linear activation (GELU). The total downsampling factor after $L$ encoder layers is $\prod_\ell S_\ell$, determining the bottleneck resolution.

**Decoder.** Each layer applies a transposed convolution (learned upsampling by $S_\ell$) and concatenates the corresponding skip connection:

$$
\mathbf{h}'_\ell = \sigma\!\left(W_\ell^{(\text{dec})} *^\top [\mathbf{h}'_{\ell+1};\, \mathbf{h}_\ell]\right)
$$

Skip connections preserve fine-grained detail that would otherwise be destroyed through the bottleneck compression, enabling high-fidelity reconstruction of transients and fine spectral structure.

The **time-domain branch** processes the raw waveform. The **frequency-domain branch** processes the complex STFT of the mixture, treating real and imaginary parts as a two-channel feature map. In the frequency branch, **layer normalisation is applied independently per frequency bin** — reflecting that the statistics of low-frequency bins (dominated by bass fundamentals) differ substantially from those of high-frequency bins (harmonics, noise floor), so global normalisation would be harmful.

### 6.2 Cross-Domain Transformer Bottleneck

At the bottleneck — after full downsampling in both branches — HTDemucs inserts a **cross-domain transformer** that enables the two branches to share information via **cross-attention**:

$$
\text{CrossAttn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

where the **queries** $Q$ originate from one domain's bottleneck features and the **keys** $K$ and **values** $V$ from the other domain's bottleneck features. This bidirectional exchange allows the model to, for example, use the precise harmonic structure visible in the frequency domain to sharpen temporal envelope estimation in the time domain, and vice versa.

In addition to cross-domain attention, each branch also applies **self-attention** at the bottleneck, granting global temporal context before the cross-domain exchange. Together, these attention layers enable the network to reason about long-range dependencies — for instance, tracking a melody line that persists over several seconds. The full attention block follows the standard **Transformer** structure (Vaswani et al., 2017): multi-head attention, residual connection, layer normalisation, and a position-wise feed-forward network.

### 6.3 HTDemucs Fine-Tuned (htdemucs_ft)

The `htdemucs_ft` variant is obtained by **fine-tuning** the base HTDemucs model after pre-training on MUSDB18-HQ plus additional data. Fine-tuning uses a lower learning rate and a source-specific loss weighting that allocates more training signal to the most acoustically challenging stems (vocals and bass). This stage consistently improves Signal-to-Distortion Ratio (SDR) on those stems across diverse musical genres.

---

## 7. MDX-Net: a Frequency-Domain U-Net

**MDX-Net** (Kim et al., 2021), winner of the 2021 Music Demixing Challenge, is a purely frequency-domain separator — it has no time-domain branch.

Its core building block is the **TFC-TDF** (Time-Frequency Convolution — Time-Distributed Fully Connected) unit, combining two complementary operations within each encoder/decoder stage:

- **TFC layers** apply 2D convolutions over the joint time-frequency plane, extracting local spectral-temporal patterns.
- **TDF layers** apply dense linear projections independently at each time frame across the **full frequency axis**. This is equivalent to a learnable frequency-selective filterbank applied uniformly in time — a data-driven generalisation of classical filterbank analysis.

The TFC-TDF U-Net predicts a **complex-valued mask**, operating on both real and imaginary parts of the STFT, thereby jointly estimating source magnitude and phase rather than reusing the mixture phase.

The `mdx_extra` variant is a **model ensemble** trained with additional data and augmentation strategies. Ensembling averages the outputs of multiple independently trained models, reducing variance and improving robustness across diverse musical genres. The architecture of each ensemble member is the same TFC-TDF U-Net; the gains come from training diversity rather than architectural differences.

---

## 8. Training Objectives for Source Separation

The loss function directly determines perceptual quality.

**$\ell_1$ waveform loss** minimises the mean absolute error between estimated and reference waveforms:

$$
\mathcal{L}_{\ell_1} = \frac{1}{T} \sum_{t=0}^{T-1} \left|\hat{s}_j[t] - s_j[t]\right|
$$

The $\ell_1$ norm is more robust to occasional large errors than $\ell_2$ and avoids the over-smoothed, "muddy" output typical of MSE-trained models.

**Multi-scale STFT loss** evaluates reconstruction quality simultaneously at multiple time-frequency resolutions:

$$
\mathcal{L}_\text{STFT} = \sum_r \left\| \log |X_r[\hat{s}]| - \log |X_r[s]| \right\|_F
$$

where $r$ indexes different STFT parameter settings (different values of $N$ and $H$). Multiple scales capture both fine temporal structure (small $N$, coarse frequency resolution) and spectral detail (large $N$, fine frequency resolution), which no single STFT can provide simultaneously.

**Signal-to-Distortion Ratio (SDR)**, the standard evaluation metric, measures the ratio of reference signal power to residual error power:

$$
\text{SDR} = 10 \log_{10}\!\left(\frac{\|s_j\|^2}{\|\hat{s}_j - s_j\|^2}\right) \; [\text{dB}]
$$

State-of-the-art models achieve SDR values of 8–10 dB on the MUSDB18-HQ benchmark for the vocals stem. An SDR of $0\,\text{dB}$ means the error has the same power as the signal; each additional $3\,\text{dB}$ roughly halves the relative error energy.

---

## 9. Gain Control and Stem Remixing

After separation, the four stems are recombined with independent **gain adjustments** to produce a custom remix. Gain is expressed in **decibels (dB)**, the standard unit for perceptual amplitude scaling in audio engineering.

The **decibel scale** for amplitude is:

$$
G_\text{dB} = 20 \log_{10}(g)
$$

where $g$ is the linear amplitude multiplier. The inverse conversion — from a dB setting to the linear multiplier applied to audio samples — is:

$$
g = 10^{G_\text{dB}\, /\, 20}
$$

The factor of $20$ (rather than $10$) appears because the decibel was originally defined for **power**, and power is proportional to the square of amplitude: $P \propto g^2$, so $G_\text{dB} = 10 \log_{10}(g^2) = 20 \log_{10}(g)$.

The gain range of the sliders is $[-60, +12]\,\text{dB}$. At $-60\,\text{dB}$, $g = 10^{-3} = 0.001$ — a reduction to $0.1\%$ of original amplitude, perceptually near-silence. At $+12\,\text{dB}$, $g \approx 3.98$ — a nearly four-fold amplitude increase. The remix signal at sample $t$ and channel $c$ is:

$$
y_c[t] = \sum_{j \in \mathcal{J}} g_j \cdot s_{j,c}[t]
$$

where $\mathcal{J} = \{\text{vocals, drums, bass, other}\}$. With all gains at $0\,\text{dB}$ ($g_j = 1$ for all $j$), this reconstructs the original mixture exactly, since the network is trained so that $\sum_j s_j \approx x$. Any deviation from $0\,\text{dB}$ on any stem produces a creative remix that cannot be obtained from the original mixture alone.

---

## 10. Peak Normalisation After Mixing

After summing the gain-weighted stems, the remix signal may exceed the representable amplitude range $[-1, 1]$ for floating-point audio — particularly when several stems are boosted simultaneously. **Hard clipping** (truncating samples to $\pm 1$) introduces severe nonlinear harmonic distortion, audible as an unpleasant buzz.

To prevent clipping, **peak normalisation** is applied only when necessary:

$$
y_\text{norm}[t] = \frac{y[t]}{\max_t |y[t]|} \quad \text{if } \max_t |y[t]| > 1
$$

This scales the entire signal uniformly so that the loudest sample reaches exactly $0\,\text{dBFS}$ (decibels relative to full scale). The **relative dynamics** among all stems and all time instants are preserved exactly — only the global level changes.

This is a form of **peak limiting** with an infinite compression ratio above a threshold of 1.0, applied globally rather than sample-by-sample. A more sophisticated approach would use a **look-ahead limiter** with finite attack and release times, or a **dynamic range compressor**, avoiding the abrupt level jump that global normalisation can cause when a brief transient forces $\max_t|y[t]| \gg 1$. For this application — interactive creative remixing — the simpler global normalisation is appropriate.

---

## References

- Rouard, S., Massa, F., & Défossez, A. (2023). Hybrid Transformers for Music Source Separation. *ICASSP 2023*.
- Défossez, A., Usunier, N., Bottou, L., & Bach, F. (2021). Music Source Separation in the Waveform Domain. *arXiv:1911.13254*.
- Kim, Y., Choi, K., Choi, M., Kim, B., & Won, M. (2021). Kuielab-MDX-Net: A Two-Stream Neural Network for Music Demixing. *ISMIR Workshop on Music Source Separation*.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.
- Vaswani, A., Shazeer, N., Parmar, N., Jiang, Z., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*.
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401, 788–791.
- Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications. *Neural Networks*, 13(4–5), 411–430.
- Rafii, Z., Liutkus, A., Stöter, F.-R., Mimilakis, S. I., & Bitteur, R. (2017). MUSDB18 — a corpus for music separation. *Zenodo*.
