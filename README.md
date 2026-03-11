# Midicoth — Micro-Diffusion Compression

**Lossless data compression via Binary Tree Tweedie Denoising**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Midicoth is a lossless text compressor that introduces *micro-diffusion* — a multi-step score-based reverse diffusion process implementing Tweedie's empirical Bayes formula — into a cascaded statistical modeling pipeline. It treats Jeffreys-prior smoothing as a shrinkage operator toward uniform and reverses it through binary tree denoising with variance-aware James-Stein shrinkage, achieving compression ratios that outperform xz, zstd, Brotli, and bzip2 on all tested inputs.

**No neural network. No training data. No GPU. ~2,000 lines of C.**

---

## Results

### alice29.txt — 152 KB (Canterbury Corpus)

![alice29.txt compression comparison](charts/chart_alice29.png)

### enwik8 — 100 MB (Large Text Compression Benchmark)

![enwik8 compression comparison](charts/chart_enwik8.png)

| Benchmark | Midicoth | xz -9 | Improvement |
|-----------|----------|--------|-------------|
| **alice29.txt** (152 KB) | **2.119 bpb** | 2.551 bpb | **16.9%** |
| **enwik8** (100 MB) | **1.753 bpb** | 1.989 bpb | **11.9%** |

Midicoth outperforms all dictionary-based compressors (gzip, zstd, xz, Brotli, bzip2) on every tested input, narrowing the gap to heavyweight context-mixing systems like PAQ and CMIX.

---

## How It Works

Midicoth processes input one byte at a time through a five-layer cascade:

```
Input byte
    |
    v
[1] PPM Model (orders 0-4, PPMC exclusion, Jeffreys prior)
    |  produces 256-way distribution + confidence + order
    v
[2] Match Model (context lengths 4-16)
    |  blends in long-range repetition predictions
    v
[3] Word Model (trie + bigram)
    |  blends in word completion predictions
    v
[4] High-Order Context (orders 5-8)
    |  blends in extended context predictions
    v
[5] Micro-Diffusion (binary tree Tweedie, K=3 steps)
    |  post-blend denoising with James-Stein shrinkage
    v
Arithmetic Coder -> compressed output
```

### The Key Idea: Post-Blend Tweedie Denoising

PPM models smooth their predictions with a Jeffreys prior (0.5 per symbol, 128 total pseudo-counts). When few observations are available, this prior dominates — pulling the distribution toward uniform and wasting bits. After blending with match, word, and high-order models, additional systematic biases are introduced. We frame this as a **denoising problem**:

- **Shrinkage operator**: Jeffreys smoothing pulls the empirical distribution toward uniform: `p = (1-gamma)*q + gamma*u`, where `gamma = 128/(C+128)`
- **Reverse diffusion**: Calibration tables estimate the additive Tweedie correction `delta = E[theta|p] - E[p]`, approximating `sigma^2 * s(p)`
- **Binary tree**: Each 256-way prediction is decomposed into 8 binary decisions (MSB to LSB), enabling data-efficient calibration
- **Multi-step**: K=3 denoising steps with independent score tables, each correcting residual noise from the previous step
- **Variance-aware shrinkage**: Each correction is attenuated based on SNR = delta^2 * N / var(error). When SNR < 4, the correction is linearly shrunk toward zero, preventing noisy estimates from hurting compression

### Ablation: Component Contributions

![Ablation study](charts/chart_ablation.png)

With PPMC exclusion providing a strong base, the post-PPM layers collectively contribute **5.6-11.1%** additional improvement. The Tweedie denoiser is the most consistent contributor (+2.6-2.8% across both datasets).

### Score Magnitude vs. Noise Level

![Delta vs noise level](charts/chart_delta_vs_noise.png)

On alice29 (small file), corrections decrease monotonically with confidence — the James-Stein shrinkage suppresses noisy corrections in sparse bins. On enwik8_3M (larger file), the pattern inverts at high confidence: enough data accumulates for the shrinkage to permit large corrections where genuine signal exists. Successive steps show rapidly decreasing corrections, confirming multi-step reverse diffusion behavior.

---

## Quick Start

### Build

```bash
make          # Linux: builds mdc and ablation
make win      # Windows cross-compile: builds mdc.exe and ablation.exe
```

Requirements: GCC (or any C99 compiler) and `libm`. No other dependencies.
For Windows cross-compilation: `x86_64-w64-mingw32-gcc`.

### Compress

```bash
./mdc compress input.txt output.mdc
```

### Decompress

```bash
./mdc decompress output.mdc restored.txt
```

### Verify round-trip

```bash
./mdc compress alice29.txt alice29.mdc
./mdc decompress alice29.mdc alice29.restored
diff alice29.txt alice29.restored  # should produce no output
```

### Run ablation study

```bash
./ablation alice29.txt
```

This runs five configurations (Base PPM, +Match, +Match+Word, +M+W+HCtx, +M+W+H+Tweedie) and reports the marginal contribution of each component with round-trip verification.

### Reproduce delta-vs-noise table

```bash
gcc -O3 -march=native -o measure_delta measure_delta.c -lm
./measure_delta alice29.txt
```

---

## Architecture

### File Structure

| File | Description |
|------|-------------|
| `mdc.c` | Main compressor/decompressor driver |
| `ablation.c` | Ablation study driver |
| `measure_delta.c` | Delta-vs-noise measurement tool |
| `ppm.h` | Adaptive PPM model (orders 0-4) with PPMC exclusion and Jeffreys prior |
| `tweedie.h` | Binary tree Tweedie denoiser with James-Stein shrinkage (K=3 steps, 155K entries) |
| `match.h` | Extended match model (context lengths 4, 6, 8, 12, 16) |
| `word.h` | Trie-based word model with bigram prediction |
| `highctx.h` | High-order context model (orders 5-8) |
| `arith.h` | 32-bit arithmetic coder (E1/E2/E3 renormalization) |
| `fastmath.h` | Fast math utilities (log, exp approximations) |
| `Makefile` | Build system (Linux + Windows cross-compile) |

### Design Principles

- **Header-only modules**: Each component is a self-contained `.h` file with `static inline` functions
- **Zero external dependencies**: Only `libm` is required
- **Fully online**: All models are adaptive — no pre-training or offline parameter estimation
- **Deterministic**: Bit-exact encoder-decoder symmetry
- **Count-based**: All learnable components use simple count accumulation, avoiding overfitting from gradient-based learners

### Calibration Table Structure

The Tweedie denoiser maintains a 6-dimensional calibration table:

```
table[step][bit_context][order_group][shape][confidence][prob_bin]
       3   x    27     x     3      x  4   x     8     x   20
                    = 155,520 entries (~4.7 MB)
```

Each entry tracks four sufficient statistics:
- `sum_pred`: sum of predicted P(right) values
- `hits`: count of times the true symbol went right
- `total`: total observations
- `sum_sq_err`: sum of squared prediction errors (for James-Stein SNR)

The Tweedie correction is: `delta = hits/total - sum_pred/total`, then shrunk by `min(1, SNR/4)` where `SNR = delta^2 * total / (sum_sq_err / total)`.

---

## Compressed Format

MDC files use the `.mdc` extension:

| Offset | Size | Content |
|--------|------|---------|
| 0 | 4 bytes | Magic: `MDC7` |
| 4 | 8 bytes | Original file size (uint64, little-endian) |
| 12 | variable | Arithmetic-coded bitstream |

The format is self-contained — no external dictionaries or model files needed.

---

## Performance

| File | Size | Ratio | Speed | bpb |
|------|------|-------|-------|-----|
| alice29.txt | 152 KB | 26.5% | ~42 KB/s | 2.119 |
| enwik8_3M | 3.0 MB | 25.0% | ~40 KB/s | 2.003 |
| enwik8 | 100 MB | 21.9% | ~42 KB/s | 1.753 |

All measurements on a single CPU core (x86-64).

### Comparison with Other Approaches

| Category | Example | enwik8 bpb | Requires |
|----------|---------|------------|----------|
| Dictionary-based | xz -9 | 1.989 | CPU |
| **Midicoth (this work)** | **-** | **1.753** | **CPU** |
| Context mixing | PAQ8px | ~1.27 | CPU (hours) |
| Context mixing | CMIX v21 | ~1.17 | CPU (16-64 GB RAM) |
| LLM-based | Nacrith | 0.939 | GPU + pre-trained model |

Midicoth occupies a unique position: better than all dictionary compressors, simpler and faster than context mixers, and fully online without any pre-trained knowledge.

---

## Paper

The full technical details are described in:

> **Micro-Diffusion Compression: Binary Tree Tweedie Denoising for Online Probability Estimation**
> Roberto Tacconelli, 2026
>
> [arXiv:2603.08771](https://arxiv.org/abs/2603.08771)

Key references:
- Efron, B. (2011). *Tweedie's formula and selection bias.* JASA, 106(496):1602-1614.
- James, W. and Stein, C. (1961). *Estimation with quadratic loss.* Proc. 4th Berkeley Symposium.
- Ho, J., Jain, A., Abbeel, P. (2020). *Denoising diffusion probabilistic models.* NeurIPS 2020.
- Cleary, J.G. and Witten, I.H. (1984). *Data compression using adaptive coding and partial string matching.* IEEE Trans. Comm., 32(4):396-402.

---

## Test Data

- **alice29.txt** (152,089 bytes): Canterbury Corpus - Lewis Carroll's *Alice's Adventures in Wonderland*.
- **enwik8** (100,000,000 bytes): First 100 MB of English Wikipedia. Download from the [Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html).

---

## Building from Source

### Prerequisites

- A C99 compiler (GCC, Clang, or MSVC)
- `make` (optional)

### Using Make

```bash
make          # builds mdc and ablation (Linux)
make win      # builds mdc.exe and ablation.exe (Windows cross-compile)
make clean    # removes all binaries
```

### Manual compilation

```bash
# Linux
gcc -O3 -march=native -o mdc mdc.c -lm
gcc -O3 -march=native -o ablation ablation.c -lm
gcc -O3 -march=native -o measure_delta measure_delta.c -lm

# Windows cross-compile
x86_64-w64-mingw32-gcc -O3 -o mdc.exe mdc.c -lm -lpthread
x86_64-w64-mingw32-gcc -O3 -o ablation.exe ablation.c -lm -lpthread
```

---

## License

Apache License 2.0 - see [LICENSE](LICENSE).

```
Copyright 2025 Roberto Tacconelli
```

---

## Citation

```bibtex
@article{tacconelli2025midicoth,
  title={Micro-Diffusion Compression: Binary Tree Tweedie Denoising
         for Online Probability Estimation},
  author={Tacconelli, Roberto},
  year={2026},
  eprint={2603.08771},
  archivePrefix={arXiv},
  primaryClass={cs.IT}
}
```

## Related Work

- [Nacrith](https://github.com/robtacconelli/Nacrith-GPU) - LLM-based lossless compression (0.939 bpb on enwik8)
- [PAQ8px](https://github.com/hxim/paq8px) - Context-mixing compressor
- [CMIX](http://www.byronknoll.com/cmix.html) - Context-mixing with LSTM
- [Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html) - enwik8 benchmark
