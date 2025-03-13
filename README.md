# DenoMAE 2.0: Wireless Signal Constellation Generation

This repository contains utilities for generating wireless signal constellation diagrams with different modulation types for machine learning applications such as denoising autoencoders.

## Project Overview

This toolkit generates constellation images and raw signal data for various modulation schemes with configurable noise levels. It supports both labeled and unlabeled dataset generation for training and testing denoising models.

## Files

- **funcSampleGeneration.py**: Core functions for generating wireless signal constellations.
  - AWGN (Additive White Gaussian Noise) simulation
  - GMSK modulation implementation
  - Constellation diagram generation

- **generateSamples.py**: Script to generate unlabeled constellation datasets for multiple modulations.
  - Creates both noisy and clean versions of constellation images
  - Configurable parameters for image size, number of samples, etc.

- **funcLabelSampleGeneration.py**: Functions for labeled dataset generation.
  - Similar to `funcSampleGeneration.py` but includes labeling capabilities
  - Organizes images by modulation type

- **generateLabeledSamples.py**: Script to generate labeled constellation datasets.
  - Creates datasets organized by SNR value and modulation type
  - Designed for supervised learning tasks

- **move_mix.py**: Utility for creating mixed datasets from individual source folders.
  - Balances class distribution for training and testing
  - Combines samples from different source directories

## Supported Modulation Types

- OOK (On-Off Keying)
- 4ASK, 8ASK (Amplitude Shift Keying)
- OQPSK (Offset Quadrature Phase Shift Keying)
- CPFSK (Continuous Phase Frequency Shift Keying)
- GFSK (Gaussian Frequency Shift Keying)
- 4PAM, 16PAM (Pulse Amplitude Modulation)
- DQPSK (Differential Quadrature Phase Shift Keying)
- GMSK (Gaussian Minimum Shift Keying)

## Usage

### Generate Unlabeled Dataset

```python
python generateSamples.py
```

### Generate Labeled Dataset

```python
python generateLabeledSamples.py
```

### Create Mixed Dataset

```python
python move_mix.py
```

## Configuration

Both generator scripts accept parameters that can be modified in the main block:
- `samples_per_image`: Number of signal points per constellation image
- `image_size`: Output image dimensions
- `image_num`: Number of images to generate per modulation type
- `mod_types`: List of modulation types to generate
- `SNR_dB`: Signal-to-noise ratio values (for labeled datasets)
- `set_types`: Types of outputs to generate (noisy/noiseless images, signal data)
- `base_path`: Output directory path

## Requirements

- Python 3.6+
- NumPy
- SciPy
- PIL (Python Imaging Library)
- tqdm (for progress bars)

