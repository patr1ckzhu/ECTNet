# ECTNet

Enhanced Convolutional Transformer Network for EEG-based motor imagery classification, targeting 8-channel OpenBCI deployment.

Based on [CTNet](https://github.com/snailpt/CTNet) (Zhao et al., Sci Rep 2024).

## Results

### BCI Competition IV-2a (22 channels, 4-class)

| Subject | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 86.81 | 71.53 | 90.97 | 78.12 | 77.78 | 63.54 | 88.89 | 85.76 | 86.11 | **81.06** |
| Kappa | 82.41 | 62.04 | 87.96 | 70.83 | 70.37 | 51.39 | 85.19 | 81.02 | 81.48 | **74.74** |

### BCI Competition IV-2a — 8 channels, 2-class (A2, hardware deployment config)

| Subject | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 86.81 | 76.39 | 98.61 | 84.03 | 92.36 | 78.47 | 88.19 | 95.83 | 91.67 | **88.04** |
| Kappa | 73.61 | 52.78 | 97.22 | 68.06 | 84.72 | 56.94 | 76.39 | 91.67 | 83.33 | **76.08** |

8 channels selected from sensorimotor cortex: **C3, C4, Cz, FCz, CP1, CP2, FC3, FC4**

### BCI Competition IV-2b (3 channels, 2-class)

| Subject | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 76.56 | 69.64 | 84.69 | 96.88 | 95.00 | 86.25 | 93.12 | 93.75 | 88.75 | **87.18** |
| Kappa | 53.12 | 39.29 | 69.38 | 93.75 | 90.00 | 72.50 | 86.25 | 87.50 | 77.50 | **74.37** |

### Channel Attention & 8ch Selection

- Channel Attention (SE-Net) on 22ch: 81.29% (no improvement over baseline)
- Manual 8ch selection: **75.50%** (C3, C4, Cz, FCz, CP1, CP2, FC3, FC4)
- CA-learned 8ch selection: 73.50%
- Hardware deployment recommendation: manual 8ch

## Architecture

ECTNet = PatchEmbeddingCNN (temporal conv + channel attention + spatial conv) + Transformer encoder

- **Channel Attention**: SE-Net style, learns per-electrode importance weights
- **Temporal Conv**: EEGNet-inspired, extracts frequency features per channel
- **Spatial Conv**: Depth-wise, fuses across EEG electrodes
- **Transformer**: Multi-head self-attention on temporal patches
- **Preprocessing**: Optional bandpass (4-40Hz) + notch (50Hz) filtering via `eeg_filter()`, shared between training and inference
- **Checkpoint**: Saves model + normalization params (mean/std) for deployment

## Project Structure

```
ECTNet/
├── model.py                 # Model architecture (ChannelAttention, PatchEmbeddingCNN, Transformer)
├── train.py                 # Training script with parallel subject training (mp.Pool)
├── utils.py                 # Data loading, metrics, filtering, GradCAM
├── preprocessing/
│   ├── preprocessing_for_2a.py   # BCI IV-2a: GDF → MAT
│   └── preprocessing_for_2b.py   # BCI IV-2b: GDF → MAT
├── acquisition/
│   ├── paradigm.py               # PsychoPy MI experiment (LSL markers)
│   ├── recorder.py               # LSL multi-stream recorder (EEG + markers → .npz)
│   ├── make_dataset.py           # Recording → .mat converter (filter, epoch, split)
│   └── recordings/               # Recorded .npz files
├── experiments/
│   ├── train_8ch.py              # 8-channel experiment
│   ├── train_2class.py           # 2-class (left/right)
│   ├── train_2class_8ch.py       # 2-class on 8 channels
│   └── train_3class.py           # 3-class (left/right/feet)
├── tools/
│   └── channel_selector.py       # Channel selection utility
└── docs/
    └── openbci_setup_guide.md    # OpenBCI Cyton acquisition guide
```

## Quick Start

### Environment

```bash
conda activate eeg-moabb
pip install torch mne einops torchsummary scikit-learn pandas scipy opencv-python-headless
```

### Data Preparation

1. Download [BCI Competition IV-2a & 2b](https://www.bbci.de/competition/iv/) datasets
2. Place GDF files in `BCICIV_2a_gdf/` and `BCICIV_2b_gdf/`
3. Run preprocessing:

```bash
python preprocessing/preprocessing_for_2a.py
python preprocessing/preprocessing_for_2b.py
```

### Training

```bash
# BCI IV-2a: 22-channel, 4-class (default)
python train.py A

# BCI IV-2a: 8-channel, 2-class left/right (hardware deployment)
python train.py A2

# BCI IV-2b: 3-channel, 2-class
python train.py B

# Custom OpenBCI data: 8-channel, 2-class
python train.py C
```

Training uses `torch.compile(mode='reduce-overhead')` (CUDA graphs, Linux) + parallel subject processing (N_WORKERS=3). On RTX 5080: A2(8ch) ~3min, A(22ch) ~10min, B(3ch) ~7min for all 9 subjects.

### Custom Data Acquisition (OpenBCI Cyton)

Collect your own MI-EEG data with an 8-channel OpenBCI Cyton board. See [`docs/openbci_setup_guide.md`](docs/openbci_setup_guide.md) for full instructions.

```bash
# Terminal 1: record EEG + markers
python acquisition/recorder.py

# Terminal 2: run experiment paradigm (30 trials default, configurable)
python acquisition/paradigm.py
python acquisition/paradigm.py -n 25    # 50 trials

# Convert recording to training format (supports merging multiple files)
python acquisition/make_dataset.py --input acquisition/recordings/rec1.npz rec2.npz

# Train on custom data
python train.py C
```

## Reference

```
Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for
EEG-based motor imagery classification. Sci Rep 14, 20237 (2024).
```
