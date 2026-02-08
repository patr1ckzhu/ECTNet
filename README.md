# ECTNet

Enhanced Convolutional Transformer Network for EEG-based motor imagery classification, targeting 8-channel OpenBCI deployment.

Based on [CTNet](https://github.com/snailpt/CTNet) (Zhao et al., Sci Rep 2024).

## Project Structure

```
ECTNet/
├── model.py                 # CTNet model architecture
├── train.py                 # Training script (subject-specific / LOSO)
├── utils.py                 # Data loading, metrics, GradCAM
├── preprocessing/
│   ├── preprocessing_for_2a.py   # BCI IV-2a: GDF → MAT
│   └── preprocessing_for_2b.py   # BCI IV-2b: GDF → MAT
├── experiments/
│   ├── train_8ch.py              # 8-channel experiment
│   ├── train_2class.py           # 2-class (left/right)
│   ├── train_2class_8ch.py       # 2-class on 8 channels
│   └── train_3class.py           # 3-class (left/right/feet)
├── tools/
│   └── channel_selector.py       # Channel selection utility
└── docs/
    └── research_directions.md    # Research improvement plans
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
# Baseline: 22-channel, 4-class, subject-specific
python train.py

# 8-channel experiments
python experiments/train_8ch.py

# 2-class on 8 channels (closest to OpenBCI deployment)
python experiments/train_2class_8ch.py
```

## Reference

```
Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for
EEG-based motor imagery classification. Sci Rep 14, 20237 (2024).
```
