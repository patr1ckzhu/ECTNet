# ECTNet

Enhanced CTNet for EEG-based motor imagery classification, targeting 8-channel OpenBCI deployment.

## Project Context

- Based on CTNet (CNN + Transformer for MI-EEG classification)
- Original paper: Zhao et al., Sci Rep 14, 20237 (2024)
- Target hardware: OpenBCI with 8 EEG channels
- Datasets: BCI Competition IV-2a (22ch, 4-class) and IV-2b (3ch, 2-class)

## Environment

- Conda environment: `eeg-moabb`
- Python 3.10, PyTorch, MNE, einops, scikit-learn

## Key Files

- `model.py` — CTNet model definition (PatchEmbeddingCNN, Transformer, EEGTransformer)
- `train.py` — Training logic (ExP class, main loop, hyperparameters)
- `main_8_channels.py` — 8-channel variant
- `main_2class_left_right.py` — 2-class (left/right hand)
- `main_2class_left_right_8ch.py` — 2-class on 8 channels (closest to deployment)
- `main_3class_left_right_feet.py` — 3-class (left/right/feet)
- `channel_selector.py` — Channel selection tool for reducing 22ch → 8ch
- `utils.py` — Data loading, metrics, GradCAM visualization
- `preprocessing_for_2a.py` / `preprocessing_for_2b.py` — GDF → MAT preprocessing

## Workflow

1. Download BCI IV-2a/2b datasets into `BCICIV_2a_gdf/` and `BCICIV_2b_gdf/`
2. Run preprocessing: `python preprocessing_for_2a.py` / `python preprocessing_for_2b.py`
3. Train: `python train.py` (or 8-channel variants)

## Conventions

- Model architecture is defined in `model.py`, training logic in `train.py`
- Variant training scripts (main_*.py) still contain inline model definitions
- Data is stored as .mat files after preprocessing
- Models are saved as .pth files in `models/` directory
