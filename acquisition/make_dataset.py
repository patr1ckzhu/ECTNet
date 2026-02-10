"""
Convert LSL recording (.npz) to training .mat files.

Extracts 4s epochs around markers, applies EEG filtering, and splits
into train/test sets compatible with the ECTNet training pipeline.

Usage:
    python acquisition/make_dataset.py --input acquisition/recordings/recording_XXXX.npz
    python acquisition/make_dataset.py --input rec1.npz rec2.npz rec3.npz   # merge multiple runs
    python acquisition/make_dataset.py --input recording.npz --subject 2 --test-size 0.3
"""

import argparse
import os
import sys
import numpy as np
import scipy.io

# Add project root to path for utils import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import eeg_filter

from sklearn.model_selection import train_test_split


SRATE = 250
EPOCH_SAMPLES = 4 * SRATE  # 4 seconds = 1000 samples
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'mymat_custom')


def load_recording(path):
    """Load .npz recording file."""
    data = np.load(path)
    return {
        'eeg_data': data['eeg_data'],           # (n_samples, 8)
        'eeg_timestamps': data['eeg_timestamps'],
        'markers': data['markers'],
        'marker_timestamps': data['marker_timestamps'],
        'srate': float(data['srate']),
    }


def extract_epochs(rec):
    """Extract 4s epochs aligned to marker onsets.

    Returns:
        epochs: (n_trials, 8, 1000) float32
        labels: (n_trials,) int — 1=left, 2=right
    """
    eeg = rec['eeg_data']          # (n_samples, 8)
    eeg_ts = rec['eeg_timestamps']
    markers = rec['markers']
    marker_ts = rec['marker_timestamps']

    if len(markers) == 0:
        raise ValueError('No markers found in recording')

    epochs = []
    labels = []
    n_dropped = 0

    for m, mt in zip(markers, marker_ts):
        # Find nearest EEG sample to marker timestamp
        idx = np.searchsorted(eeg_ts, mt)
        if idx + EPOCH_SAMPLES > len(eeg):
            n_dropped += 1
            continue

        epoch = eeg[idx:idx + EPOCH_SAMPLES, :].T  # (8, 1000)
        epochs.append(epoch)
        labels.append(int(m))

    if n_dropped > 0:
        print(f'Dropped {n_dropped} trials (insufficient data at end)')

    epochs = np.array(epochs, dtype=np.float32)   # (n_trials, 8, 1000)
    labels = np.array(labels, dtype=np.int32)      # (n_trials,)
    return epochs, labels


def main():
    parser = argparse.ArgumentParser(description='Convert recording to .mat training files')
    parser.add_argument('--input', required=True, nargs='+', help='Path to .npz recording file(s), multiple files will be merged')
    parser.add_argument('--subject', type=int, default=1, help='Subject number (default: 1)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio (default: 0.2)')
    args = parser.parse_args()

    # Load and extract epochs from all input files
    all_epochs = []
    all_labels = []
    srate = None

    for path in args.input:
        print(f'Loading: {path}')
        rec = load_recording(path)
        print(f'  EEG: {rec["eeg_data"].shape[0]} samples @ {rec["srate"]} Hz')
        print(f'  Markers: {len(rec["markers"])}')
        srate = int(rec['srate'])

        epochs, labels = extract_epochs(rec)
        print(f'  Extracted: {len(epochs)} epochs, shape {epochs.shape}')
        all_epochs.append(epochs)
        all_labels.append(labels)

    epochs = np.concatenate(all_epochs)
    labels = np.concatenate(all_labels)

    if len(args.input) > 1:
        print(f'\nMerged: {len(epochs)} total epochs from {len(args.input)} files')

    # Apply EEG filtering (bandpass 4-40Hz + notch 50Hz)
    print('  Filtering: bandpass 4-40Hz + notch 50Hz')
    epochs = eeg_filter(epochs, fs=srate)

    # Summary
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        name = 'left' if u == 1 else 'right'
        print(f'  Class {u} ({name}): {c} trials')

    # Train/test split
    train_data, test_data, train_labels, test_labels = train_test_split(
        epochs, labels, test_size=args.test_size, stratify=labels, random_state=42)

    # Reshape labels to (n, 1) to match BCI IV convention
    train_labels = train_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, f'C{args.subject:02d}T.mat')
    test_path = os.path.join(OUTPUT_DIR, f'C{args.subject:02d}E.mat')

    scipy.io.savemat(train_path, {'data': train_data, 'label': train_labels})
    scipy.io.savemat(test_path, {'data': test_data, 'label': test_labels})

    print(f'\nSaved:')
    print(f'  Train: {train_path} — data {train_data.shape}, label {train_labels.shape}')
    print(f'  Test:  {test_path} — data {test_data.shape}, label {test_labels.shape}')


if __name__ == '__main__':
    main()
