"""
Real-time MI-BCI Inference

Receives EEG from OpenBCI via LSL, classifies left/right hand motor imagery
in real-time using a trained ECTNet model.

Requirements:
    - OpenBCI GUI streaming LSL (TimeSeriesRaw, name: obci_eeg1)
    - Trained model checkpoint (.pth)
    - conda activate acquisition + pip install torch (cpu only is fine)

Usage:
    python acquisition/realtime_inference.py
    python acquisition/realtime_inference.py --model path/to/model.pth --channels 0,1,2
"""

import argparse
import time
import sys
import os
import numpy as np
import scipy.signal
from collections import deque

# ── Filtering (copied from utils.py to avoid heavy imports) ──────────────

def eeg_filter(data, fs=250, bandpass=(4, 40), notch=50):
    """Apply bandpass and notch filter to EEG data.
    Args:
        data: (..., n_samples) — filtering applied on last axis
        fs: sampling rate in Hz
    """
    filtered = data.astype(np.float64)
    if bandpass is not None:
        b, a = scipy.signal.butter(4, bandpass, btype='band', fs=fs)
        filtered = scipy.signal.filtfilt(b, a, filtered, axis=-1)
    if notch is not None and notch < fs / 2:
        b_n, a_n = scipy.signal.iirnotch(notch, Q=30, fs=fs)
        filtered = scipy.signal.filtfilt(b_n, a_n, filtered, axis=-1)
    return filtered.astype(np.float32)


# ── Model loading ────────────────────────────────────────────────────────

def load_model(model_path, device='cpu'):
    """Load trained ECTNet model from checkpoint."""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from model import EEGTransformer

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    saved_model = ckpt['model']
    norm_mean = ckpt.get('norm_mean')
    norm_std = ckpt.get('norm_std')

    # Handle both state_dict and full model object
    if isinstance(saved_model, dict):
        model = EEGTransformer(
            heads=2, emb_size=16, depth=6,
            database_type='C',
            eeg1_f1=8, eeg1_D=2, eeg1_kernel_size=64,
            eeg1_pooling_size1=8, eeg1_pooling_size2=8,
            eeg1_dropout_rate=0.5,
            eeg1_number_channel=3,
            flatten_eeg1=240,
        )
        model.load_state_dict(saved_model)
    else:
        model = saved_model
    model.to(device)
    model.eval()
    return model, norm_mean, norm_std


# ── Main loop ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Real-time MI-BCI inference')
    parser.add_argument('--model', default='C_heads_2_depth_6/model_1.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--channels', type=str, default='0,1,2',
                        help='Channel indices to use (default: 0,1,2 for C3,Cz,C4)')
    parser.add_argument('--stream', default='obci_eeg1',
                        help='LSL stream name')
    parser.add_argument('--window', type=float, default=4.0,
                        help='Classification window in seconds (default: 4.0)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Classification interval in seconds (default: 1.0)')
    args = parser.parse_args()

    import torch
    from pylsl import resolve_byprop, StreamInlet

    ch_idx = [int(c) for c in args.channels.split(',')]
    n_channels = len(ch_idx)
    srate = 250
    window_samples = int(args.window * srate)  # 1000 samples for 4s

    # Load model
    print(f'Loading model: {args.model}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, norm_mean, norm_std = load_model(args.model, device)
    print(f'Model loaded on {device}, channels: {ch_idx}')

    # Connect to LSL
    print(f'Resolving EEG stream "{args.stream}"...')
    streams = resolve_byprop('name', args.stream, minimum=1, timeout=30)
    if not streams:
        print(f'ERROR: No stream "{args.stream}" found')
        return
    inlet = StreamInlet(streams[0], max_chunklen=64)
    info = inlet.info()
    print(f'Connected: {info.channel_count()} channels @ {info.nominal_srate()} Hz')

    # Ring buffer for EEG data
    buffer = deque(maxlen=window_samples + srate)  # extra 1s padding for filter edge effects

    print(f'\n{"="*50}')
    print(f'  Real-time MI Classification')
    print(f'  Window: {args.window}s | Interval: {args.interval}s')
    print(f'  Channels: {ch_idx} (C3, Cz, C4)')
    print(f'{"="*50}')
    print(f'Buffering {args.window}s of data...\n')

    last_classify_time = 0
    classify_count = 0

    try:
        while True:
            # Pull EEG data
            chunk, timestamps = inlet.pull_chunk(timeout=0.05)
            if chunk:
                for sample in chunk:
                    # Select only the channels we need
                    selected = [sample[i] for i in ch_idx]
                    buffer.append(selected)

            # Classify when we have enough data
            now = time.time()
            if len(buffer) >= window_samples and (now - last_classify_time) >= args.interval:
                last_classify_time = now

                # Extract window: (window_samples, n_channels) → (n_channels, window_samples)
                window = np.array(list(buffer))[-window_samples:]  # (1000, 3)
                window = window.T  # (3, 1000)

                # Filter
                window = eeg_filter(window, fs=srate)

                # Normalize (if checkpoint has norm stats)
                if norm_mean is not None and norm_std is not None:
                    window = (window - norm_mean.reshape(-1, 1)) / norm_std.reshape(-1, 1)

                # Model input: (1, 1, 3, 1000)
                x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                    pred = int(np.argmax(probs))

                label = 'LEFT ' if pred == 0 else 'RIGHT'
                confidence = probs[pred] * 100
                bar_len = int(confidence / 2)

                classify_count += 1
                print(f'  [{classify_count:4d}]  {label}  {confidence:5.1f}%  '
                      f'|{"█" * bar_len}{"░" * (50 - bar_len)}|  '
                      f'L:{probs[0]:.2f}  R:{probs[1]:.2f}')

    except KeyboardInterrupt:
        print(f'\n\nStopped. Total classifications: {classify_count}')


if __name__ == '__main__':
    main()
