"""
Real-time EEG Signal Viewer

Shows live waveforms from LSL stream to verify electrode signal quality.
Bandpass 4-40 Hz + notch 50 Hz (same as training pipeline).

Usage:
    python acquisition/check_signal.py

Controls:
    Up/Down arrow: zoom in/out Y-axis

Signal quality (RMS):
    GREEN:  5-50 µV  (good EEG)
    YELLOW: 50-100 µV (noisy but usable)
    RED:    >100 or <5 µV (bad contact)
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylsl import resolve_byprop, StreamInlet

SRATE = 250
WINDOW_SEC = 4
WINDOW_SAMPLES = SRATE * WINDOW_SEC
CHANNEL_NAMES = ['C3', 'C4', 'Cz']
N_CH = len(CHANNEL_NAMES)
Y_SCALES = [25, 50, 100, 200, 500]
y_idx = 2  # default ±100

# Pre-compute filter coefficients (do once, not every frame)
b_bp, a_bp = scipy.signal.butter(4, [4, 40], btype='band', fs=SRATE)
b_n, a_n = scipy.signal.iirnotch(50, Q=30, fs=SRATE)

# Resolve LSL stream
print('Resolving EEG stream...')
streams = resolve_byprop('name', 'obci_eeg1', minimum=1, timeout=15)
if not streams:
    print('ERROR: No stream found')
    exit(1)
inlet = StreamInlet(streams[0], max_chunklen=64)
print(f'Connected: {inlet.info().channel_count()}ch @ {inlet.info().nominal_srate()}Hz')
print('Controls: Up/Down arrow to zoom Y-axis\n')

# Ring buffer
buf = np.zeros((N_CH, WINDOW_SAMPLES))

# Setup plot
fig, axes = plt.subplots(N_CH, 1, figsize=(10, 6), sharex=True)
fig.suptitle('EEG Signal Check')
lines = []
quality_texts = []
for i, ax in enumerate(axes):
    line, = ax.plot(np.arange(WINDOW_SAMPLES) / SRATE, buf[i], linewidth=0.8)
    lines.append(line)
    ax.set_ylabel(f'{CHANNEL_NAMES[i]} (µV)')
    ax.set_ylim(-Y_SCALES[y_idx], Y_SCALES[y_idx])
    ax.grid(True, alpha=0.3)
    # RMS quality text in top-right corner
    txt = ax.text(0.98, 0.92, '', transform=ax.transAxes,
                  ha='right', va='top', fontsize=10, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    quality_texts.append(txt)
axes[-1].set_xlabel('Time (s)')
scale_text = fig.text(0.02, 0.98, f'Scale: ±{Y_SCALES[y_idx]} µV',
                      va='top', fontsize=9, color='gray')
plt.tight_layout(rect=[0, 0, 1, 0.96])


def on_key(event):
    global y_idx
    if event.key == 'up' and y_idx > 0:
        y_idx -= 1
    elif event.key == 'down' and y_idx < len(Y_SCALES) - 1:
        y_idx += 1
    else:
        return
    for ax in axes:
        ax.set_ylim(-Y_SCALES[y_idx], Y_SCALES[y_idx])
    scale_text.set_text(f'Scale: ±{Y_SCALES[y_idx]} µV')

fig.canvas.mpl_connect('key_press_event', on_key)


def rms_quality(rms):
    """Return (label, color) based on RMS value."""
    if rms < 5:
        return 'BAD', '#E74C3C'
    elif rms <= 50:
        return 'GOOD', '#2ECC71'
    elif rms <= 100:
        return 'OK', '#F39C12'
    else:
        return 'BAD', '#E74C3C'


def update(frame):
    # Pull all available samples
    samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=128)
    if samples:
        chunk = np.array(samples)[:, :N_CH].T
        n = min(chunk.shape[1], WINDOW_SAMPLES)
        buf[:, :-n] = buf[:, n:]
        buf[:, -n:] = chunk[:, -n:]

    # Bandpass 4-40 Hz + notch 50 Hz
    filtered = scipy.signal.filtfilt(b_bp, a_bp, buf, axis=-1)
    filtered = scipy.signal.filtfilt(b_n, a_n, filtered, axis=-1)

    for i, line in enumerate(lines):
        signal = filtered[i]
        line.set_ydata(signal)

        # RMS over last 1 second
        rms = np.sqrt(np.mean(signal[-SRATE:]**2))
        label, color = rms_quality(rms)
        quality_texts[i].set_text(f'{rms:.1f} µV [{label}]')
        quality_texts[i].get_bbox_patch().set_facecolor(color)
        quality_texts[i].set_color('white')

    return lines


ani = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
plt.show()
