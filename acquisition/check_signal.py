"""
Real-time EEG Signal Viewer + FFT

Left: time-domain waveforms (bandpass 4-40 Hz + notch 50 Hz)
Right: frequency spectrum (0-50 Hz) with band power highlights

Usage:
    python acquisition/check_signal.py

Controls:
    Up/Down arrow: zoom in/out Y-axis

Signal quality (RMS):
    GREEN:  5-50 µV  (good EEG)
    YELLOW: 50-100 µV (noisy but usable)
    RED:    >100 or <5 µV (bad contact)

Frequency bands:
    theta (4-8 Hz), alpha/mu (8-13 Hz), beta (13-30 Hz)
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
CHANNEL_COLORS = ['#3498DB', '#E74C3C', '#2ECC71']
N_CH = len(CHANNEL_NAMES)
Y_SCALES = [25, 50, 100, 200, 500]
y_idx = 2  # default ±100

# Frequency bands
BANDS = {
    'theta': (4, 8, '#9B59B6'),
    'mu/alpha': (8, 13, '#E67E22'),
    'beta': (13, 30, '#27AE60'),
}

# Pre-compute filter coefficients
b_bp, a_bp = scipy.signal.butter(4, [4, 40], btype='band', fs=SRATE)
b_n, a_n = scipy.signal.iirnotch(50, Q=30, fs=SRATE)

# FFT frequencies
fft_n = WINDOW_SAMPLES
freqs = np.fft.rfftfreq(fft_n, 1.0 / SRATE)
freq_mask = freqs <= 50  # show 0-50 Hz

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

# Setup plot: left = waveforms, right = FFT
fig, all_axes = plt.subplots(N_CH, 2, figsize=(14, 7),
                              gridspec_kw={'width_ratios': [2, 1]})
fig.suptitle('EEG Signal Check + FFT')

wave_lines = []
fft_lines = []
quality_texts = []
band_spans = []

for i in range(N_CH):
    # Left: waveform
    ax_wave = all_axes[i, 0]
    line, = ax_wave.plot(np.arange(WINDOW_SAMPLES) / SRATE, buf[i],
                         linewidth=0.8, color=CHANNEL_COLORS[i])
    wave_lines.append(line)
    ax_wave.set_ylabel(f'{CHANNEL_NAMES[i]} (µV)')
    ax_wave.set_ylim(-Y_SCALES[y_idx], Y_SCALES[y_idx])
    ax_wave.grid(True, alpha=0.3)
    txt = ax_wave.text(0.98, 0.92, '', transform=ax_wave.transAxes,
                       ha='right', va='top', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    quality_texts.append(txt)

    # Right: FFT
    ax_fft = all_axes[i, 1]
    fft_line, = ax_fft.plot(freqs[freq_mask], np.zeros(freq_mask.sum()),
                            linewidth=0.8, color=CHANNEL_COLORS[i])
    fft_lines.append(fft_line)
    ax_fft.set_ylabel('Power (µV²/Hz)')
    ax_fft.set_xlim(0, 50)
    ax_fft.set_ylim(0, 50)
    ax_fft.grid(True, alpha=0.3)

    # Band highlights
    ch_spans = {}
    for band_name, (f_lo, f_hi, color) in BANDS.items():
        span = ax_fft.axvspan(f_lo, f_hi, alpha=0.15, color=color)
        ch_spans[band_name] = span
    band_spans.append(ch_spans)

    # Band labels (only on first channel)
    if i == 0:
        for band_name, (f_lo, f_hi, color) in BANDS.items():
            ax_fft.text((f_lo + f_hi) / 2, 0.95, band_name, transform=ax_fft.get_xaxis_transform(),
                        ha='center', va='top', fontsize=7, color=color, fontweight='bold')

all_axes[-1, 0].set_xlabel('Time (s)')
all_axes[-1, 1].set_xlabel('Frequency (Hz)')

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
    for i in range(N_CH):
        all_axes[i, 0].set_ylim(-Y_SCALES[y_idx], Y_SCALES[y_idx])
    scale_text.set_text(f'Scale: ±{Y_SCALES[y_idx]} µV')

fig.canvas.mpl_connect('key_press_event', on_key)


def rms_quality(rms):
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

    # Bandpass filter for waveform display
    filtered = scipy.signal.filtfilt(b_bp, a_bp, buf, axis=-1)
    filtered = scipy.signal.filtfilt(b_n, a_n, filtered, axis=-1)

    # FFT on filtered signal
    window = np.hanning(fft_n)
    fft_all = []
    for i in range(N_CH):
        spectrum = np.abs(np.fft.rfft(filtered[i] * window)) ** 2
        spectrum = spectrum / fft_n  # normalize
        fft_all.append(spectrum)

    # Auto-scale FFT y-axis
    fft_max = max(np.max(fft_all[i][freq_mask]) for i in range(N_CH))
    fft_ylim = max(fft_max * 1.3, 1)

    for i in range(N_CH):
        # Waveform
        signal = filtered[i]
        wave_lines[i].set_ydata(signal)

        # RMS
        rms = np.sqrt(np.mean(signal[-SRATE:] ** 2))
        label, color = rms_quality(rms)
        quality_texts[i].set_text(f'{rms:.1f} µV [{label}]')
        quality_texts[i].get_bbox_patch().set_facecolor(color)
        quality_texts[i].set_color('white')

        # FFT
        fft_lines[i].set_ydata(fft_all[i][freq_mask])
        all_axes[i, 1].set_ylim(0, fft_ylim)

    return wave_lines + fft_lines


ani = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
plt.show()
