"""
LSL Multi-Stream Recorder

Records EEG data (from OpenBCI GUI) and markers (from paradigm.py) via LSL.
Saves to .npz for offline processing.

Usage:
    python acquisition/recorder.py

    Press Ctrl+C to stop recording and save.
"""

import time
import datetime
import os
import numpy as np
from pylsl import resolve_byprop, StreamInlet


RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), 'recordings')


def resolve_eeg_stream(name='obci_eeg1', timeout=30):
    """Resolve EEG stream by name."""
    print(f'Resolving EEG stream "{name}"...')
    streams = resolve_byprop('name', name, minimum=1, timeout=timeout)
    if not streams:
        raise RuntimeError(f'No EEG stream "{name}" found within {timeout}s')
    inlet = StreamInlet(streams[0], max_chunklen=64)
    info = inlet.info()
    srate = info.nominal_srate()
    n_ch = info.channel_count()
    print(f'EEG stream: {n_ch} channels @ {srate} Hz')
    if n_ch != 8:
        print(f'WARNING: Expected 8 channels, got {n_ch}')
    return inlet, srate, n_ch


def resolve_marker_stream(name='MI-Markers', timeout=5):
    """Resolve marker stream. Returns None if not found (starts recording anyway)."""
    print(f'Resolving marker stream "{name}"...')
    streams = resolve_byprop('name', name, minimum=1, timeout=timeout)
    if not streams:
        print(f'Marker stream "{name}" not found — recording EEG only (start paradigm.py to send markers)')
        return None
    inlet = StreamInlet(streams[0])
    print('Marker stream connected')
    return inlet


def record():
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    eeg_inlet, srate, n_ch = resolve_eeg_stream()
    marker_inlet = resolve_marker_stream()

    eeg_data = []
    eeg_timestamps = []
    markers = []
    marker_timestamps = []

    print('\nRecording... Press Ctrl+C to stop.\n')

    try:
        while True:
            # Pull EEG chunks (non-blocking with short timeout for throughput)
            chunk, ts = eeg_inlet.pull_chunk(timeout=0.05)
            if chunk:
                eeg_data.extend(chunk)
                eeg_timestamps.extend(ts)

            # Pull markers (non-blocking)
            if marker_inlet is not None:
                sample, ts_m = marker_inlet.pull_sample(timeout=0.0)
                if sample is not None:
                    markers.append(int(sample[0]))
                    marker_timestamps.append(ts_m)
                    label = 'LEFT' if int(sample[0]) == 1 else 'RIGHT'
                    print(f'  Marker: {label} (t={ts_m:.3f})')

    except KeyboardInterrupt:
        pass
    finally:
        eeg_data = np.array(eeg_data, dtype=np.float64)
        eeg_timestamps = np.array(eeg_timestamps, dtype=np.float64)
        markers = np.array(markers, dtype=np.int32)
        marker_timestamps = np.array(marker_timestamps, dtype=np.float64)

        if len(eeg_data) == 0:
            print('No EEG data recorded. Not saving.')
            return

        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(RECORDINGS_DIR, f'recording_{stamp}.npz')

        np.savez(filename,
                 eeg_data=eeg_data,
                 eeg_timestamps=eeg_timestamps,
                 markers=markers,
                 marker_timestamps=marker_timestamps,
                 srate=np.float64(srate))

        duration = eeg_timestamps[-1] - eeg_timestamps[0] if len(eeg_timestamps) > 1 else 0
        print(f'\nSaved: {filename}')
        print(f'  EEG samples: {len(eeg_data)} ({duration:.1f}s)')
        print(f'  Markers: {len(markers)}')


if __name__ == '__main__':
    record()
