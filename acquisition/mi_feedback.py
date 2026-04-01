"""
MI-BCI Real-time Feedback Demo

Displays a block that moves left/right based on motor imagery classification.
Uses a trained ECTNet model for real-time inference from LSL EEG stream.

Usage:
    python acquisition/mi_feedback.py
    python acquisition/mi_feedback.py --model C_heads_2_depth_6/model_1.pth
"""

import argparse
import sys
import os
import time
import numpy as np
import scipy.signal
from collections import deque

import pygame
from pylsl import resolve_byprop, StreamInlet


# ── EEG Filter (same as training pipeline) ──────────────
def eeg_filter(data, fs=250, bandpass=(4, 40), notch=50):
    filtered = data.astype(np.float64)
    if bandpass is not None:
        b, a = scipy.signal.butter(4, bandpass, btype='band', fs=fs)
        filtered = scipy.signal.filtfilt(b, a, filtered, axis=-1)
    if notch is not None and notch < fs / 2:
        b_n, a_n = scipy.signal.iirnotch(notch, Q=30, fs=fs)
        filtered = scipy.signal.filtfilt(b_n, a_n, filtered, axis=-1)
    return filtered.astype(np.float32)


# ── Model Loading ───────────────────────────────────────
def load_model(model_path, device='cpu'):
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from model import EEGTransformer

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    saved_model = ckpt['model']
    norm_mean = ckpt.get('norm_mean')
    norm_std = ckpt.get('norm_std')

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


# ── Pygame Feedback UI ──────────────────────────────────
WIDTH, HEIGHT = 800, 400
BG_COLOR = (30, 30, 30)
BLOCK_SIZE = 60
CENTER_X = WIDTH // 2
CENTER_Y = HEIGHT // 2
MOVE_SPEED = 8
COLORS = {
    'left': (52, 152, 219),    # blue
    'right': (231, 76, 60),    # red
    'neutral': (149, 165, 166), # gray
    'text': (236, 240, 241),
    'bar_bg': (60, 60, 60),
}


def main():
    parser = argparse.ArgumentParser(description='MI-BCI Feedback Demo')
    parser.add_argument('--model', default='C_heads_2_depth_6/model_1.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--channels', type=str, default='0,1,2',
                        help='Channel indices (default: 0,1,2 for C3,C4,Cz)')
    parser.add_argument('--stream', default='obci_eeg1',
                        help='LSL stream name')
    parser.add_argument('--window', type=float, default=4.0,
                        help='Classification window in seconds')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Classification interval in seconds')
    args = parser.parse_args()

    import torch

    ch_idx = [int(c) for c in args.channels.split(',')]
    n_channels = len(ch_idx)
    srate = 250
    window_samples = int(args.window * srate)

    # Load model
    print(f'Loading model: {args.model}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, norm_mean, norm_std = load_model(args.model, device)
    print(f'Model loaded on {device}')

    # Connect to LSL
    print(f'Resolving EEG stream "{args.stream}"...')
    streams = resolve_byprop('name', args.stream, minimum=1, timeout=30)
    if not streams:
        print(f'ERROR: No stream "{args.stream}" found')
        return
    inlet = StreamInlet(streams[0], max_chunklen=64)
    info = inlet.info()
    print(f'Connected: {info.channel_count()}ch @ {info.nominal_srate()}Hz')

    # Ring buffer
    buffer = deque(maxlen=window_samples + srate)

    # Init pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('MI-BCI Feedback')
    clock = pygame.time.Clock()
    font_large = pygame.font.Font(None, 48)
    font_small = pygame.font.Font(None, 28)

    block_x = CENTER_X
    target_x = CENTER_X
    last_classify_time = 0
    current_label = 'WAIT'
    confidence = 0.0
    left_prob = 0.5
    right_prob = 0.5
    classify_count = 0

    print(f'\nBuffering {args.window}s of data...')
    print('Press ESC or close window to quit.\n')

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Pull EEG data
        chunk, timestamps = inlet.pull_chunk(timeout=0.0)
        if chunk:
            for sample in chunk:
                selected = [sample[i] for i in ch_idx]
                buffer.append(selected)

        # Classify
        now = time.time()
        if len(buffer) >= window_samples and (now - last_classify_time) >= args.interval:
            last_classify_time = now

            window = np.array(list(buffer))[-window_samples:]
            window = window.T  # (3, 1000)
            window = eeg_filter(window, fs=srate)

            if norm_mean is not None and norm_std is not None:
                window = (window - norm_mean.reshape(-1, 1)) / norm_std.reshape(-1, 1)

            x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(x)
                logits = output[1] if isinstance(output, tuple) else output
                probs = logits.softmax(dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))

            left_prob = probs[0]
            right_prob = probs[1]
            confidence = probs[pred] * 100
            current_label = 'LEFT' if pred == 0 else 'RIGHT'
            classify_count += 1

            # Set target position
            if pred == 0:  # LEFT
                target_x = CENTER_X - int((left_prob - 0.5) * 2 * (WIDTH // 2 - BLOCK_SIZE))
            else:  # RIGHT
                target_x = CENTER_X + int((right_prob - 0.5) * 2 * (WIDTH // 2 - BLOCK_SIZE))

        # Smooth block movement
        diff = target_x - block_x
        block_x += diff * 0.15

        # ── Draw ────────────────────────────────────
        screen.fill(BG_COLOR)

        # Center line
        pygame.draw.line(screen, (80, 80, 80), (CENTER_X, 50), (CENTER_X, HEIGHT - 50), 1)

        # Left/Right labels
        left_text = font_small.render('LEFT', True, COLORS['left'])
        right_text = font_small.render('RIGHT', True, COLORS['right'])
        screen.blit(left_text, (30, CENTER_Y - 10))
        screen.blit(right_text, (WIDTH - 100, CENTER_Y - 10))

        # Confidence bar
        bar_y = HEIGHT - 60
        bar_width = WIDTH - 100
        bar_x = 50
        pygame.draw.rect(screen, COLORS['bar_bg'], (bar_x, bar_y, bar_width, 20), border_radius=10)
        # Left portion (blue)
        left_width = int(left_prob * bar_width)
        pygame.draw.rect(screen, COLORS['left'], (bar_x, bar_y, left_width, 20), border_radius=10)
        # Divider
        div_x = bar_x + left_width
        pygame.draw.line(screen, COLORS['text'], (div_x, bar_y - 3), (div_x, bar_y + 23), 2)
        # Labels
        l_pct = font_small.render(f'L:{left_prob:.0%}', True, COLORS['left'])
        r_pct = font_small.render(f'R:{right_prob:.0%}', True, COLORS['right'])
        screen.blit(l_pct, (bar_x, bar_y - 25))
        screen.blit(r_pct, (bar_x + bar_width - 60, bar_y - 25))

        # Block
        color = COLORS['left'] if current_label == 'LEFT' else COLORS['right'] if current_label == 'RIGHT' else COLORS['neutral']
        block_rect = pygame.Rect(int(block_x) - BLOCK_SIZE // 2, CENTER_Y - BLOCK_SIZE // 2,
                                 BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(screen, color, block_rect, border_radius=8)

        # Classification label
        if classify_count > 0:
            label_text = font_large.render(f'{current_label} {confidence:.0f}%', True, COLORS['text'])
            screen.blit(label_text, (CENTER_X - label_text.get_width() // 2, 20))
        else:
            wait_text = font_large.render('Buffering...', True, COLORS['neutral'])
            screen.blit(wait_text, (CENTER_X - wait_text.get_width() // 2, 20))

        # Count
        count_text = font_small.render(f'#{classify_count}', True, (100, 100, 100))
        screen.blit(count_text, (WIDTH - 60, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print(f'Total classifications: {classify_count}')


if __name__ == '__main__':
    main()
