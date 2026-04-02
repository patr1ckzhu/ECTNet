"""
MI-BCI Real-time Feedback Demo (Trial-based)

Trial-based cursor control: a block starts at center, Roger uses motor imagery
to push it toward a target zone (left or right). Visual feedback in real-time.

Trial structure:
  [Fixation]  Block at center, crosshair (1.5s)
  [Cue]       Target zone appears with arrow (1s)
  [MI]        Block moves until it reaches target zone (no time limit)
  [Result]    Success animation (1.5s)
  [Rest]      Blank (2s)
  Press SPACE to skip a trial, ESC to quit.

Usage:
    python acquisition/mi_feedback.py
    python acquisition/mi_feedback.py --model transfer_C_freeze_none/model_1.pth --trials 20
"""

import argparse
import sys
import os
import time
import random
import numpy as np
import scipy.signal
from collections import deque

import pygame
from pylsl import resolve_byprop, StreamInlet


# ── EEG Filter ────────────────────────────────────────────
def eeg_filter(data, fs=250, bandpass=(4, 40), notch=50):
    filtered = data.astype(np.float64)
    if bandpass is not None:
        b, a = scipy.signal.butter(4, bandpass, btype='band', fs=fs)
        filtered = scipy.signal.filtfilt(b, a, filtered, axis=-1)
    if notch is not None and notch < fs / 2:
        b_n, a_n = scipy.signal.iirnotch(notch, Q=30, fs=fs)
        filtered = scipy.signal.filtfilt(b_n, a_n, filtered, axis=-1)
    return filtered.astype(np.float32)


# ── Model Loading ─────────────────────────────────────────
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


# ── Constants ─────────────────────────────────────────────
WIDTH, HEIGHT = 1000, 600
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
BLOCK_SIZE = 50
TARGET_ZONE_WIDTH = 120

# Timing (seconds)
T_FIXATION = 1.5
T_CUE = 1.0
T_RESULT = 1.5
T_REST = 2.0

# Movement
MOVE_SCALE = 12.0       # pixels per classification at prob=1.0
SMOOTHING = 0.25        # block position smoothing factor (higher = snappier)
TARGET_HIT_X = 0.65     # block must reach 65% of way to edge to succeed

# Colors
C_BG = (20, 20, 30)
C_TEXT = (220, 225, 230)
C_DIM = (80, 85, 90)
C_CROSS = (150, 150, 160)
C_LEFT = (52, 152, 219)
C_RIGHT = (231, 76, 60)
C_SUCCESS = (46, 204, 113)
C_FAIL = (192, 57, 43)
C_TARGET_ZONE = (50, 55, 65)
C_BAR_BG = (45, 48, 55)


# ── Trial phases ──────────────────────────────────────────
PHASE_FIXATION = 'fixation'
PHASE_CUE = 'cue'
PHASE_MI = 'mi'
PHASE_RESULT = 'result'
PHASE_REST = 'rest'
PHASE_DONE = 'done'


def main():
    parser = argparse.ArgumentParser(description='MI-BCI Feedback Demo')
    parser.add_argument('--model', default='transfer_C_freeze_none/model_1.pth')
    parser.add_argument('--channels', type=str, default='0,1,2')
    parser.add_argument('--stream', default='obci_eeg1')
    parser.add_argument('--window', type=float, default=4.0)
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()

    import torch

    ch_idx = [int(c) for c in args.channels.split(',')]
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

    buffer = deque(maxlen=window_samples + srate)

    # Init pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('MI-BCI Feedback')
    clock = pygame.time.Clock()
    # Use Chinese-capable font (Microsoft YaHei on Windows, fallback to default)
    _cn_fonts = ['microsoftyahei', 'msyh', 'simhei', 'simsun', 'arial']
    _font_name = pygame.font.match_font(_cn_fonts[0]) or pygame.font.match_font(_cn_fonts[1]) \
                 or pygame.font.match_font(_cn_fonts[2])
    if _font_name:
        font_huge = pygame.font.Font(_font_name, 64)
        font_large = pygame.font.Font(_font_name, 42)
        font_med = pygame.font.Font(_font_name, 28)
        font_small = pygame.font.Font(_font_name, 20)
    else:
        font_huge = pygame.font.Font(None, 72)
        font_large = pygame.font.Font(None, 48)
        font_med = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 24)

    # Generate trial sequence (balanced left/right, shuffled)
    n_trials = args.trials
    targets = [0] * (n_trials // 2) + [1] * (n_trials - n_trials // 2)  # 0=left, 1=right
    random.shuffle(targets)

    # State
    trial_idx = 0
    phase = PHASE_FIXATION
    phase_start = time.time()
    block_x = float(CENTER_X)
    target_x = float(CENTER_X)
    last_classify_time = 0
    mi_start_time = 0
    left_prob, right_prob = 0.5, 0.5
    successes = 0
    streak = 0
    max_streak = 0
    trial_times = []       # seconds per successful trial
    results_history = []   # list of (target, success, mi_acc)
    skipped = False
    trial_classify_total = 0    # classifications in current trial
    trial_classify_correct = 0  # correct direction in current trial

    print(f'\n  {n_trials} trials, SPACE=skip, ESC=quit.\n')
    print(f'  Buffering {args.window}s of data...')

    running = True
    while running:
        now = time.time()
        elapsed = now - phase_start

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and phase == PHASE_MI:
                    skipped = True

        # ── Pull EEG continuously ──
        chunk, _ = inlet.pull_chunk(timeout=0.0)
        if chunk:
            for sample in chunk:
                buffer.append([sample[i] for i in ch_idx])

        # ── Phase transitions ──
        if phase == PHASE_FIXATION:
            block_x = float(CENTER_X)
            target_x = float(CENTER_X)
            left_prob, right_prob = 0.5, 0.5
            skipped = False
            trial_classify_total = 0
            trial_classify_correct = 0
            if elapsed >= T_FIXATION:
                phase = PHASE_CUE
                phase_start = now

        elif phase == PHASE_CUE:
            if elapsed >= T_CUE:
                phase = PHASE_MI
                phase_start = now
                mi_start_time = now
                last_classify_time = 0

        elif phase == PHASE_MI:
            cur_target = targets[trial_idx]

            # Classify periodically
            if len(buffer) >= window_samples and (now - last_classify_time) >= args.interval:
                last_classify_time = now
                window = np.array(list(buffer))[-window_samples:].T
                window = eeg_filter(window, fs=srate)
                if norm_mean is not None and norm_std is not None:
                    window = (window - norm_mean.reshape(-1, 1)) / norm_std.reshape(-1, 1)
                x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(x)
                    logits = output[1] if isinstance(output, tuple) else output
                    probs = logits.softmax(dim=1).cpu().numpy()[0]
                left_prob, right_prob = probs[0], probs[1]

                # Track per-trial MI accuracy
                trial_classify_total += 1
                pred_dir = 0 if left_prob > right_prob else 1
                if pred_dir == cur_target:
                    trial_classify_correct += 1

                # Push block based on probability
                force = (right_prob - left_prob) * MOVE_SCALE
                target_x = max(BLOCK_SIZE, min(WIDTH - BLOCK_SIZE, target_x + force))

            # Smooth movement
            block_x += (target_x - block_x) * SMOOTHING

            # Check if block reached target zone
            hit_left = block_x < CENTER_X * (1 - TARGET_HIT_X)
            hit_right = block_x > CENTER_X * (1 + TARGET_HIT_X)
            success = (cur_target == 0 and hit_left) or (cur_target == 1 and hit_right)

            if success or skipped:
                mi_duration = now - mi_start_time
                mi_acc = trial_classify_correct / trial_classify_total if trial_classify_total > 0 else 0
                results_history.append((cur_target, success, mi_acc, mi_duration, trial_classify_correct, trial_classify_total))
                if success:
                    successes += 1
                    streak += 1
                    max_streak = max(max_streak, streak)
                    trial_times.append(mi_duration)
                else:
                    streak = 0
                phase = PHASE_RESULT
                phase_start = now

        elif phase == PHASE_RESULT:
            if elapsed >= T_RESULT:
                phase = PHASE_REST
                phase_start = now

        elif phase == PHASE_REST:
            if elapsed >= T_REST:
                trial_idx += 1
                if trial_idx >= n_trials:
                    phase = PHASE_DONE
                else:
                    phase = PHASE_FIXATION
                    phase_start = now

        elif phase == PHASE_DONE:
            pass  # show final screen

        # ── Draw ──────────────────────────────────────
        screen.fill(C_BG)
        cur_target = targets[trial_idx] if trial_idx < n_trials else -1

        # Target zones (always visible as subtle background)
        left_zone = pygame.Rect(0, 80, TARGET_ZONE_WIDTH, HEIGHT - 160)
        right_zone = pygame.Rect(WIDTH - TARGET_ZONE_WIDTH, 80, TARGET_ZONE_WIDTH, HEIGHT - 160)

        if phase == PHASE_DONE:
            # Final results screen
            _draw_final_screen(screen, font_huge, font_large, font_med, font_small,
                               results_history, successes, n_trials, max_streak, trial_times)
            pygame.display.flip()
            clock.tick(60)
            continue

        # Draw target zone highlight during cue/mi/result
        if phase in (PHASE_CUE, PHASE_MI, PHASE_RESULT):
            zone = left_zone if cur_target == 0 else right_zone
            zone_color = C_TARGET_ZONE
            if phase == PHASE_RESULT:
                last_success = results_history[-1][1] if results_history else False
                # Use a surface for alpha
                s = pygame.Surface((zone.width, zone.height), pygame.SRCALPHA)
                s.fill((*C_SUCCESS, 40) if last_success else (*C_FAIL, 40))
                screen.blit(s, zone.topleft)
            else:
                pygame.draw.rect(screen, zone_color, zone, border_radius=8)

        # Center line
        pygame.draw.line(screen, (50, 52, 58), (CENTER_X, 80), (CENTER_X, HEIGHT - 80), 1)

        # Phase-specific elements
        if phase == PHASE_FIXATION:
            # Crosshair
            cross_len = 20
            pygame.draw.line(screen, C_CROSS, (CENTER_X - cross_len, CENTER_Y),
                             (CENTER_X + cross_len, CENTER_Y), 2)
            pygame.draw.line(screen, C_CROSS, (CENTER_X, CENTER_Y - cross_len),
                             (CENTER_X, CENTER_Y + cross_len), 2)
            _draw_phase_label(screen, font_med, "准备", C_DIM)

        elif phase == PHASE_CUE:
            # Arrow indicating target direction
            arrow_color = C_LEFT if cur_target == 0 else C_RIGHT
            arrow_text = "◀  左手想象" if cur_target == 0 else "右手想象  ▶"
            text_surf = font_large.render(arrow_text, True, arrow_color)
            screen.blit(text_surf, (CENTER_X - text_surf.get_width() // 2, 25))

        elif phase == PHASE_MI:
            mi_elapsed = now - mi_start_time

            # Elapsed time
            time_text = font_small.render(f'{mi_elapsed:.1f}s', True, C_DIM)
            screen.blit(time_text, (CENTER_X - time_text.get_width() // 2, 15))

            # Direction reminder (subtle)
            arrow_text = "◀ LEFT" if cur_target == 0 else "RIGHT ▶"
            arrow_color = C_LEFT if cur_target == 0 else C_RIGHT
            text_surf = font_small.render(arrow_text, True, (*arrow_color[:3],))
            if cur_target == 0:
                screen.blit(text_surf, (15, 15))
            else:
                screen.blit(text_surf, (WIDTH - text_surf.get_width() - 15, 15))

            # Skip hint
            skip_text = font_small.render("SPACE 跳过", True, (60, 60, 65))
            screen.blit(skip_text, (CENTER_X - skip_text.get_width() // 2, HEIGHT - 25))

            # Probability bar at bottom
            _draw_prob_bar(screen, font_small, left_prob, right_prob)

        elif phase == PHASE_RESULT:
            last_result = results_history[-1]
            last_success = last_result[1]
            last_mi_acc = last_result[2]
            last_duration = last_result[3]
            last_correct = last_result[4]
            last_total = last_result[5]

            result_text = "成功!" if last_success else "跳过"
            result_color = C_SUCCESS if last_success else C_FAIL
            text_surf = font_huge.render(result_text, True, result_color)
            screen.blit(text_surf, (CENTER_X - text_surf.get_width() // 2, 15))

            # Per-trial MI accuracy
            acc_text = font_med.render(f'MI正确率: {last_correct}/{last_total} ({last_mi_acc*100:.0f}%)', True, C_TEXT)
            screen.blit(acc_text, (CENTER_X - acc_text.get_width() // 2, HEIGHT - 80))
            time_text = font_small.render(f'用时 {last_duration:.1f}s', True, C_DIM)
            screen.blit(time_text, (CENTER_X - time_text.get_width() // 2, HEIGHT - 50))

        elif phase == PHASE_REST:
            _draw_phase_label(screen, font_med, "休息", C_DIM)

        # Block (draw in all phases except rest and done)
        if phase not in (PHASE_REST, PHASE_DONE):
            if phase == PHASE_RESULT:
                last_success = results_history[-1][1]
                block_color = C_SUCCESS if last_success else C_FAIL
            elif phase == PHASE_MI:
                # Color based on which direction the block is moving
                if block_x < CENTER_X - 10:
                    block_color = C_LEFT
                elif block_x > CENTER_X + 10:
                    block_color = C_RIGHT
                else:
                    block_color = C_DIM
            else:
                block_color = C_DIM

            bx = int(block_x) - BLOCK_SIZE // 2
            by = CENTER_Y - BLOCK_SIZE // 2
            pygame.draw.rect(screen, block_color, (bx, by, BLOCK_SIZE, BLOCK_SIZE), border_radius=10)

        # Stats panel (top-right)
        _draw_stats(screen, font_small, trial_idx, n_trials, successes, streak)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    # Print summary
    if results_history:
        completed = len(results_history)
        print(f'\n{"="*40}')
        print(f'  Results: {successes}/{completed} ({successes/completed*100:.0f}%)')
        print(f'  Max streak: {max_streak}')
        if trial_times:
            print(f'  Avg time to target: {np.mean(trial_times):.1f}s')
        left_trials = [r for r in results_history if r[0] == 0]
        right_trials = [r for r in results_history if r[0] == 1]
        if left_trials:
            left_succ = sum(r[1] for r in left_trials)
            print(f'  Left:  {left_succ}/{len(left_trials)} ({left_succ/len(left_trials)*100:.0f}%)')
        if right_trials:
            right_succ = sum(r[1] for r in right_trials)
            print(f'  Right: {right_succ}/{len(right_trials)} ({right_succ/len(right_trials)*100:.0f}%)')
        all_correct = sum(r[4] for r in results_history)
        all_total = sum(r[5] for r in results_history)
        if all_total > 0:
            print(f'  Overall MI acc: {all_correct}/{all_total} ({all_correct/all_total*100:.0f}%)')
        print(f'{"="*40}')


# ── Drawing helpers ───────────────────────────────────────

def _draw_phase_label(screen, font, text, color):
    surf = font.render(text, True, color)
    screen.blit(surf, (CENTER_X - surf.get_width() // 2, 30))


def _draw_prob_bar(screen, font, left_prob, right_prob):
    bar_y = HEIGHT - 50
    bar_x, bar_w, bar_h = 80, WIDTH - 160, 16
    pygame.draw.rect(screen, C_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=8)
    left_w = int(left_prob * bar_w)
    pygame.draw.rect(screen, C_LEFT, (bar_x, bar_y, left_w, bar_h), border_radius=8)
    # Right portion
    right_w = bar_w - left_w
    if right_w > 0:
        pygame.draw.rect(screen, C_RIGHT, (bar_x + left_w, bar_y, right_w, bar_h), border_radius=8)

    l_text = font.render(f'L {left_prob:.0%}', True, C_LEFT)
    r_text = font.render(f'R {right_prob:.0%}', True, C_RIGHT)
    screen.blit(l_text, (bar_x - l_text.get_width() - 8, bar_y - 2))
    screen.blit(r_text, (bar_x + bar_w + 8, bar_y - 2))


def _draw_stats(screen, font, trial_idx, n_trials, successes, streak):
    completed = min(trial_idx, n_trials)
    lines = [
        f'Trial {completed}/{n_trials}',
        f'成功 {successes}/{completed}' if completed > 0 else '',
        f'正确率 {successes/completed*100:.0f}%' if completed > 0 else '',
        f'连续 {streak}' if streak >= 2 else '',
    ]
    y = 80
    for line in lines:
        if line:
            surf = font.render(line, True, C_DIM)
            screen.blit(surf, (WIDTH - surf.get_width() - 15, y))
            y += 24


def _draw_final_screen(screen, font_huge, font_large, font_med, font_small,
                       results_history, successes, n_trials, max_streak,
                       trial_times=None):
    completed = len(results_history)
    acc = successes / completed * 100 if completed > 0 else 0

    # Title
    title = font_huge.render("实验结束", True, C_TEXT)
    screen.blit(title, (CENTER_X - title.get_width() // 2, 60))

    # Big accuracy
    acc_color = C_SUCCESS if acc >= 70 else C_RIGHT if acc >= 50 else C_FAIL
    acc_text = font_huge.render(f'{acc:.0f}%', True, acc_color)
    screen.blit(acc_text, (CENTER_X - acc_text.get_width() // 2, 150))

    label = font_med.render(f'{successes} / {completed} 正确', True, C_TEXT)
    screen.blit(label, (CENTER_X - label.get_width() // 2, 230))

    # Left/Right breakdown
    left_trials = [r for r in results_history if r[0] == 0]
    right_trials = [r for r in results_history if r[0] == 1]
    y = 290
    if left_trials:
        l_succ = sum(r[1] for r in left_trials)
        l_text = font_med.render(f'左手: {l_succ}/{len(left_trials)} ({l_succ/len(left_trials)*100:.0f}%)', True, C_LEFT)
        screen.blit(l_text, (CENTER_X - l_text.get_width() // 2, y))
        y += 40
    if right_trials:
        r_succ = sum(r[1] for r in right_trials)
        r_text = font_med.render(f'右手: {r_succ}/{len(right_trials)} ({r_succ/len(right_trials)*100:.0f}%)', True, C_RIGHT)
        screen.blit(r_text, (CENTER_X - r_text.get_width() // 2, y))
        y += 40

    # Overall MI classification accuracy
    all_correct = sum(r[4] for r in results_history)
    all_total = sum(r[5] for r in results_history)
    if all_total > 0:
        mi_text = font_med.render(f'MI分类正确率: {all_correct}/{all_total} ({all_correct/all_total*100:.0f}%)', True, C_TEXT)
        screen.blit(mi_text, (CENTER_X - mi_text.get_width() // 2, y))
        y += 40

    if max_streak >= 2:
        streak_text = font_med.render(f'最长连续正确: {max_streak}', True, C_TEXT)
        screen.blit(streak_text, (CENTER_X - streak_text.get_width() // 2, y))
        y += 40

    if trial_times:
        avg_t = np.mean(trial_times)
        time_text = font_med.render(f'平均用时: {avg_t:.1f}s', True, C_TEXT)
        screen.blit(time_text, (CENTER_X - time_text.get_width() // 2, y))
        y += 40

    # Trial history dots
    y += 20
    dot_r = 8
    dot_gap = 22
    total_w = completed * dot_gap
    start_x = CENTER_X - total_w // 2
    for i, r in enumerate(results_history):
        success = r[1]
        cx = start_x + i * dot_gap + dot_r
        color = C_SUCCESS if success else C_FAIL
        pygame.draw.circle(screen, color, (cx, y + dot_r), dot_r)

    # Quit hint
    hint = font_small.render("按 ESC 退出", True, C_DIM)
    screen.blit(hint, (CENTER_X - hint.get_width() // 2, HEIGHT - 40))


if __name__ == '__main__':
    main()
