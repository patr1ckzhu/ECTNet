"""
PsychoPy Motor Imagery Experiment Paradigm

Displays left/right arrow cues and sends LSL markers for EEG recording.
Includes block breaks and progress display to reduce subject fatigue.

Usage:
    python acquisition/paradigm.py              # default 15 per class = 30 trials
    python acquisition/paradigm.py -n 25        # 25 per class = 50 trials
    python acquisition/paradigm.py -n 50        # 50 per class = 100 trials
    python acquisition/paradigm.py -n 50 --block-size 25  # break every 25 trials

Controls:
    SPACE - start experiment / resume after break
    ESC   - abort
"""

import argparse
import random
import numpy as np
from psychopy import visual, core, event
from pylsl import StreamInfo, StreamOutlet


# --- Configuration ---
FIXATION_RANGE = (1.5, 2.5)   # seconds
CUE_DURATION = 4.0            # seconds
REST_RANGE = (2.0, 3.0)       # seconds
BLOCK_SIZE = 15               # trials per block before break
BREAK_DURATION = 30           # break duration in seconds

MARKER_LEFT = 1
MARKER_RIGHT = 2


def make_trial_list(n_per_class):
    """Create shuffled list of trials: n left + n right."""
    trials = [MARKER_LEFT] * n_per_class + [MARKER_RIGHT] * n_per_class
    random.shuffle(trials)
    return trials


def run(n_per_class=15, block_size=BLOCK_SIZE):
    total = n_per_class * 2
    n_blocks = (total + block_size - 1) // block_size
    est_min = total * 8.5 / 60

    # --- LSL marker outlet ---
    info = StreamInfo('MI-Markers', 'Markers', 1, 0, 'int32', 'paradigm_mi')
    outlet = StreamOutlet(info)

    # --- PsychoPy window ---
    win = visual.Window(fullscr=True, color=[0, 0, 0], units='height')

    fixation = visual.TextStim(win, text='+', height=0.1, color='white')
    arrow_left = visual.TextStim(win, text='\u2190', height=0.2, color='white')   # ←
    arrow_right = visual.TextStim(win, text='\u2192', height=0.2, color='white')  # →
    progress_text = visual.TextStim(win, text='', height=0.03, color=[0.5, 0.5, 0.5],
                                    pos=(0, -0.45))
    instr = visual.TextStim(win, text=(
        'Motor Imagery Experiment\n\n'
        'When you see an arrow, imagine moving that hand.\n'
        '\u2190 = left hand    \u2192 = right hand\n\n'
        '要点:\n'
        '- 开始前先实际握拳3次，记住肌肉发力的感觉\n'
        '- 看到箭头后，想象同样的握拳感觉，但不要真的动\n'
        '- 专注于手掌和前臂的肌肉感觉，不要想画面\n'
        '- 保持身体放松，不要咬牙、皱眉、耸肩\n\n'
        f'{total} trials ({n_blocks} blocks of {block_size}) ~{est_min:.1f} min\n\n'
        'Press SPACE to start'
    ), height=0.035, color='white', wrapWidth=1.5)
    end_text = visual.TextStim(win, text='Experiment complete.\nThank you!',
                               height=0.06, color='white')

    # --- Instruction screen ---
    instr.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

    trials = make_trial_list(n_per_class)

    for i, marker in enumerate(trials):
        # Check for ESC
        if event.getKeys(keyList=['escape']):
            print(f'Aborted at trial {i + 1}/{total}')
            break

        # --- Block break ---
        if i > 0 and i % block_size == 0:
            block_num = i // block_size
            break_text = visual.TextStim(win, text=(
                f'Block {block_num}/{n_blocks} complete!\n\n'
                f'Rest for {BREAK_DURATION} seconds...\n'
                f'{total - i} trials remaining'
            ), height=0.05, color='white')

            # Countdown break
            for sec in range(BREAK_DURATION, 0, -1):
                break_text.text = (
                    f'Block {block_num}/{n_blocks} complete!\n\n'
                    f'Rest: {sec}s\n'
                    f'{total - i} trials remaining'
                )
                break_text.draw()
                win.flip()
                core.wait(1.0)
                if event.getKeys(keyList=['escape']):
                    print(f'Aborted during break')
                    win.close()
                    core.quit()
                    return

            # Resume prompt
            resume_text = visual.TextStim(win, text=(
                f'Ready for block {block_num + 1}/{n_blocks}?\n\n'
                'Press SPACE to continue'
            ), height=0.05, color='white')
            resume_text.draw()
            win.flip()
            event.waitKeys(keyList=['space', 'escape'])
            if event.getKeys(keyList=['escape']):
                break

        # 1) Fixation cross (1.5-2.5s) + progress
        fixation_dur = random.uniform(*FIXATION_RANGE)
        progress_text.text = f'Trial {i + 1} / {total}'
        fixation.draw()
        progress_text.draw()
        win.flip()
        core.wait(fixation_dur)

        # 2) Cue arrow — send marker right after flip
        arrow = arrow_left if marker == MARKER_LEFT else arrow_right
        arrow.draw()
        progress_text.draw()
        win.flip()
        outlet.push_sample([marker])
        label = 'LEFT' if marker == MARKER_LEFT else 'RIGHT'
        print(f'Trial {i + 1}/{total}: {label}')

        # 3) MI period (4s total from cue onset)
        core.wait(CUE_DURATION)

        # 4) Rest (blank screen, 2-3s)
        rest_dur = random.uniform(*REST_RANGE)
        win.flip()
        core.wait(rest_dur)

    # --- End screen ---
    end_text.draw()
    win.flip()
    core.wait(3.0)

    win.close()
    core.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MI Experiment Paradigm')
    parser.add_argument('-n', '--n-per-class', type=int, default=15,
                        help='Number of trials per class (default: 15, total = 2x)')
    parser.add_argument('--block-size', type=int, default=BLOCK_SIZE,
                        help=f'Trials per block before break (default: {BLOCK_SIZE})')
    args = parser.parse_args()
    print(f'Trials: {args.n_per_class} per class = {args.n_per_class * 2} total')
    run(n_per_class=args.n_per_class, block_size=args.block_size)
