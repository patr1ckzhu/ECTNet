"""
PsychoPy Motor Imagery Experiment Paradigm

Displays left/right arrow cues and sends LSL markers for EEG recording.
30 trials (15 left + 15 right), ~4.5 min total.

Usage:
    python acquisition/paradigm.py

Controls:
    SPACE - start experiment
    ESC   - abort
"""

import random
import numpy as np
from psychopy import visual, core, event
from pylsl import StreamInfo, StreamOutlet


# --- Configuration ---
N_TRIALS_PER_CLASS = 15
FIXATION_RANGE = (1.5, 2.5)   # seconds
CUE_DURATION = 4.0            # seconds
REST_RANGE = (2.0, 3.0)       # seconds

MARKER_LEFT = 1
MARKER_RIGHT = 2


def make_trial_list():
    """Create shuffled list of trials: 15 left + 15 right."""
    trials = [MARKER_LEFT] * N_TRIALS_PER_CLASS + [MARKER_RIGHT] * N_TRIALS_PER_CLASS
    random.shuffle(trials)
    return trials


def run():
    # --- LSL marker outlet ---
    info = StreamInfo('MI-Markers', 'Markers', 1, 0, 'int32', 'paradigm_mi')
    outlet = StreamOutlet(info)

    # --- PsychoPy window ---
    win = visual.Window(fullscr=True, color=[0, 0, 0], units='height')

    fixation = visual.TextStim(win, text='+', height=0.1, color='white')
    arrow_left = visual.TextStim(win, text='\u2190', height=0.15, color='white')   # ←
    arrow_right = visual.TextStim(win, text='\u2192', height=0.15, color='white')  # →
    instr = visual.TextStim(win, text=(
        'Motor Imagery Experiment\n\n'
        'When you see an arrow, imagine moving that hand.\n'
        '\u2190 = left hand    \u2192 = right hand\n\n'
        'Press SPACE to start'
    ), height=0.04, color='white', wrapWidth=1.5)
    end_text = visual.TextStim(win, text='Experiment complete.\nThank you!',
                               height=0.06, color='white')

    # --- Instruction screen ---
    instr.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

    trials = make_trial_list()
    clock = core.Clock()

    for i, marker in enumerate(trials):
        # Check for ESC
        if event.getKeys(keyList=['escape']):
            print(f'Aborted at trial {i + 1}/{len(trials)}')
            break

        # 1) Fixation cross (1.5-2.5s)
        fixation_dur = random.uniform(*FIXATION_RANGE)
        fixation.draw()
        win.flip()
        core.wait(fixation_dur)

        # 2) Cue arrow — send marker right after flip
        arrow = arrow_left if marker == MARKER_LEFT else arrow_right
        arrow.draw()
        win.flip()
        outlet.push_sample([marker])
        print(f'Trial {i + 1}/{len(trials)}: {"LEFT" if marker == MARKER_LEFT else "RIGHT"}')

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
    run()
