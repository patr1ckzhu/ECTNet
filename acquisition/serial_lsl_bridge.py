"""
Serial-to-LSL Bridge for ADS1299 + STM32 + HC-05

Reads EEG packets from the HC-05 Bluetooth serial port and pushes
to an LSL stream. Replaces the OpenBCI GUI in the pipeline — once
running, recorder.py / realtime_inference.py work unchanged.

Packet format (27 bytes, from STM32 firmware):
    [0xA0] [seq] [CH1_3B] [CH2_3B] ... [CH8_3B] [0xC0]

Usage:
    python acquisition/serial_lsl_bridge.py --port COM5
    python acquisition/serial_lsl_bridge.py --port /dev/tty.HC-05 --scale

HC-05 pairing (Windows):
    1. Settings → Bluetooth → Add device → HC-05 (PIN: 1234)
    2. Check Device Manager → Ports → find "HC-05" COM port number
"""

import argparse
import time
import serial
from pylsl import StreamInfo, StreamOutlet

# ADS1299 conversion: raw 24-bit count → µV
# LSB = VREF / (gain × 2^23) = 4.5 / (24 × 8388608) ≈ 22.35 nV
VREF = 4.5
GAIN = 24
SCALE_UV = (VREF / (GAIN * (2**23))) * 1e6   # ≈ 0.02235 µV/count

PKT_START = 0xA0
PKT_END   = 0xC0
NUM_CH    = 3       # 3ch for BT demo (C3, C4, Cz); change to 8 for USB/full
PKT_SIZE  = 2 + NUM_CH * 3 + 1   # [A0][seq][N×3B][C0] = 12 (3ch) or 27 (8ch)
SRATE     = 250

CHANNEL_NAMES_3CH = ['C3', 'C4', 'Cz']
CHANNEL_NAMES_8CH = ['C3', 'C4', 'Cz', 'FCz', 'CP1', 'CP2', 'FC3', 'FC4']
CHANNEL_NAMES = CHANNEL_NAMES_3CH if NUM_CH == 3 else CHANNEL_NAMES_8CH


def parse_int24(buf, offset):
    """Parse 24-bit two's complement signed integer (MSB first)."""
    val = (buf[offset] << 16) | (buf[offset + 1] << 8) | buf[offset + 2]
    if val & 0x800000:
        val -= 0x1000000
    return val


def main():
    parser = argparse.ArgumentParser(description='Serial-to-LSL bridge for ADS1299+STM32')
    parser.add_argument('--port', required=True,
                        help='Serial port (e.g. COM5, /dev/tty.HC-05)')
    parser.add_argument('--baud', type=int, default=115200,
                        help='Baud rate (must match STM32 USART2, default: 115200)')
    parser.add_argument('--stream-name', default='obci_eeg1',
                        help='LSL stream name (default: obci_eeg1)')
    parser.add_argument('--scale', action='store_true',
                        help='Convert raw counts to µV (default: raw counts)')
    args = parser.parse_args()

    # ── Create LSL outlet ────────────────────────────────
    info = StreamInfo(args.stream_name, 'EEG', NUM_CH, SRATE,
                      'float32', 'ads1299_stm32_custom')
    chns = info.desc().append_child('channels')
    for name in CHANNEL_NAMES:
        ch = chns.append_child('channel')
        ch.append_child_value('label', name)
        ch.append_child_value('unit', 'microvolts' if args.scale else 'counts')
        ch.append_child_value('type', 'EEG')
    outlet = StreamOutlet(info)

    # ── Open serial port ─────────────────────────────────
    print(f'Opening {args.port} @ {args.baud} baud...')
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)  # HC-05 needs a moment after connection
    ser.reset_input_buffer()

    unit = 'µV' if args.scale else 'raw'
    print(f'Connected. LSL stream: "{args.stream_name}" ({NUM_CH}ch @ {SRATE}Hz, {unit})')
    print('Waiting for data...\n')

    pkt_count = 0
    drop_count = 0
    sync_errors = 0
    last_seq = -1
    last_report = time.time()
    report_count = 0
    wait_dots = 0

    try:
        while True:
            # ── Sync to start marker ─────────────────────
            b = ser.read(1)
            if len(b) == 0:
                # No data from serial port
                if pkt_count == 0:
                    wait_dots = (wait_dots + 1) % 4
                    print(f'\r  Waiting for board{"." * (wait_dots + 1):<4s}', end='', flush=True)
                continue
            if b[0] != PKT_START:
                sync_errors += 1
                if pkt_count == 0:
                    print(f'\r  Syncing... ({sync_errors} bytes skipped)', end='', flush=True)
                continue

            # ── Read remaining 26 bytes ──────────────────
            rest = ser.read(PKT_SIZE - 1)
            if len(rest) < PKT_SIZE - 1:
                continue

            # ── Verify end marker ────────────────────────
            if rest[-1] != PKT_END:
                sync_errors += 1
                continue

            # ── Parse channels ───────────────────────────
            seq = rest[0]
            sample = []
            for ch in range(NUM_CH):
                raw = parse_int24(rest, 1 + ch * 3)
                val = raw * SCALE_UV if args.scale else float(raw)
                sample.append(val)

            # ── Check for dropped packets ────────────────
            if last_seq >= 0:
                expected = (last_seq + 1) & 0xFF
                if seq != expected:
                    missed = (seq - last_seq - 1) & 0xFF
                    drop_count += missed
            last_seq = seq

            # ── Push to LSL ──────────────────────────────
            outlet.push_sample(sample)
            if pkt_count == 0:
                print(f'\r  Locked! Receiving {NUM_CH}ch @ {SRATE}Hz          ')
            pkt_count += 1
            report_count += 1

            # ── Status report every 5 seconds ────────────
            now = time.time()
            elapsed = now - last_report
            if elapsed >= 5.0:
                rate = report_count / elapsed
                print(f'  {rate:6.1f} Hz  |  packets: {pkt_count}  |  '
                      f'dropped: {drop_count}  |  sync_err: {sync_errors}')
                last_report = now
                report_count = 0

    except KeyboardInterrupt:
        print(f'\n\nStopped.')
        print(f'  Total packets: {pkt_count}')
        print(f'  Dropped: {drop_count}')
        print(f'  Sync errors: {sync_errors}')
    finally:
        ser.close()


if __name__ == '__main__':
    main()
