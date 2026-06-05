import array
import math
import random

from .log import log


def _add_kick(samples: array.array, start_idx: int, fs: int) -> None:
    kick_len = int(0.12 * fs)
    f_start = 150.0
    f_end = 40.0
    for j in range(kick_len):
        idx = start_idx + j
        if idx >= len(samples):
            break
        t = j / fs
        # Linear pitch sweep
        phase = 2.0 * math.pi * (f_start * t + 0.5 * (f_end - f_start) * (t**2) / 0.12)
        val = math.sin(phase)
        env = math.exp(-t * 25.0)
        amp = 15000.0 * env * val
        cur = samples[idx] + int(amp)
        if cur > 32767:
            cur = 32767
        elif cur < -32768:
            cur = -32768
        samples[idx] = cur


def _add_clap(samples: array.array, start_idx: int, fs: int) -> None:
    clap_len = int(0.18 * fs)
    for j in range(clap_len):
        idx = start_idx + j
        if idx >= len(samples):
            break
        t = j / fs
        noise = random.uniform(-1.0, 1.0)  # noqa: S311
        if t < 0.01:
            env = 0.2
        elif t < 0.02:
            env = 0.35
        elif t < 0.03:
            env = 0.5
        else:
            env = 0.8 * math.exp(-(t - 0.03) * 18.0)
        amp = 10000.0 * env * noise
        cur = samples[idx] + int(amp)
        if cur > 32767:
            cur = 32767
        elif cur < -32768:
            cur = -32768
        samples[idx] = cur


def _add_cymbal(samples: array.array, start_idx: int, fs: int) -> None:
    cymbal_len = int(0.6 * fs)
    for j in range(cymbal_len):
        idx = start_idx + j
        if idx >= len(samples):
            break
        t = j / fs
        noise = random.uniform(-1.0, 1.0)  # noqa: S311
        metal = (
            math.sin(2.0 * math.pi * 8000.0 * t)
            + math.sin(2.0 * math.pi * 9500.0 * t)
            + math.sin(2.0 * math.pi * 11000.0 * t)
        ) / 3.0
        sig = 0.7 * noise + 0.3 * metal
        env = 0.3 * math.exp(-t * 6.0)
        amp = 8000.0 * env * sig
        cur = samples[idx] + int(amp)
        if cur > 32767:
            cur = 32767
        elif cur < -32768:
            cur = -32768
        samples[idx] = cur


async def detect_bpm_and_offset(
    src: str, start_time: float | None, duration: float | None
) -> tuple[float, float]:
    from .ffmpeg import run_ffmpeg

    analysis_start = start_time if start_time is not None else 0.0
    dur_arg = ('-t', str(duration)) if duration is not None and duration < 30.0 else ('-t', '30')

    pcm_data = await run_ffmpeg(
        '-ss',
        str(analysis_start),
        '-i',
        src,
        *dur_arg,
        '-f',
        's16le',
        '-ac',
        '1',
        '-ar',
        '11025',
        'pipe:1',
        desc='bpm_detect',
        capture=True,
    )

    samples = array.array('h')
    samples.frombytes(pcm_data)

    fs = 11025
    frame_size = 512
    frame_dur = frame_size / fs
    num_frames = len(samples) // frame_size
    if num_frames < 10:
        raise ValueError('Not enough audio data for BPM detection')

    energies = []
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frame = samples[start:end]
        avg_abs = sum(abs(x) for x in frame) / len(frame)
        energies.append(avg_abs)

    novelty = [0.0] * num_frames
    for i in range(1, num_frames):
        diff = energies[i] - energies[i - 1]
        if diff > 0:
            novelty[i] = diff

    # Smooth novelty curve with [0.5, 1.0, 0.5] window
    smoothed = [0.0] * num_frames
    for i in range(num_frames):
        val = novelty[i]
        if i > 0:
            val += 0.5 * novelty[i - 1]
        if i < num_frames - 1:
            val += 0.5 * novelty[i + 1]
        smoothed[i] = val

    mean_val = sum(smoothed) / len(smoothed)
    centered = [val - mean_val for val in smoothed]

    # Pass 1: Coarse search (BPM 80 to 160 in steps of 1.0, offset in frames)
    best_score = -float('inf')
    best_bpm = 120.0
    best_offset_sec = 0.0

    for bpm in range(80, 161):
        beat_interval = 60.0 / bpm
        beat_frames = beat_interval / frame_dur
        max_offset_frames = int(beat_frames)
        if max_offset_frames <= 0:
            continue
        for offset_frames in range(max_offset_frames):
            score = 0.0
            k = 0
            while True:
                idx = round(offset_frames + k * beat_frames)
                if idx >= num_frames:
                    break
                score += centered[idx]
                k += 1
            if score > best_score:
                best_score = score
                best_bpm = float(bpm)
                best_offset_sec = offset_frames * frame_dur

    # Pass 2: Fine search (BPM ±1.0 around best_bpm in steps of 0.1, offset in 10ms steps)
    fine_best_score = -float('inf')
    fine_best_bpm = best_bpm
    fine_best_offset_sec = best_offset_sec

    for bpm_diff in range(-10, 11):
        bpm = best_bpm + bpm_diff * 0.1
        if bpm < 50.0:
            continue
        beat_interval = 60.0 / bpm
        num_steps = int(beat_interval / 0.01)
        if num_steps <= 0:
            continue
        for step in range(num_steps):
            offset_sec = step * 0.01
            score = 0.0
            k = 0
            while True:
                t = offset_sec + k * beat_interval
                idx = round(t / frame_dur)
                if idx >= num_frames:
                    break
                score += centered[idx]
                k += 1
            if score > fine_best_score:
                fine_best_score = score
                fine_best_bpm = bpm
                fine_best_offset_sec = offset_sec

    log.info('Detected BPM: %.1f, Offset: %.3f s', fine_best_bpm, fine_best_offset_sec)
    return fine_best_bpm, fine_best_offset_sec


def generate_drum_track(
    filepath: str,
    duration: float,
    bpm: float,
    offset: float,
) -> None:
    import wave

    fs = 44100
    num_samples = int(duration * fs)
    samples = array.array('h', [0] * num_samples)

    beat_interval = 60.0 / bpm
    start_t = offset % beat_interval
    k_offset = round((offset - start_t) / beat_interval)

    k = 0
    while True:
        t = start_t + k * beat_interval
        start_idx = int(t * fs)
        if start_idx >= num_samples:
            break

        beat_idx = (k - k_offset) % 4

        if beat_idx == 0:
            _add_kick(samples, start_idx, fs)
            _add_cymbal(samples, start_idx, fs)
        elif beat_idx == 2:
            _add_kick(samples, start_idx, fs)
        elif beat_idx in (1, 3):
            _add_clap(samples, start_idx, fs)

        k += 1

    with wave.open(filepath, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(samples.tobytes())
