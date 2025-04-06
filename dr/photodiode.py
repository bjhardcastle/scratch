import matplotlib.pyplot as plt
import npc_ephys
import npc_sync
import numpy as np
import numpy.typing as npt
import scipy.signal


def norm(x: npt.NDArray):
    shifted = x - np.min(x)
    return shifted / np.max(shifted)


def get_crossings(signal, level=0.1) -> tuple[npt.NDArray[int], npt.NDArray[int]]:
    # - keep level low in case there are multiple steps in brightness
    # - use diff with n>1, on a smoothed signal, to avoid spurious crossings
    crossing_pos = np.where(np.diff(np.sign(signal - level), n=3) > 0)[0]
    crossing_neg = np.where(np.diff(np.sign(signal - level), n=3) < 0)[0]
    assert (
        np.abs(len(crossing_pos) - len(crossing_neg)) < 2
    ), f"{len(crossing_pos)=}, {len(crossing_neg)=}: should be within 1 of each other"
    last = min(len(crossing_pos), len(crossing_neg)) - 1
    assert np.all(np.sign(crossing_pos[:last] - crossing_neg[:last]))
    return crossing_pos, crossing_neg


def get_frame_on_off_samples(diode, rate):
    signal_norm = norm(
        scipy.signal.savgol_filter(diode, window_length=int(rate / 500), polyorder=4)
    )
    assert len(signal_norm) == len(diode)
    vel = norm(
        np.diff(
            scipy.signal.savgol_filter(
                signal_norm, window_length=int(rate / 1000), polyorder=3
            )
        )
    )
    accel = norm(
        scipy.signal.savgol_filter(
            np.abs(np.diff(vel)), window_length=int(rate / 1000), polyorder=3
        )
    )
    rising_crossings, falling_crossings = get_crossings(signal_norm, level=0.1)
    print(f"{len(rising_crossings)=}, {len(falling_crossings)=}")

    all_crossings = np.sort(np.concatenate([rising_crossings, falling_crossings]))
    on = []
    off = []
    expected_frametime = rate / 60
    first_crossing_pos = rising_crossings[0] < falling_crossings[0]

    if first_crossing_pos:
        is_ON_frame = 1
    else:
        is_ON_frame = 0
    count_too_short = 0
    for i in range(len(all_crossings) - 1):
        is_ON_frame = (
            signal_norm[all_crossings[i]] - signal_norm[max(all_crossings[i] - 1, 0)]
            > 0
        )

        # for every crossing, find the velocity peak within a preceding interval
        if is_ON_frame:
            interval_len = 0.2 * expected_frametime  # onset is fast
        else:
            interval_len = 0.9 * expected_frametime  # offset is slow
        start = max(0, int(all_crossings[i] - interval_len))
        stop = all_crossings[i]

        rel_sample = np.argmax(np.abs(accel[start:stop]))
        abs_sample = rel_sample + start - 1  # compensate for diffs

        if is_ON_frame and on or not is_ON_frame and off:
            too_close = (
                abs_sample - (on[-1] if is_ON_frame else off[-1])
                < 1.8 * expected_frametime
            )

            if too_close:
                # skip this point
                count_too_short += 1
                continue
                if interval > 0.05:
                    plt.plot(signal_norm[all_crossings[i - 1] : all_crossings[i + 1]])
                    plt.plot(
                        all_crossings[i - 1 : i + 1] - all_crossings[i - 1],
                        signal_norm[all_crossings[i - 1 : i + 1]],
                        "*",
                    )
                    print(f"short {interval=} frames")
                break
        if is_ON_frame:
            on.append(abs_sample)
        else:
            off.append(abs_sample)

    print(f"{count_too_short=}")
    print(f"{len(on)=}, {len(off)=}")
    print(
        f"{np.diff(on).min()/expected_frametime=}, {np.diff(off).min()/expected_frametime=}"
    )
    print(
        f"{np.median(np.diff(on))/expected_frametime=}, {np.median(np.diff(off))/expected_frametime=}"
    )

    # checks:
    for i in range(len(on) - 2):
        if on[i + 1] < off[i]:
            a = on[i]
            b = on[i + 1]
            raise AssertionError(f"{on[i]-a}, {off[i]-a}, {on[i+1]-a}")
    for array in (on, off):
        for i in range(1, len(array) - 1):
            if array[i] == array[i - 1]:
                raise AssertionError(f"{array[i]=} == {array[i - 1]=}")
            if array[i] - array[i - 1] < 1.8 * expected_frametime:
                raise AssertionError(f"{array[i] - array[i - 1]=}")

    return on, off


sync = npc_sync.SyncDataset(
    "/scratch/photodiode/DRpilot_366122_20250106/20250106T191444.h5"
)
recording_dir = (
    "/scratch/photodiode/DRpilot_366122_20250106/Record Node 104/experiment1/recording1"
)
diode = npc_ephys.get_pxi_nidaq_data(recording_dir)[:, 0]

nominal_diode_sampling_rate = 30_000

on_frame_samples, off_frame_samples = get_frame_on_off_samples(
    diode, nominal_diode_sampling_rate
)

sync_timing_info = next(
    npc_ephys.get_ephys_timing_on_sync(
        sync, recording_dirs=[recording_dir], only_devices_including="NI-DAQmx"
    )
)

# - adjust on/off times from photodiode to match sync clock
on_frame_times = (
    sync_timing_info.start_time
    + np.array(on_frame_samples) / sync_timing_info.sampling_rate
)
off_frame_times = (
    sync_timing_info.start_time
    + np.array(off_frame_samples) / sync_timing_info.sampling_rate
)

# - find next photodiode time following each vsync
all_frame_times = np.sort(np.concatenate([on_frame_times, off_frame_times]))
vsync_times = sync.get_falling_edges("vsync_stim", units="seconds")
frame_times = []
for block in sync.vsync_times_in_blocks:
    first = np.searchsorted(all_frame_times, block[0], 'right')
    frame_times.extend(all_frame_times[first: first + len(block)])
frame_times = np.array(frame_times)

len(all_frame_times), len(vsync_times), len(frame_times), sum(
    [len(a) for a in sync.frame_display_time_blocks]
)
def plot_diode_around_time(
    event_time: float,
    duration: float = 0.1,
):
    event_sample = int(
        (event_time - sync_timing_info.start_time) * sync_timing_info.sampling_rate
    )
    event_sample_range = slice(
        event_sample - int(duration * sync_timing_info.sampling_rate),
        event_sample + int(duration * sync_timing_info.sampling_rate),
    )
    t = (
        sync_timing_info.start_time
        + np.arange(len(diode))[event_sample_range] / sync_timing_info.sampling_rate
    )
    plt.figure()
    plt.plot(t, diode[event_sample_range], c="k", lw=0.5)
    # plt.axvline(t[len(t)//2], color="k", linestyle="--", lw=.5)

    def filt(array):
        return np.array(array)[
            (np.array(array) >= event_time - duration) & (np.array(array) <= event_time + duration)
        ]
    
    def plot(array, **kwargs):
        for t in filt(array):
            kwargs.setdefault("linewidth", 0.5)
            kwargs.setdefault("linestyle", "--")
            plt.axvline(t, **kwargs)
    
    plot(vsync_times, color="cyan", lw=.8)
    plot(frame_times, color="magenta")
    # plot(np.concatenate(sync.frame_display_time_blocks) - npc_sync.MONITOR_CENTER_REFRESH_TIME, color="orange") 
    # plot(vsync_times + 0.0182 , color="red") 
    return plt.gcf()

# plot_diode_around_time(sync.vsync_times_in_blocks[0][0], 0.15)

a = vsync_times
event_times = a[
    np.where(
        # (np.diff(a) > 0.018) & (np.diff(a) < 0.03)
        (np.diff(a) < 0.01)
    )[0]
]
for t in event_times[:2]:
    plot_diode_around_time(t)

# 1. discretize vsync times to multiples of 1/60
#    - use photdiode signal to corroborate long intervals 
# 2. in epochs, estimate monitor delay (adjusted vsync time to rising edge of photodiode signal on sync)
# 3. apply monitor delay to adjusted vsync times within epoch
