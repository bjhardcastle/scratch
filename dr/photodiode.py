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


def discretize_vsync_times(
    vsync_times: npt.NDArray,
    on_flip_times: npt.NDArray,
    is_first_frame_on: bool,
    frame_rate: float = 60.0,
) -> npt.NDArray:
    """
    Discretize vsync times to multiples of 1/frame_rate, using photodiode signals
    to determine long frames, whether they are visible in vsync intervals or not.
        
    >>> on_frame_times = np.array([1, 4, 6, 10]) * 1/60

    When vsyncs do not show long intervals:
    >>> discretize_vsync_times(np.array([0, 1, 2, 3, 4, 5, 6]) * 1/60, on_frame_times, is_first_frame_on=True) * 60
    array([0. , 1.5, 3. , 4. , 5. , 7. , 9. ])
    
    When vsyncs show long intervals:
    >>> discretize_vsync_times(np.array([0, 1, 3, 4, 5, 6, 9]) * 1/60, on_frame_times, is_first_frame_on=True) * 60
    array([0., 1., 3., 4., 5., 6., 9.])
    
    When first frame is an OFF transition we can't adjust the first vsync time:
    >>> discretize_vsync_times(np.array([0, 1, 2, 3, 4, 5, 6, 7]) * 1/60, on_frame_times, is_first_frame_on=False) * 60
    array([ 0. ,  1. ,  2.5,  4. ,  5. ,  6. ,  8. , 10. ])
    """
    frame_duration = 1.0 / frame_rate
    
    # Get first vsync time as reference
    t0 = vsync_times[0]
    
    if not is_first_frame_on:
        # If the first frame is off, we can't adjust the first vsync time - discard it for now
        vsync_times = vsync_times[1:]

    flip_intervals = np.diff(on_flip_times)
    # Note: flip_intervals represents 2 frames duration (one on_frame + one off_frame)
    expected_interval_frames = 2  # Expected number of frames between consecutive on_frame signals
    
    # Identify long photodiode intervals
    long_intervals = []
    for i, interval_dur in enumerate(flip_intervals):
        # Calculate how many frames this interval spans (expected is 2 frames)
        n_frames_in_interval = round(interval_dur / (frame_duration), ndigits=1)
        
        assert n_frames_in_interval >= expected_interval_frames, (
            f"Short photodiode interval at index {i} - should not happen if photodiode signal has been properly filtered: {interval_dur:.4f}s ≈ {interval_dur/frame_duration:.2f} frames"
        ) 
        assert n_frames_in_interval == round(n_frames_in_interval), f"Interval duration is not a multiple of frame duration (within +-0.1x): {interval_dur:.4f}s ≈ {interval_dur/frame_duration:.2f} frames"
        if n_frames_in_interval > expected_interval_frames:
            long_intervals.append((i, round(n_frames_in_interval)))
                
    # initial discretization of vsync times assuming no dropped frames
    initial_frame_indices = np.arange(len(vsync_times), dtype=float)

    adjusted_frame_indices = initial_frame_indices.copy()
    
    # Process each abnormal interval to determine actual frame timing
    for i, diode_n_frames in long_intervals:

        # get indices of vsyncs for on and off frames
        vsync_idx = (expected_interval_frames * i, (expected_interval_frames * i) + 1)

        vsync_intervals = np.diff(vsync_times[vsync_idx[0]: vsync_idx[0] + 3])
        
        vsync_n_frames = [np.round(interval / frame_duration) for interval in vsync_intervals]
        assert len(vsync_n_frames) == 2
        assert not np.any(vsync_n_frames == 0)
        
        if np.sum(vsync_n_frames) == diode_n_frames:
            # use the n frames estimated from vsync intervals to adjust the off-frame:
            adjustment = vsync_n_frames[0]
            adjusted_frame_indices[vsync_idx[1]] = adjusted_frame_indices[vsync_idx[0]] + adjustment
        else:
            # n frames estimated from vsync intervals is not consistent with estimate from photodiode intervals
            #! set off-frame to be at the center of the interval:
            adjustment = (diode_n_frames / 2)
            assert adjustment > 1, "n_frames should be > 2, otherwise it's not a long interval"
            adjusted_frame_indices[vsync_idx[1]] = adjusted_frame_indices[vsync_idx[0]] + adjustment
            
        # regardless of within-interval modification, shift all subsequent frames to account for
        # long interval:
        adjusted_frame_indices[vsync_idx[1] + 1:] = adjusted_frame_indices[vsync_idx[1] + 1:] + (diode_n_frames - expected_interval_frames)
            
    if not is_first_frame_on:
        # If the first frame is off, we need to add the first vsync time back, and we have to assume
        # it is a single frame interval (in practice this is likely unimportant, as no events happen
        # on the first frame of a stimulus):
        adjusted_frame_indices = np.insert(adjusted_frame_indices + 1, 0, 0) # shift all other times by 1 frame
        
    # Convert adjusted frame indices back to times
    discretized_times = t0 + adjusted_frame_indices * frame_duration

    return discretized_times


def apply_monitor_delay(discretized_vsync_times: npt.NDArray, 
                        on_frame_times: npt.NDArray,
                        is_first_frame_on: bool,
                        window_sec: float = 30,
                        fps: float = 60.0,
                        rolling: bool = True,
                        ) -> npt.NDArray:
    assert 0 <= (len(discretized_vsync_times) / 2) - len(on_frame_times) <= 1
        
    if rolling:
        window = np.ones(window_sec * fps)
        if is_first_frame_on:
            on_vsync_times = discretized_vsync_times[::2]
        else:
            on_vsync_times = discretized_vsync_times[1::2]
        delay = np.convolve(on_frame_times - on_vsync_times, window, mode="same")
        # make delay the same length as discretized_vsync_times:
        return discretized_vsync_times + np.repeat(delay, 2)[:len(discretized_vsync_times)]
    
    else:
        if is_first_frame_on:
            delay = on_frame_times - discretized_vsync_times[::2]
        else:
            delay = on_frame_times - discretized_vsync_times[1::2]
        # add the median of the delay in blocks of window_sec * fps to vsync times,
        # making sure that the last block isn't skipped
        adjusted_vsync_times = discretized_vsync_times.copy()
        window_len = int(window_sec * fps * 0.5)
        for i in range(0, len(delay), window_len):
            if i + window_len < len(delay):
                adjusted_vsync_times[i : i + 2*window_len] += np.median(delay[i : i + window_len])
            else:
                adjusted_vsync_times[i:] += np.median(delay[i:])
        return adjusted_vsync_times


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


# Execute the processing
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    exit()
    
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
    for block_vsyncs in sync.vsync_times_in_blocks:
        first = np.searchsorted(all_frame_times, block_vsyncs[0], 'right')
        frame_times.extend(all_frame_times[first: first + len(block_vsyncs)])
    frame_times = np.array(frame_times)

    len(all_frame_times), len(vsync_times), len(frame_times), sum(
        [len(a) for a in sync.frame_display_time_blocks]
    )

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
    #!   - need to know if first vsync corresponds to an on or off frame 
    # 2. in epochs, estimate monitor delay (adjusted vsync time to rising edge of photodiode signal on sync)
    # 3. apply monitor delay to adjusted vsync times within epoch