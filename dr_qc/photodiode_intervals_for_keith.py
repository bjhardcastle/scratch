# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "npc-sync",
# ]
# ///

import pathlib
import sys

import matplotlib.pyplot as plt
import npc_sync
import numpy as np

TEST_SYNC_PATH = "//allen/programs/mindscope/workgroups/np-exp/1423899502_760327_20250304/1423899502_760327_20250304.sync"

def plot_intervals(sync_path: str) -> plt.Figure:
    sync = npc_sync.SyncDataset(sync_path)
    diode_events = sync.get_edges(kind='all', keys='stim_photodiode', units='seconds')
    timestamps = np.diff(diode_events) / 2 + diode_events[:-1]
    markerline, stemline, baseline = plt.stem(timestamps, np.diff(diode_events), bottom=1.)
    plt.setp(stemline, linewidth=.5, alpha=.3)
    plt.setp(markerline, markersize=.5, alpha=.8)
    plt.setp(baseline, visible=False)
    plt.xlabel('session time (s)')
    plt.ylabel('diode interval (s)')
    plt.title(pathlib.Path(sync_path).name)
    return plt.gcf()

def main():
    sync_path = sys.argv[1] if len(sys.argv) > 1 else None
    if sync_path is None:
        print(f"Usage: {sys.argv[0]} <path_to_sync_file>")
        print(f"Using default sync path for testing purposes: {TEST_SYNC_PATH}")
    plot_intervals(sync_path or TEST_SYNC_PATH).savefig(f'{pathlib.Path(sync_path).stem}.png')
    
if __name__ == "__main__":
    main()