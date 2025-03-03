import time
from typing import Iterable

import polars as pl
import polars._typing


def n_spikes(
    intervals_frame: polars._typing.FrameType,
    spike_times: Iterable[float],
    start: pl.Expr,
    end: pl.Expr,
    col_name: str = "n_spikes",
    start_inclusive: bool = True,
    end_inclusive: bool = True,
    keep_spike_times: bool = False,
) -> polars._typing.FrameType:
    """Count the number of spikes within each interval and return `intervals_frame` with `col_name`
    added. Optionally keep the spike times within each interval (disabled by default to save memory)."""
    if isinstance(intervals_frame, pl.LazyFrame):
        intervals_lf = intervals_frame
    else:
        intervals_lf = intervals_frame.lazy()
    with_spike_count = (
        intervals_lf
        # add temporary columns for the interval start and stop times
        .with_columns(
            start.alias("_start"),
            end.alias("_end"),
        )
        # extract spike times within each interval as: start <= spike_times <= stop
        .join_where(
            pl.LazyFrame({"spike_times": spike_times}),
            (
                start.le(pl.col("spike_times"))
                if start_inclusive
                else start.lt(pl.col("spike_times"))
            ),
            (
                pl.col("spike_times").le(end)
                if end_inclusive
                else pl.col("spike_times").lt(end)
            ),
        )
        # at this point, the df has one row per spike; we want a list of spike times per interval
        .sort("spike_times")
        .group_by("_start", "_end", maintain_order=True)
        .agg(
            *[
                # all existing columns were duplicated with the join_where, so take the first:
                pl.all().exclude("spike_times").first(),  
                pl.col("spike_times").count().alias(col_name),
            ]
            + (
                # optionally keep the spike times:
                [pl.col("spike_times")] if keep_spike_times else []
            ),  
        )
        .sort("_start")
        .drop("_start", "_end")
    )
    if isinstance(intervals_frame, pl.LazyFrame):
        return with_spike_count
    return with_spike_count.collect()


def get_spike_counts_per_interval(
    trials_frame: polars._typing.FrameType,
    starts: pl.Expr | Iterable[pl.Expr],
    ends: pl.Expr | Iterable[pl.Expr],
    units_frame: polars._typing.FrameType,
    col_names: str | Iterable[str] = 'n_spikes',
    unit_id_col: str = "unit_id",
    start_inclusive: bool = True,
    end_inclusive: bool = True,
    rechunk: bool = True,
) -> polars._typing.FrameType:
    """For"""
    if isinstance(starts, pl.Expr):
        starts = (starts, )
    if isinstance(ends, pl.Expr):
        ends = (ends, )
    if isinstance(col_names, str):
        col_names = (col_names, )
    if len(set(col_names)) != len(col_names):
        raise ValueError("col_names must be unique")
    if len(starts) != len(ends) != len(col_names):
        raise ValueError("starts, ends, and col_names must have the same length")
    
    if isinstance(trials_frame, pl.LazyFrame):
        trials_lf = trials_frame
    else:
        trials_lf = trials_frame.lazy()
        
    units_df = units_frame.select(unit_id_col, 'spike_times')
    if isinstance(units_frame, pl.LazyFrame):
        units_df = units_df.collect()
    
    dfs = []
    # for each unit:
    for (unit_id, *_), unit_df in units_df.group_by(unit_id_col):
        assert len(unit_df) == 1, "Expected one row per unit"
        temp_intervals = (
            trials_lf
            .with_columns(
                pl.lit(unit_id).alias(unit_id_col),
            )
        )
        # for each interval requested:
        for start, end, col_name in zip(starts, ends, col_names):
            # spike count column will be appended each iteration, and we overwrite the variable:
            temp_intervals = (
                n_spikes(
                    intervals_frame=temp_intervals,
                    spike_times=unit_df["spike_times"][0],
                    start=start,
                    end=end,
                    col_name=col_name,
                    start_inclusive=start_inclusive,
                    end_inclusive=end_inclusive,
                    keep_spike_times=False,
                )
            )
        dfs.append(temp_intervals)
        
    df = pl.concat(dfs, rechunk=rechunk)
    # return frame in the same format as the input
    if isinstance(trials_frame, pl.LazyFrame):
        return df
    else:
        assert isinstance(df, pl.LazyFrame)
        return df.collect()


trials = pl.DataFrame(
    {"start_time": range(0, 900, 100), "end_time": range(100, 1000, 100)}
)

df = (
    trials
    .pipe(
        n_spikes,
        spike_times=range(1000),
        start=pl.col("start_time") + 10,
        end=pl.col("end_time") - 20,
        start_inclusive=False,
        end_inclusive=True,
        keep_spike_times=False,
    )
)
print(df)

## two intervals per trial:
t0 = time.time()
# df = get_spike_counts_per_interval(
#     trials_frame=pl.scan_parquet("C:/Users/ben.hardcastle/github/scratch/dr_qc/trials.parquet"),
#     starts=(pl.col("start_time"), pl.col("stim_start_time"), ),
#     ends=(pl.col("stim_start_time"), pl.col("response_window_stop_time"), ),
#     col_names=("baseline", "response",),
#     units_frame=pl.scan_parquet("C:/Users/ben.hardcastle/github/scratch/dr_qc/units.parquet"),
# )
print(f"time elapsed: {time.time() - t0:.2f} s")

## one interval per trial:
t0 = time.time()
df = get_spike_counts_per_interval(
    trials_frame=pl.scan_parquet("C:/Users/ben.hardcastle/github/scratch/dr_qc/trials.parquet"),
    starts=(pl.col("start_time"), ),
    ends=(pl.col("stim_start_time"),),
    col_names=("baseline",),
    units_frame=pl.scan_parquet("C:/Users/ben.hardcastle/github/scratch/dr_qc/units.parquet"),
)
print(df.collect())

print(f"time elapsed: {time.time() - t0:.2f} s")