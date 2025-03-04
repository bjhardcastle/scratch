import logging
import time
from typing import Iterable

import polars as pl
import polars._typing

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def insert_spike_counts(
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
    added. Optionally keep the spike times within each interval (disabled by default to save memory).
    """
    if isinstance(intervals_frame, pl.LazyFrame):
        intervals_lf = intervals_frame
    else:
        intervals_lf = intervals_frame.lazy()
            
    # note: join_where currently only supports inner join so intervals containing no spikes are lost.
    # To deal with this we must update the original intervals frame:
    with_spike_count = (
        intervals_lf
        .with_columns(
            start.alias('_start'),
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
        .select('_start', 'spike_times')
        # at this point, the df has one row per spike; we want a list of spike times per interval
        .sort("spike_times")
        .group_by("_start",  maintain_order=True)
        .agg(
            *[
                # all existing columns were duplicated with the join_where, so take the first:
                pl.all().exclude("spike_times").first(),
                pl.col("spike_times").len().alias(col_name),
            ]
            + (
                # optionally keep the spike times:
                [pl.col("spike_times")]
                if keep_spike_times
                else []
            ),
        )
    )
    # join counts back to the original intervals frame, to preserve intervals without spikes:
    with_spike_count = (
        intervals_lf
        .with_columns(
            start.alias("_start"),
            pl.lit(0).alias(col_name),
        )
        .update(
            other=with_spike_count.select('_start', col_name),
            on="_start",
            how='left',
            include_nulls=False,
        )
        .sort("_start")
        .drop("_start")
    )
    if isinstance(intervals_frame, pl.LazyFrame):
        return with_spike_count
    return with_spike_count.collect()


def insert_spike_counts_per_interval(
    trials_frame: polars._typing.FrameType,
    starts: pl.Expr | Iterable[pl.Expr],
    ends: pl.Expr | Iterable[pl.Expr],
    units_frame: polars._typing.FrameType,
    col_names: str | Iterable[str] = "n_spikes",
    unit_id_col: str = "unit_id",
    start_inclusive: bool = True,
    end_inclusive: bool = True,
    rechunk: bool = True,
    apply_obs_intervals: bool = True,
) -> polars._typing.FrameType:
    """For every unit, count the number of spikes within each interval defined by pairs of expressions
    in `starts` and `ends`, and store the counts in columns named according to `col_names`.

    By default, obs_intervals in the units table are applied to determine whether a unit was
    actually recorded during each interval: if not, the spike count will be `pl.Null`, rather than
    0. (nulls are converted to nans by the frame.to_numpy() method)

    Notes:
    - returns a frame of the same type as the input `trials_frame` (lazy or eager)
    - returned length will be equal to len(trials) * len(units)
    - spike times themselves are not stored in the returned frame
    - runs in series over each unit, to avoid making copies of spike times


    """
    if isinstance(starts, pl.Expr):
        starts = (starts,)
    if isinstance(ends, pl.Expr):
        ends = (ends,)
    if isinstance(col_names, str):
        col_names = (col_names,)
    if len(set(col_names)) != len(col_names):
        raise ValueError("col_names must be unique")
    if len(starts) != len(ends) != len(col_names):
        raise ValueError("starts, ends, and col_names must have the same length")

    if isinstance(trials_frame, pl.LazyFrame):
        trials_lf = trials_frame
    else:
        trials_lf = trials_frame.lazy()

    units_columns = [unit_id_col, "spike_times"]
    if apply_obs_intervals:
        units_columns.append("obs_intervals")
    units_df = units_frame.select(units_columns)
    if isinstance(units_frame, pl.LazyFrame):
        units_df = units_df.collect() # cannot iterate over LazyFrame.group_by()
    logger.debug(f"adding spike counts for {len(units_df)} units")
    dfs = []
    # for each unit:
    for (unit_id, *_), unit_df in units_df.group_by(unit_id_col):
        assert len(unit_df) == 1, "Expected one row per unit"
        temp_intervals = trials_lf.with_columns(
            pl.lit(unit_id).alias(unit_id_col),
        )
        # for each interval requested:
        for start, end, col_name in zip(starts, ends, col_names):
            # spike count column will be appended each iteration, and we overwrite the variable:
            temp_intervals = insert_spike_counts(
                intervals_frame=temp_intervals,
                spike_times=unit_df["spike_times"][0],
                start=start,
                end=end,
                col_name=col_name,
                start_inclusive=start_inclusive,
                end_inclusive=end_inclusive,
                keep_spike_times=False,
            )
            #! rm after testing:
            assert len(temp_intervals.collect()) == len(trials_lf.collect())
        dfs.append(temp_intervals)
    
    df = pl.concat(dfs, rechunk=rechunk)
    if apply_obs_intervals:
        df = (
            insert_is_observed(
                intervals_frame=df,
                units_frame=units_df,
                col_name="is_observed",
            )
            .with_columns(
                *[
                    pl.when(
                        pl.col("is_observed").eq(False)
                    )
                    .then(None)
                    .otherwise(pl.col(name))
                    .alias(name)
                    for name in col_names
                ]
            )
            .drop('is_observed')
        )
    # return frame in the same format as the input
    if isinstance(trials_frame, pl.LazyFrame):
        return df
    else:
        assert isinstance(df, pl.LazyFrame)
        return df.collect()


def insert_is_observed(
    intervals_frame: polars._typing.FrameType,
    units_frame: polars._typing.FrameType,
    col_name: str = "is_observed",
    unit_id_col: str = "unit_id",
) -> polars._typing.FrameType:

    if isinstance(intervals_frame, pl.LazyFrame):
        intervals_lf = intervals_frame
    else:
        intervals_lf = intervals_frame.lazy()

    if isinstance(units_frame, pl.LazyFrame):
        units_lf = units_frame
    else:
        units_lf = units_frame.lazy()

    units_schema = units_lf.collect_schema()
    if unit_id_col not in units_schema:
        raise ValueError(
            f"units_frame does not contain {unit_id_col!r} column: can be customized by passing unit_id_col"
        )
    if "obs_intervals" not in units_schema:
        raise ValueError("units_frame must contain 'obs_intervals' column")

    unit_ids = units_lf.select(unit_id_col).collect().get_column(unit_id_col).unique()
    intervals_schema = intervals_lf.collect_schema()
    if unit_id_col not in intervals_schema:
        if len(unit_ids) > 1:
            raise ValueError(
                f"units_frame contains multiple units, but intervals_frame does not contain {unit_id_col!r} column to perform join"
            )
        elif len(unit_ids) == 0:
            raise ValueError(
                f"units_frame contains no unit ids in {unit_id_col=} column"
            )
        else:
            intervals_lf = intervals_lf.with_columns(
                pl.lit(unit_ids[0]).alias(unit_id_col)
            )
    if not all(c in intervals_schema for c in ("start_time", "stop_time")):
        raise ValueError(
            "intervals_frame must contain 'start_time' and 'stop_time' columns"
        )

    if units_schema["obs_intervals"] in (
        pl.List(pl.List(pl.Float64())),
        pl.List(pl.List(pl.Int64())),
        pl.List(pl.List(pl.Null())),
    ):
        logger.info("Converting 'obs_intervals' column to list of lists")
        units_lf = units_lf.explode("obs_intervals")
    assert (type_ := units_lf.collect_schema()["obs_intervals"]) == pl.List(
        pl.Float64
    ), f"Expected exploded obs_intervals to be pl.List(f64), got {type_}"
    intervals_lf = (
        intervals_lf.join(
            units_lf.select(unit_id_col, "obs_intervals"), on=unit_id_col, how="left"
        )
        .with_columns(
            pl.when(
                pl.col("obs_intervals").list.get(0).gt(pl.col("start_time"))
                | pl.col("obs_intervals").list.get(1).lt(pl.col("stop_time")),
            )
            .then(pl.lit(False))
            .otherwise(pl.lit(True))
            .alias(col_name),
        )
        .group_by("unit_id", "start_time")
        .agg(
            pl.all().exclude("obs_intervals", col_name).first(),
            pl.col(col_name).any(),
        )
    )
    if isinstance(intervals_frame, pl.LazyFrame):
        return intervals_lf
    return intervals_lf.collect()


trials = pl.DataFrame(
    {"start_time": range(0, 900, 100), "stop_time": range(100, 1000, 100)}
)
df = insert_is_observed(
    trials,
    pl.DataFrame(
        {
            "unit_id": [
                "unit_0",
            ],
            "obs_intervals": [
                [(-10.0, 500.0), (600.0, 1100.0)],
            ],
        }
    ),
)
# print(df)
df = insert_is_observed(trials, pl.DataFrame({"unit_id": ["unit_0",], "obs_intervals": [(-10., 500.), ]}))
# print(df)

t = pl.concat([trials.with_columns(pl.lit("unit_0").alias("unit_id")), trials.with_columns(pl.lit("unit_1").alias("unit_id"))])
# print(t)
# df = insert_is_observed(t, pl.DataFrame({"unit_id": ["unit_0", "unit_1"], "obs_intervals": [[(-10., 150.), (600., 1100.)], [(-100., 1400.)]]}))
# print(df.sort('unit_id', 'start_time').head(20))

# df = trials.pipe(
#     insert_spike_counts,
#     spike_times=range(1000),
#     start=pl.col("start_time") + 10,
#     end=pl.col("start_time") - 20,
#     start_inclusive=False,
#     end_inclusive=True,
#     keep_spike_times=False,
# )
# print(df)

# pl.DataFrame({"unit_id": ["unit_0", "unit_1"], "obs_intervals": [[(-10., 150.), (600., 1100.)], [(-100., 1400.)]]})
# df = insert_is_observed(t, pl.DataFrame({"unit_id": ["unit_0", "unit_1"], "obs_intervals": [[(-10., 150.), (600., 1100.)], [(-100., 1400.)]]}))

## two intervals per trial:
t0 = time.time()

df = insert_spike_counts_per_interval(
    trials_frame=pl.read_parquet("C:/Users/ben.hardcastle/github/scratch/dr_qc/trials.parquet"),
    starts=(pl.col("start_time"), pl.col("stim_start_time"), ),
    ends=(pl.col("stim_start_time"), pl.col("response_window_stop_time"), ),
    col_names=("baseline", "response",),
    units_frame=pl.read_parquet("C:/Users/ben.hardcastle/github/scratch/dr_qc/units.parquet"),
    apply_obs_intervals=True,
)
if isinstance(df, pl.LazyFrame):
    df = df.collect()
print(df.sort('unit_id', 'start_time'))
print(f"time elapsed: {time.time() - t0:.2f} s")

## one interval per trial:
t0 = time.time()
df = insert_spike_counts_per_interval(
    trials_frame=pl.read_parquet("C:/Users/ben.hardcastle/github/scratch/dr_qc/trials.parquet"),
    starts=(pl.col("start_time"), ),
    ends=(pl.col("stim_start_time"),),
    col_names=("baseline",),
    units_frame=pl.read_parquet("C:/Users/ben.hardcastle/github/scratch/dr_qc/units.parquet"),
    apply_obs_intervals=True,

)
if isinstance(df, pl.LazyFrame):
    df = df.collect()
print(df.sort('unit_id', 'start_time'))

print(f"time elapsed: {time.time() - t0:.2f} s")
