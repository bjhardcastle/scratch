# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "polars",
#     "zarr",
# ]
# ///
# run with `uv run filename.py`

import polars as pl
import zarr

cache_path = 's3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.265/consolidated'

# get a dataframe for lazy evaluation:
units_lf: pl.LazyFrame = pl.scan_parquet(f"{cache_path}/units.parquet")

# parquet files have a schema (corresponding to pyarrow in-memory data types, also used by polars):
print(f"{units_lf.collect_schema() = }")

# write a query to filter and extract a single unit:
example_unit: dict = (
    units_lf
    .filter(
        # - comma-separated expressions are treated as logical AND
        # - selects rows where the expression evaluates to True
        # - specifying a boolean column by name is sufficient for filtering
        # - all other operations on the column require `pl.col('column_name')` to access methods and properties
        'is_not_drift',                             
        pl.col('amplitude_cutoff') < 0.1,           # shorthand for pl.col('amplitude_cutoff').lt(0.1)
        pl.col('isi_violations_ratio') < 0.5,
        (pl.col('num_spikes').gt(1_000) | pl.col('presence_ratio').gt(0.7)),
        # use .str / .list etc. to access appropriate methods for the c olumn's data type:
        pl.col('structure').str.contains('MOs'),    
    )
    .join(
        other=(
            pl.scan_parquet(f"{cache_path}/session.parquet")
            .filter(
                ~pl.col('keywords').list.contains('issues'),
            )
            .select(
                'session_id',   
                # create and return a new column:
                pl.col('keywords').list.eval(pl.element().str.contains('perturbation')).list.any().alias('is_perturbation'),
            )
        ),
        on='session_id',
        how='left',
    )
    .filter('is_perturbation')          # filter on joined column (we could have also used `how='semi'` in the join)
    .select('session_id', 'unit_id')    # only keep these columns
    .first()                            # only keep the first row (`head(len)` and `slice(offset, len)` also available)
    .collect()                          # run query optimizer and materialize the result
    .to_dicts()[0]
)
print(f"{example_unit = }")

# spike times are stored separate from the units table (large list columns negatively impact
# performance of parquet files in cloud storage):
spike_times = zarr.open(f'{cache_path}/spike_times.zarr')

# looking up data in this zarr file is unfortunately slow:
unit_spike_times = spike_times[example_unit['session_id']][example_unit['unit_id']]
print(f"{unit_spike_times = }")