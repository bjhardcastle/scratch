import polars as pl

cols = ["session_id", "structure", "electrode_group_name"]
(
    pl.scan_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.265/consolidated/units.parquet')
    .select(cols)
    .unique(cols)
    .group_by('session_id', 'structure')
    .agg(
        pl.col('electrode_group_name').alias('electrode_group_names')
    )
    .with_columns(
        pl.when(pl.col('electrode_group_names').list.n_unique().gt(1))
        .then(pl.col('electrode_group_names').repeat_by(2))
        .otherwise(pl.col('electrode_group_names').repeat_by(1))
    )
    .explode('electrode_group_names')
    .with_columns(
        pl.col('electrode_group_names').list.join('_')
    )
    .with_columns(
        pl.when(pl.col('electrode_group_names').is_first_distinct().over('session_id', 'structure'))
        .then(pl.col('electrode_group_names').cast(pl.List(pl.String)))
        .otherwise(pl.col('electrode_group_names').str.split('_'))
    )
    .explode('electrode_group_names')
    .cast({'electrode_group_names': pl.List(pl.String)})
    .with_columns(
        pl.col('electrode_group_names').list.eval(pl.element().str.split('_'))
    )
    .explode('electrode_group_names')
    .sort('session_id', 'structure', 'electrode_group_names')
    .collect()
)