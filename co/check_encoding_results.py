import polars as pl

lf = pl.scan_parquet("s3://aind-scratch-data/dynamic-routing/encoding/results/v268_4/")
saved_ids = lf.select(pl.col("session_id")).unique().collect()["session_id"]

session_ids = (
    pl.scan_parquet(
        "s3://aind-scratch-data/dynamic-routing/session_metadata/session_table.parquet"
    )
    .filter(
        "is_ephys",
        "is_task",
        "is_annotated",
        "is_production",
        pl.col("issues").list.len() == 0,
    )
    .collect()
)["session_id"]

missing = set(session_ids) - set(saved_ids)
print(f"Missing {len(missing)} sessions: {missing}")