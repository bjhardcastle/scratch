import polars as pl

def transpose(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    index_name = "__index"
    return (
        df
        .explode(col_name)
        .with_columns(pl.int_ranges(pl.lit(0), pl.col(col_name).list.len()).alias(index_name))
        .explode(col_name, index_name)
        .group_by(index_name, pl.all().exclude(index_name, col_name), maintain_order=True)
        .agg(
            pl.all().exclude(index_name, col_name).first(),
            pl.col(col_name)    
        )
        .group_by(pl.all().exclude(index_name, col_name), maintain_order=True)
        .agg(pl.col(col_name))
    )
(
    pl.DataFrame(
        {
            "a": [[[0, 1, 2]] *3] * 3,
            "b": [0, 1, 2],
        }, 
        # schema_overrides={"a": pl.Array(pl.Int64, shape=(3, 3, 3))},
    )
    .pipe(transpose, "a")
    .with_columns(
        pl.col("a").list.slice("b", 1).alias('sliced'),
    )
)