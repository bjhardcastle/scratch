import math
from typing import Iterable

import polars as pl
import polars_ds as pds


def psth(
    df: pl.DataFrame,
    response_col: str,
    duration_col: str | pl.Expr,
    group_by: str | list[str] = "index",
    bin_size=0.001,
    conv_kernel=0.005,
    parallel=True,
    with_original=False,
) -> pl.DataFrame:
    """
    Compute the peristimulus time histogram (PSTH) for a given response and duration.
    """
    if isinstance(group_by, str) or not isinstance(group_by, Iterable):
        group_by = [group_by]
    if isinstance(duration_col, str):
        duration = pl.col(duration_col)
    n_bins = duration.first().truediv(bin_size).ceil().cast(int)
    bin_edges = pl.linear_space(
        0, duration.first(), 1 + n_bins, closed="both"
    )
    bin_centers = pl.linear_space(
        bin_size / 2, duration.first() - bin_size / 2, n_bins
    ).alias("bin_centers")
    conv_kernel_size: int = math.ceil(conv_kernel / bin_size)
    if with_original:
        extra_cols = [
            bin_centers,
            pl.col(response_col)
            .hist(bins=bin_edges, include_breakpoint=False)
            .alias("unconv_psth"),
        ]
    else:
        extra_cols = [bin_centers]
    return (
        df.with_row_index()
        .explode(response_col)
        .group_by(group_by)
        .agg(
            pl.all().exclude(response_col),
            psth=pds.convolve(
                pl.col(response_col).hist(bins=bin_edges, include_breakpoint=False),
                kernel=[1] * conv_kernel_size,
                mode="same",
                parallel=parallel,
            ).truediv(
                conv_kernel_size * pl.col("index").n_unique()
            ),  # ,
            # psth=pds.convolve(pl.col(response_col).hist(bins=bin_edges, include_breakpoint=False), kernel=[1]*conv_kernel_size, mode='full', parallel=parallel).slice(conv_kernel_size-1, n_bins).truediv(conv_kernel_size * pl.col('index').n_unique()),#,
            *extra_cols,
        )
        ## DEBUGGING: check lengths of list columns created above -  should all be equal
        # .with_columns(
        #     pl.col('psth', 'unconv_psth', 'bin_centers').list.len(),
        # )
        .drop("index")
    )


if __name__ == "__main__":
    import polars as pl
    import polars_ds as pds

    # Example usage
    df = pl.DataFrame(
        {
            "response": [
                [],
                [0.001, 0.0012, 0.0022, 0.004, 0.0066, 0.009],
                [0.0022, 0.0033, 0.008, 0.009, 0.01],
            ],
            "duration": [0.01, 0.01, 0.01],
            "unit_id": [2, 1, 0],
        }
    )
    print(df)

    result = df.pipe(
        psth,
        "response",
        "duration",
        with_original=True,
        group_by=["duration", "unit_id"],
        bin_size=0.001,
        conv_kernel=0.003,
    )
    print(result)
    (
        result.explode("psth", "unconv_psth", "bin_centers")
        .unpivot(
            on=["psth", "unconv_psth"], index=["bin_centers", "duration", "unit_id"]
        )
        .plot.line(x="bin_centers", y="value", color="variable", row="unit_id")
        .save("psth_example.png")
    )
