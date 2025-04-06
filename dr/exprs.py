import polars as pl

PROD_EPHYS_SESSION_FILTER = pl.Expr.and_(
    *[
        pl.col("keywords").list.contains("production"),
        ~pl.col("keywords").list.contains("issues"),
        pl.col("keywords").list.contains("task"),
        pl.col("keywords").list.contains("ephys"),
        pl.col("keywords").list.contains("ccf"),
        ~pl.col("keywords").list.contains("opto_perturbation"),
        ~pl.col("keywords").list.contains("opto_control"),
        ~pl.col("keywords").list.contains("injection_perturbation"),
        ~pl.col("keywords").list.contains("injection_control"),
        ~pl.col("keywords").list.contains("hab"),
        ~pl.col("keywords").list.contains("training"),
        ~pl.col("keywords").list.contains("context_naive"),
        ~pl.col("keywords").list.contains("templeton"),
    ]
)

LATE_AUTOREWARDS_SESSION_FILTER = (
    pl.col("keywords").list.contains("late_autorewards").eq(True)
)
EARLY_AUTOREWARDS_SESSION_FILTER = (
    pl.col("keywords").list.contains("early_autorewards").eq(True)
)


def get_passing_blocks_performance_filter(
    cross_modal_dprime: float | None = 1.0,
    same_modal_dprime: float | None = None,
    min_n_blocks: int = 2,
    of_each_rewarded_modality: bool = True,
    min_trials: int | None = 10,
    min_contingent_rewards: int | None = 10,
) -> pl.Expr:
    cross_modal_dprime_filter: pl.Expr = (
        pl.col("cross_modal_dprime") > cross_modal_dprime
        if cross_modal_dprime is not None
        else pl.lit(True)
    )
    same_modal_dprime_filter: pl.Expr = (
        pl.col("same_modal_dprime") > same_modal_dprime
        if same_modal_dprime is not None
        else pl.lit(True)
    )
    min_n_trials_filter: pl.Expr = (
        pl.col("n_trials_in_block") > min_trials
        if min_trials is not None
        else pl.lit(True)
    )
    min_n_responses_filter: pl.Expr = (
        pl.col("n_responses_in_block") > min_contingent_rewards
        if min_contingent_rewards is not None
        else pl.lit(True)
    )
    n_blocks_each_context: pl.Expr = (
        pl.col("rewarded_modality")
        .filter(
            cross_modal_dprime_filter,
            same_modal_dprime_filter,
            min_n_trials_filter,
            min_n_responses_filter,
        )
        .unique_counts()
        .over("session_id", mapping_strategy="join")
    )
    if of_each_rewarded_modality:
        return (
            # at least n_blocks of each context, and two contexts:
            n_blocks_each_context.list.eval(pl.element() >= min_n_blocks).list.all()
            & (n_blocks_each_context.list.len() == 2)
        )
    else:
        return (
            # at least n_blocks of any context:
            n_blocks_each_context.list.sum()
            >= min_n_blocks
        )


PASSING_BLOCKS_PERFORMANCE_FILTER = get_passing_blocks_performance_filter()


def get_prod_trials(
    cross_modal_dprime_threshold: float = 1.0, late_autorewards: bool | None = None
) -> pl.DataFrame:
    late_autorewards_expr = {
        True: LATE_AUTOREWARDS_SESSION_FILTER,
        False: EARLY_AUTOREWARDS_SESSION_FILTER,
        None: pl.lit(True),
    }[late_autorewards]


    # session_ids to use based on project, experiment-type, training history etc.:
    session_ids_by_type: pl.Series = (
        utils.get_df("session")
        .filter(PROD_EPHYS_SESSION_FILTER, late_autorewards_expr)
        ["session_id"]
    )
    # session_ids to use based on passing blocks:
    session_ids_by_performance: pl.Series = (
        utils.get_df("performance")
        .filter(PASSING_BLOCKS_PERFORMANCE_FILTER)
        ["session_id"]
    )

    return (
        utils.get_df("trials")
        .filter(
            pl.col("session_id").is_in(
                set(session_ids_by_type).intersection(session_ids_by_performance)
            ),
        )
        # add a column that indicates if the first block in a session is aud context:
        .with_columns(
            (pl.col("context_name").first() == "aud")
            .over("session_id")
            .alias("is_first_block_aud"),
        )
    )
