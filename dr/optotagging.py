import polars as pl
import utils

def get_probe_distance_from_opto_stim(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    For each probe, get an approximate horizontal distance from the opto stim site on the surface of the brain.
    
    - requires an opto trials table with 'unit_id' and 'electrode_group_name' already present
    This assumes stimulus location ('probeA' etc.) can be used as a proxy for probe location. 
    If the area around the probe insertion wasn't stimulated, it will get a null value here.
    Depth is not considered.
    """
    x_dist = pl.col('bregma_x').filter(pl.col('location') == pl.col('electrode_group_name')).first().sub(pl.col('bregma_x')).over('session_id', 'electrode_group_name')
    y_dist = pl.col('bregma_y').filter(pl.col('location') == pl.col('electrode_group_name')).first().sub(pl.col('bregma_y')).over('session_id', 'electrode_group_name')
    return (
        df
        .with_columns(
            x_dist.pow(2).mul(y_dist.pow(2)).pow(0.5).alias('stim_distance')
        )
    )


df = pl.scan_parquet('/code/optotagging.parquet')
x_dist = pl.col('bregma_x').filter(pl.col('location') == pl.col('electrode_group_name')).first().sub(pl.col('bregma_x')).over('session_id', 'electrode_group_name')
y_dist = pl.col('bregma_y').filter(pl.col('location') == pl.col('electrode_group_name')).first().sub(pl.col('bregma_y')).over('session_id', 'electrode_group_name')
(
    df
    .join(
        utils.get_df('units', lazy=True).select('unit_id', 'electrode_group_name'),
        on='unit_id',
        how='inner',
    )
    .with_columns(
        # For each probe, get an approximate horizontal distance from the opto stim site on the surface of the brain.

        # - requires an opto trials table with 'unit_id' and 'electrode_group_name' already present
        # This assumes stimulus location ('probeA' etc.) can be used as a proxy for probe location. 
        # If the area around the probe insertion wasn't stimulated, it will get a null value here.
        # Depth is not considered.
        x_dist.pow(2).mul(y_dist.pow(2)).pow(0.5).alias('stim_distance')
    )
    .group_by('unit_id', 'location')
    .agg(
        pl.col('opto_baseline', 'opto_response').median(),
        pl.col('opto_response').sub('opto_baseline').median().alias('reponse_sub_baseline'),
        pl.col('stim_distance', 'duration', 'power', 'wavelength').first(),
    )
    .filter(
        pl.col('duration') == 0.2,
        pl.col('power') == 5,
        pl.col('wavelength') == 488,
        # pl.col('stim_distance') < 1,
        pl.col('opto_response') > pl.col('opto_baseline'),
        # pl.col('reponse_sub_baseline') > 20,
    )
    .sort('unit_id', 'stim_distance')
    .group_by('unit_id', maintain_order=True)
    .agg(
        pl.col('opto_response', 'opto_baseline', 'stim_distance'),
    )
    .join(
        utils.get_df('subject', lazy=True).select('subject_id', 'genotype').unique('subject_id'),
        left_on=pl.col('unit_id').str.split('_').list.get(0).cast(int),
        right_on='subject_id',
        how='inner',
    )
    .drop('subject_id')
    .with_columns(
        pl.col('genotype').str.replace('/', '-').str.split('-').list.get(0).str.to_lowercase(),
    )
    .sort('unit_id')
).collect()