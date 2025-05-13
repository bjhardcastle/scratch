import time
import aind_session
import codeocean.computation
import codeocean.data_asset
import polars as pl

write_session_cache_capsule_id = '8e9fb886-21e6-4c00-b756-ad43fc4b1466'
decoding_pipeline_id = '45d0369a-ba77-42d7-bfb1-fffe62c3bd4e'
datacube_asset_id = 'b59511ab-e888-4f96-8772-5627adc12e31' # v0.0.261
datacube_asset_id = 'bb3a03a8-3977-4dcb-97ec-fc00eeddb135' # v0.0.265

for session_id in pl.scan_csv(
        r'C:\Users\ben.hardcastle\github\scratch\co\errors_table.csv'
    ).select('session_id').collect()['session_id'][:-2]:
    run_params = codeocean.computation.RunParams(
        capsule_id=write_session_cache_capsule_id,
        # data_assets=[
        #     codeocean.data_asset.DataAssetAttachParams(
        #         id=datacube_asset_id,
        #         mount=aind_session.get_codeocean_model(datacube_asset_id).mount,
        #     ),
        #     # add more DataAssetAttachParams as needed..
        # ],
        named_parameters=[
            codeocean.computation.NamedRunParam(
                param_name='session_id',
                value=session_id, # required
            ),
            codeocean.computation.NamedRunParam(
                param_name='skip_existing',
                value='0',  # all values must be supplied as strings
            ),
            codeocean.computation.NamedRunParam(
                param_name='skip_previously_failed',
                value='0',  # all values must be supplied as strings
            ),
            codeocean.computation.NamedRunParam(
                param_name='run_id',
                value='v268',  # all values must be supplied as strings
            ),
            # add more NamedRunParams as needed..
        ],
    )

    computation = aind_session.get_codeocean_client().computations.run_capsule(run_params)
    print(f"Started computation {computation.id} for session {session_id}")
    time.sleep(60)