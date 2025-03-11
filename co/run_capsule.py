import time
import aind_session
import codeocean.computation
import codeocean.data_asset


decoding_pipeline_id = '45d0369a-ba77-42d7-bfb1-fffe62c3bd4e'
datacube_asset_id = 'b59511ab-e888-4f96-8772-5627adc12e31' # v0.0.261
datacube_asset_id = 'bb3a03a8-3977-4dcb-97ec-fc00eeddb135' # v0.0.265

run_params = codeocean.computation.RunParams(
    pipeline_id=decoding_pipeline_id,
    data_assets=[
        codeocean.data_asset.DataAssetAttachParams(
            id=datacube_asset_id,
            mount=aind_session.get_codeocean_model(datacube_asset_id).mount,
        ),
        # add more DataAssetAttachParams as needed..
    ]
    named_parameters=[
        codeocean.computation.NamedRunParam(
            param_name='run_id',
            value='?????', # required
        ),
        codeocean.computation.NamedRunParam(
            param_name='skip_existing',
            value='0',  # all values must be supplied as strings
        ),
        # add more NamedRunParams as needed..
    ],
)

computation = aind_session.get_codeocean_client().computations.run_capsule(run_params)