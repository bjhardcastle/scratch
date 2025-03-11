import time

import aind_session
import codeocean.computation
import codeocean.data_asset
import polars as pl

client = aind_session.get_codeocean_client()

def write_cache(session_id: str):
    run_params = codeocean.computation.RunParams(
        capsule_id='8e9fb886-21e6-4c00-b756-ad43fc4b1466', # write cache
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
                value='2025-03-08T21:28:41.150114',  # all values must be supplied as strings
            ),
        ],
        
    )
    computation = client.computations.run_capsule(run_params)
    return computation


for session_id in (
        pl.read_csv("C:/Users/ben.hardcastle/Downloads/errors_table (1).csv")
        .filter(
            pl.col('output').str.contains('ConnectionError')
        )
    )['session_id']:
    print(session_id)
    write_cache(session_id)
    time.sleep(120)    time.sleep(120)