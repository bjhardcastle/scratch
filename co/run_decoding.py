import time

import aind_session
import codeocean.computation
import codeocean.data_asset

decoding_capsule_id = "5f65e876-7b76-4269-9927-79bf181e6e11"

for unit_subsample_size in [10, 20, 30, 40, None][::-1]:
    # empty string indicates no minimum unit count.
    for unit_criteria in [
        "loose_drift"
    ]:  # ['no_drift', 'loose_drift', 'medium_drift', 'strict_drift']:
        named_parameters=[
            codeocean.computation.NamedRunParam(
                param_name="result_prefix",
                value="v265",  # required
            ),
            codeocean.computation.NamedRunParam(
                param_name="run_id",
                value="0",
            ),
            codeocean.computation.NamedRunParam(
                param_name="unit_criteria",
                value=unit_criteria,
            ),
            codeocean.computation.NamedRunParam(
                param_name="skip_existing",
                value='1',  # all values must be supplied as strings
            ),
            codeocean.computation.NamedRunParam(
                param_name="test",
                value='0',
            ),
            # add more NamedRunParams as needed..
        ]
        if unit_subsample_size is not None:
            named_parameters.append(
                codeocean.computation.NamedRunParam(
                    param_name="unit_subsample_size",
                    value=str(unit_subsample_size),
                )
            )
        run_params = codeocean.computation.RunParams(
            capsule_id=decoding_capsule_id,
            named_parameters=named_parameters,
        )

        computation = aind_session.get_codeocean_client().computations.run_capsule(
            run_params
        )
        time.sleep(5)
