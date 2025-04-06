"""
Normally, the values passed to init are used, combined with the default values:
    >>> CapsuleParameters(n_units=[0, 1])
    CapsuleParameters(n_units=[0, 1], logging_level='INFO', test=False)

With a `parameters.json` file containing `{"n_units": [25, 50, 100]}`, we don't need to pass
the `n_units` parameter to the constructor:
    >>> CapsuleParameters()
    CapsuleParameters(n_units=[25, 50, 100], logging_level='INFO', test=False)

And with `cli_parse_args=True`, values can also be input from the command line.

The order of the sources in `settings_customise_sources` determines the priority of the sources.

This allows us to combine inputs from multiple sources, particularly useful for running code in a
capsule that is standalone or part of a pipeline.
"""
import pydantic_settings
import pydantic_settings.sources

class CapsuleParameters(pydantic_settings.BaseSettings):

    n_units: list[int]
    
    logging_level: str | int = 'INFO'
    test: bool = False

    # set the priority of the sources:
    # ignore the function signature for now: concentrate on the return value
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        *args,
        **kwargs,
    ):
        # the order of the sources is what defines the priority:
        # - first source is highest priority
        # - for each field in the class, the first source that contains a value will be used
        return (
            init_settings,
            pydantic_settings.sources.JsonConfigSettingsSource(settings_cls, json_file='parameters.json'),
            pydantic_settings.CliSettingsSource(settings_cls, cli_parse_args=True),
        )

if __name__ == "__main__":
    params = CapsuleParameters()
    print(repr(params))