"""
Normally, the values passed to init are used, combined with the default values:
    >>> CapsuleParameters(n_units_list=[0, 1])
    CapsuleParameters(n_units_list=[0, 1], logging_level='INFO', test=False)

With a `parameters.json` file containing `{"n_units_list": [25, 50, 100]}`, we don't need to pass
the `n_units_list` parameter to the constructor:
    >>> CapsuleParameters()
    CapsuleParameters(n_units_list=[25, 50, 100], logging_level='INFO', test=False)

And with `cli_parse_args=True`, values can also be input from the command line.

The order of the sources in `settings_customise_sources` determines the priority of the sources.

This allows us to combine inputs from multiple sources, particularly useful for running code in a
capsule that is standalone or part of a pipeline.
"""

from typing import Annotated, Literal
import pydantic
import pydantic_settings
import pydantic_settings.sources
import polars as pl

from pydantic.functional_serializers import PlainSerializer

Expr = Annotated[
    pl.Expr, PlainSerializer(lambda expr: expr.meta.serialize(format='json'), return_type=str)
]


class CapsuleParameters(pydantic_settings.BaseSettings):

    n_units: int
    unit_criteria: Literal['strict', 'medium', None] = 'strict'
    logging_level: str | int = pydantic.Field('INFO', exclude=True)
    test: bool =  pydantic.Field(False, exclude=True)
    
    drift_col: Expr = pydantic.Field(default_factory=lambda: pl.col('activity_drift'), repr=False)
    
    @pydantic.computed_field(repr=False)
    @property
    def unit_filter(self) -> Expr:
        return {
            'strict': (pl.col('activity_drift') < 0.1) & (pl.col('isi_violations_ratio') < 0.5),
            'medium': (pl.col('activity_drift') < 0.2) & (pl.col('isi_violations_ratio') < 0.7),
            None: pl.lit(True),
        }[self.unit_criteria]

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
            pydantic_settings.sources.JsonConfigSettingsSource(settings_cls, json_file='params.json'),
            pydantic_settings.CliSettingsSource(settings_cls, cli_parse_args=True),
        )
    
if __name__ == "__main__":
    params = CapsuleParameters()
    
    print(params.model_dump())
    # {'n_units': 25, 'unit_criteria': 'medium', 'drift_col': '{"Column":"activity_drift"}', 'unit_filter': '{"BinaryExpr":{"left":{"BinaryExpr":{"left":{"Column":"activity_drift"},"op":"Lt","right":{"Literal":{"Float":0.2}}}},"op":"And","right":{"BinaryExpr":{"left":{"Column":"isi_violations_ratio"},"op":"Lt","right":{"Literal":{"Float":0.7}}}}}}'}
    
    print(repr(params))
    # CapsuleParameters(n_units=25, unit_criteria='medium', logging_level='INFO', test=False)
