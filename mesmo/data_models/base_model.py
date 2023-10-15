"""Custom pydantic base model."""

from typing import Annotated, Optional

import numpy as np
import pandas as pd
import pandera as pa
import pydantic as pyd
from pydantic.json import timedelta_isoformat


class BaseModel(pyd.BaseModel):
    """Custom base model which overrides pydantic default definitions."""

    model_config = pyd.ConfigDict(
        extra="allow",  # TODO: In the long term, this should be set to false for more predictable model content
        arbitrary_types_allowed=True,
        json_encoders={  # TODO: Review how to correctly set custom encoders for pydantic v2
            pd.Index: lambda v: v.tolist(),
            pd.Timestamp: lambda v: v.strftime("%Y-%m-%dT%H:%M:%S%z"),
            pd.Timedelta: timedelta_isoformat,
            np.bool_: bool,
        },
    )


def get_index_annotation(dtype: type[pd.Timestamp | int | str], optional: bool = False) -> type[pd.Index]:
    """Get pydantic-compatible type annotation for pandas.Index."""
    return Annotated[
        pd.Index,
        pyd.PlainValidator(lambda v: pa.Index(dtype, coerce=True).validate(pd.DataFrame(index=v)).index),
        pyd.Field(default_factory=lambda: pd.Index([], dtype=dtype)) if optional else pyd.Field(),
    ]


def _coerce_to_multiindex(v: pd.MultiIndex | list, names: list[str]) -> pd.MultiIndex:
    if isinstance(v, pd.MultiIndex):
        return v
    if isinstance(v, list):
        return pd.MultiIndex.from_tuples(v, names=names)
    raise NotImplementedError(f"Unable to coerce given value of type {type(v)} to pandas MultiIndex")


def get_multiindex_annotation(dtypes: dict[str, pd.Timestamp | int | str]) -> type[pd.MultiIndex]:
    """Get pydantic-compatible type annotation for pandas.MultiIndex."""
    indexes = [pa.Index(dtype, name=name, coerce=True) for name, dtype in dtypes.items()]
    return Annotated[
        pd.MultiIndex,
        pyd.PlainValidator(
            lambda v: pa.MultiIndex(indexes)
            .validate(pd.DataFrame(index=_coerce_to_multiindex(v, names=dtypes.keys())))
            .index
        ),
    ]


def _coerce_to_dataframe(v: pd.DataFrame | str) -> pd.DataFrame:
    if isinstance(v, pd.DataFrame):
        return v
    if isinstance(v, dict):
        df = pd.DataFrame.from_dict(v, orient="tight")
        if df.index.name == "timestep":
            df.index = pd.DatetimeIndex(df.index)
        return df
    raise NotImplementedError(f"Unable to coerce given value of type {type(v)} to pandas DataFrame")


def get_dataframe_annotation(dtype: type, column_index_levels: Optional[int] = None) -> type[pd.DataFrame]:
    """Get pydantic-compatible type annotation for pandas.DataFrame."""
    columns = ".*" if column_index_levels is None else (".*",) * column_index_levels
    return Annotated[
        pd.DataFrame,
        pyd.PlainValidator(
            lambda v: pa.DataFrameSchema({columns: pa.Column(dtype, coerce=True, regex=True)}).validate(
                _coerce_to_dataframe(v)
            )
        ),
        pyd.PlainSerializer(lambda v: v.to_dict(orient="tight")),
    ]
