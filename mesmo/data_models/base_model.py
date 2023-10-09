"""Custom pydantic base model."""

from typing import Callable

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
        json_encoders={
            pd.Index: lambda v: v.tolist(),
            pd.Timestamp: lambda v: v.strftime("%Y-%m-%dT%H:%M:%S%z"),
            pd.Timedelta: timedelta_isoformat,
            np.bool_: bool,
        },
    )


def check_index(schema: pa.Index) -> Callable[[pd.Index], pd.Index]:
    """Create pydantic validation function for pandas.Index based on given pandera.Index schema."""
    return lambda v: schema.validate(pd.Series(index=v)).index
