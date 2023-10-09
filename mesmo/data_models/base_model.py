"""Custom pydantic base model."""

import numpy as np
import pandas as pd
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
