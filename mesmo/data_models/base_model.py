"""Custom pydantic base model."""

import numpy as np
import pandas as pd
import pydantic as pyd
from pydantic.json import timedelta_isoformat


class BaseModel(pyd.BaseModel):
    """Custom base model which overrides pydantic default definitions."""
    model_config = pyd.ConfigDict(json_encoders={
        pd.Timestamp: lambda v: v.strftime("%Y-%m-%dT%H:%M:%S%z"),
        pd.Timedelta: timedelta_isoformat,
        np.bool_: bool,
    })
