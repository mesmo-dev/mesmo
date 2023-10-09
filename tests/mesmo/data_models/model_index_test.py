import pandas as pd
import pytest

from mesmo.data_models import model_index


class TestSampleModelIndex:
    def test_init(self):
        timesteps = pd.Index([pd.Timestamp("now"), pd.Timestamp("now")])
        names = pd.Index(["a", "b", "c"])

        model_index.SampleModelIndex(timesteps=timesteps, names=names)
