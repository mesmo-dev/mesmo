from contextlib import nullcontext as does_not_raise

import pandas as pd
import pandera as pa
import pytest

from mesmo.data_models import base_model


class SampleModelIndex(base_model.BaseModel):
    timesteps: base_model.get_index_annotation(pd.Timestamp)
    names: base_model.get_multiindex_annotation({"xxx": str, "yyy": int})


class TestModeIndex:
    @pytest.mark.parametrize(
        "timesteps,names,expectation",
        [
            # happy case
            (
                pd.Index([pd.Timestamp("now"), pd.Timestamp("now")]),
                pd.MultiIndex.from_arrays([["a", "b", "c"], [1, 2, 3]], names=["xxx", "yyy"]),
                does_not_raise(),
            ),
            # validation error: int instead of str in multiindex
            (
                pd.Index([pd.Timestamp("now"), pd.Timestamp("now")]),
                pd.MultiIndex.from_arrays([[1, 2, 3], ["a", "b", "c"]], names=["xxx", "yyy"]),
                pytest.raises(pa.errors.SchemaError),
            ),
        ],
    )
    def test_index_validator(self, timesteps, names, expectation):
        with expectation:
            SampleModelIndex(timesteps=timesteps, names=names)

    @pytest.fixture
    def sample_model_index(self):
        return SampleModelIndex(
            timesteps=pd.Index(
                [pd.Timestamp("2023-10-15T18:41:50.047742"), pd.Timestamp("2023-10-15T18:41:50.047744")]
            ),
            names=pd.MultiIndex.from_arrays([["a", "b", "c"], [1, 2, 3]], names=["xxx", "yyy"]),
        )

    def test_serialization(self, sample_model_index: SampleModelIndex):
        want = (
            '{"timesteps":["2023-10-15T18:41:50.047742","2023-10-15T18:41:50.047744"],'
            '"names":[["a",1],["b",2],["c",3]]}'
        )

        got = sample_model_index.model_dump_json()

        assert got == want

    def test_serialization_deserialization(self, sample_model_index):
        want = sample_model_index

        got = SampleModelIndex.model_validate_json(want.model_dump_json())

        assert got.model_dump_json() == want.model_dump_json()


class SampleModelDataFrame(base_model.BaseModel):
    data_single: base_model.get_dataframe_annotation(float)
    data_multi: base_model.get_dataframe_annotation(float, column_index_levels=2)


class TestModelDataFrame:
    @pytest.mark.parametrize(
        "data_single,data_multi,expectation",
        [
            # happy case
            (
                pd.DataFrame(
                    [[1.0, 2.0], [3.0, 4.0]],
                    columns=["aaa", "bbb"],
                    index=["xxx", "yyy"],
                ),
                pd.DataFrame(
                    [[1.0, 2.0], [3.0, 4.0]],
                    columns=pd.MultiIndex.from_tuples([("aa", 1), ("bb", 2)]),
                    index=["xxx", "yyy"],
                ),
                does_not_raise(),
            ),
            # validation error: str instead of float in data
            (
                pd.DataFrame(
                    [[1.0, 2.0], [3.0, 4.0]],
                    columns=["aaa", "bbb"],
                    index=["xxx", "yyy"],
                ),
                pd.DataFrame(
                    [["a", "b"], [3.0, 4.0]],
                    columns=pd.MultiIndex.from_tuples([("aa", 1), ("bb", 2)]),
                    index=["xxx", "yyy"],
                ),
                pytest.raises(pa.errors.SchemaError),
            ),
            # validation error: NaN values in data
            (
                pd.DataFrame(
                    [[1.0, 2.0], [3.0, 4.0]],
                    columns=["aaa", "bbb"],
                    index=["xxx", "yyy"],
                ),
                pd.DataFrame(
                    [[float("nan"), float("nan")], [3.0, 4.0]],
                    columns=pd.MultiIndex.from_tuples([("aa", 1), ("bb", 2)]),
                    index=["xxx", "yyy"],
                ),
                pytest.raises(pa.errors.SchemaError),
            ),
            # validation error: Wrong number of multi-index levels
            (
                pd.DataFrame(
                    [[1.0, 2.0], [3.0, 4.0]],
                    columns=["aaa", "bbb"],
                    index=["xxx", "yyy"],
                ),
                pd.DataFrame(
                    [[1.0, 2.0], [3.0, 4.0]],
                    columns=pd.MultiIndex.from_tuples([("aa", 1, "xx"), ("bb", 2, "yy")]),
                    index=["xxx", "yyy"],
                ),
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_validation(self, data_single, data_multi, expectation):
        with expectation:
            SampleModelDataFrame(data_single=data_single, data_multi=data_multi)

    @pytest.fixture
    def sample_model_dataframe(self):
        return SampleModelDataFrame(
            data_single=pd.DataFrame(
                [[1.0, 2.0], [3.0, 4.0]],
                columns=["aaa", "bbb"],
                index=pd.Index(
                    [pd.Timestamp("2023-10-15T18:41:50.047742"), pd.Timestamp("2023-10-15T18:41:50.047744")],
                    name="timestep",
                ),
            ),
            data_multi=pd.DataFrame(
                [[1.0, 2.0], [3.0, 4.0]],
                columns=pd.MultiIndex.from_tuples([("aa", 1), ("bb", 2)], names=["some", "thing"]),
                index=["xxx", "yyy"],
            ),
        )

    def test_serialization(self, sample_model_dataframe: SampleModelDataFrame):
        want = (
            '{"data_single":{"index":["2023-10-15T18:41:50.047742","2023-10-15T18:41:50.047744"],'
            '"columns":["aaa","bbb"],"data":[[1.0,2.0],[3.0,4.0]],'
            '"index_names":["timestep"],"column_names":[null]},"data_multi":{"index":["xxx","yyy"],'
            '"columns":[["aa",1],["bb",2]],"data":[[1.0,2.0],[3.0,4.0]],"index_names":[null],'
            '"column_names":["some","thing"]}}'
        )

        got = sample_model_dataframe.model_dump_json()

        assert got == want

    def test_serialization_deserialization(self, sample_model_dataframe: SampleModelDataFrame):
        want = sample_model_dataframe

        got = SampleModelDataFrame.model_validate_json(sample_model_dataframe.model_dump_json())

        assert got.model_dump_json() == want.model_dump_json()
