"""Model index set data models."""

from typing import Annotated

import pandas as pd
import pandera as pa
import pydantic as pyd

from mesmo.data_models import base_model


class SampleModelIndex(base_model.BaseModel):
    timesteps: Annotated[pd.Index, pyd.AfterValidator(base_model.check_index(pa.Index(pd.Timestamp)))]
    names: Annotated[pd.Index, pyd.AfterValidator(base_model.check_index(pa.Index(str, unique=True)))]


class ElectricGridModelIndex(base_model.BaseModel):
    timesteps: pd.Index
    phases: pd.Index
    node_names: pd.Index
    node_types: pd.Index
    line_names: pd.Index
    transformer_names: pd.Index
    branch_types: pd.Index
    der_names: pd.Index
    der_types: pd.Index
    nodes: pd.Index
    branches: pd.Index
    lines: pd.Index
    transformers: pd.Index
    ders: pd.Index


class ThermalGridModelIndex(base_model.BaseModel):
    timesteps: pd.Index
    node_names: pd.Index
    line_names: pd.Index
    der_names: pd.Index
    der_types: pd.Index
    nodes: pd.Index
    branches: pd.Index
    branch_loops: pd.Index
    ders: pd.Index


class DERModelIndex(base_model.BaseModel):
    der_type: str
    der_name: str
    timesteps: pd.Index
    # Following is only defined for FlexibleDERModels.
    states: pd.Index = pd.Index([])
    storage_states: pd.Index = pd.Index([])
    controls: pd.Index = pd.Index([])
    disturbances: pd.Index = pd.Index([])
    outputs: pd.Index = pd.Index([])


class DERModelSetIndex(base_model.BaseModel):
    timesteps: pd.Index
    ders: pd.Index
    electric_ders: pd.Index
    thermal_ders: pd.Index
    der_names: pd.Index
    fixed_der_names: pd.Index
    flexible_der_names: pd.Index
    states: pd.Index
    controls: pd.Index
    outputs: pd.Index
    storage_states: pd.Index
