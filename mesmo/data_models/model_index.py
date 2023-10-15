"""Model index set data models."""

import pandas as pd

from mesmo.data_models import base_model


class ElectricGridModelIndex(base_model.BaseModel):
    timesteps: base_model.get_index_annotation(pd.Timestamp)
    phases: base_model.get_index_annotation(int)
    node_names: base_model.get_index_annotation(str)
    node_types: base_model.get_index_annotation(str)
    line_names: base_model.get_index_annotation(str)
    transformer_names: base_model.get_index_annotation(str)
    branch_types: base_model.get_index_annotation(str)
    der_names: base_model.get_index_annotation(str)
    der_types: base_model.get_index_annotation(str)
    nodes: base_model.get_multiindex_annotation({"node_type": str, "node_name": str, "phase": int})
    branches: base_model.get_multiindex_annotation({"branch_type": str, "branch_name": str, "phase": int})
    lines: base_model.get_multiindex_annotation({"branch_type": str, "branch_name": str, "phase": int})
    transformers: base_model.get_multiindex_annotation({"branch_type": str, "branch_name": str, "phase": int})
    ders: base_model.get_multiindex_annotation({"der_type": str, "der_name": str})


class ThermalGridModelIndex(base_model.BaseModel):
    timesteps: base_model.get_index_annotation(pd.Timestamp)
    node_names: base_model.get_index_annotation(str)
    line_names: base_model.get_index_annotation(str)
    der_names: base_model.get_index_annotation(str)
    der_types: base_model.get_index_annotation(str)
    nodes: base_model.get_multiindex_annotation({"node_type": str, "node_name": str})
    branches: base_model.get_multiindex_annotation({"branch_name": str, "loop_type": str})
    branch_loops: base_model.get_multiindex_annotation({"loop_id": str, "branch_name": str})
    ders: base_model.get_multiindex_annotation({"der_type": str, "der_name": str})


class DERModelIndex(base_model.BaseModel):
    der_type: str
    der_name: str
    timesteps: base_model.get_index_annotation(pd.Timestamp)
    # The following fields are only defined for FlexibleDERModels.
    states: base_model.get_index_annotation(str, optional=True)
    storage_states: base_model.get_index_annotation(str, optional=True)
    controls: base_model.get_index_annotation(str, optional=True)
    disturbances: base_model.get_index_annotation(str, optional=True)
    outputs: base_model.get_index_annotation(str, optional=True)


class DERModelSetIndex(base_model.BaseModel):
    timesteps: base_model.get_index_annotation(pd.Timestamp)
    ders: base_model.get_multiindex_annotation({"der_type": str, "der_name": str})
    electric_ders: base_model.get_multiindex_annotation({"der_type": str, "der_name": str})
    thermal_ders: base_model.get_multiindex_annotation({"der_type": str, "der_name": str})
    der_names: base_model.get_index_annotation(str)
    fixed_der_names: base_model.get_index_annotation(str)
    flexible_der_names: base_model.get_index_annotation(str)
    states: base_model.get_multiindex_annotation({"der_name": str, "state": str})
    controls: base_model.get_multiindex_annotation({"der_name": str, "control": str})
    outputs: base_model.get_multiindex_annotation({"der_name": str, "output": str})
    storage_states: base_model.get_multiindex_annotation({"der_name": str, "state": str})
