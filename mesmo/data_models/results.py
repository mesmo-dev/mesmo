"""Results data models."""
from typing import Annotated, Optional

import pandas as pd

from mesmo import utils
from mesmo.data_models import base_model, model_index


class ElectricGridDEROperationResults(utils.ResultsBase):
    der_active_power_vector: pd.DataFrame
    der_active_power_vector_per_unit: pd.DataFrame
    der_reactive_power_vector: pd.DataFrame
    der_reactive_power_vector_per_unit: pd.DataFrame


class ElectricGridOperationResults(ElectricGridDEROperationResults):
    electric_grid_model_index: Optional[model_index.ElectricGridModelIndex]
    node_voltage_magnitude_vector: pd.DataFrame
    node_voltage_magnitude_vector_per_unit: pd.DataFrame
    node_voltage_angle_vector: pd.DataFrame
    branch_power_magnitude_vector_1: pd.DataFrame
    branch_power_magnitude_vector_1_per_unit: pd.DataFrame
    branch_active_power_vector_1: pd.DataFrame
    branch_active_power_vector_1_per_unit: pd.DataFrame
    branch_reactive_power_vector_1: pd.DataFrame
    branch_reactive_power_vector_1_per_unit: pd.DataFrame
    branch_power_magnitude_vector_2: pd.DataFrame
    branch_power_magnitude_vector_2_per_unit: pd.DataFrame
    branch_active_power_vector_2: pd.DataFrame
    branch_active_power_vector_2_per_unit: pd.DataFrame
    branch_reactive_power_vector_2: pd.DataFrame
    branch_reactive_power_vector_2_per_unit: pd.DataFrame
    loss_active: pd.DataFrame
    loss_reactive: pd.DataFrame


class ElectricGridDLMPResults(utils.ResultsBase):
    electric_grid_energy_dlmp_node_active_power: pd.DataFrame
    electric_grid_voltage_dlmp_node_active_power: pd.DataFrame
    electric_grid_congestion_dlmp_node_active_power: pd.DataFrame
    electric_grid_loss_dlmp_node_active_power: pd.DataFrame
    electric_grid_total_dlmp_node_active_power: pd.DataFrame
    electric_grid_voltage_dlmp_node_reactive_power: pd.DataFrame
    electric_grid_congestion_dlmp_node_reactive_power: pd.DataFrame
    electric_grid_loss_dlmp_node_reactive_power: pd.DataFrame
    electric_grid_energy_dlmp_node_reactive_power: pd.DataFrame
    electric_grid_total_dlmp_node_reactive_power: pd.DataFrame
    electric_grid_energy_dlmp_der_active_power: pd.DataFrame
    electric_grid_voltage_dlmp_der_active_power: pd.DataFrame
    electric_grid_congestion_dlmp_der_active_power: pd.DataFrame
    electric_grid_loss_dlmp_der_active_power: pd.DataFrame
    electric_grid_total_dlmp_der_active_power: pd.DataFrame
    electric_grid_voltage_dlmp_der_reactive_power: pd.DataFrame
    electric_grid_congestion_dlmp_der_reactive_power: pd.DataFrame
    electric_grid_loss_dlmp_der_reactive_power: pd.DataFrame
    electric_grid_energy_dlmp_der_reactive_power: pd.DataFrame
    electric_grid_total_dlmp_der_reactive_power: pd.DataFrame
    electric_grid_total_dlmp_price_timeseries: pd.DataFrame


class ThermalGridDEROperationResults(utils.ResultsBase):
    der_thermal_power_vector: pd.DataFrame
    der_thermal_power_vector_per_unit: pd.DataFrame


class ThermalGridOperationResults(ThermalGridDEROperationResults):
    thermal_grid_model_index: Optional[model_index.ThermalGridModelIndex]
    node_head_vector: pd.DataFrame
    node_head_vector_per_unit: pd.DataFrame
    branch_flow_vector: pd.DataFrame
    branch_flow_vector_per_unit: pd.DataFrame
    pump_power: pd.DataFrame


class ThermalGridDLMPResults(utils.ResultsBase):
    thermal_grid_energy_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_head_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_congestion_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_pump_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_total_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_energy_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_head_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_congestion_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_pump_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_total_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_total_dlmp_price_timeseries: pd.DataFrame


class DERModelOperationResults(utils.ResultsBase):
    # TODO: Drop support for this class
    der_model_index: Optional[model_index.DERModelIndex]
    state_vector: pd.DataFrame
    control_vector: pd.DataFrame
    output_vector: pd.DataFrame


class DERModelSetOperationResults(ElectricGridDEROperationResults):
    der_model_set_index: Optional[model_index.DERModelSetIndex]
    state_vector: pd.DataFrame
    control_vector: pd.DataFrame
    output_vector: pd.DataFrame
    # TODO: Add output constraint and disturbance timeseries.
    der_thermal_power_vector: pd.DataFrame
    der_thermal_power_vector_per_unit: pd.DataFrame


class ElectricGridOperationRunResults(base_model.BaseModel):
    der_active_power_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    der_active_power_vector_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    der_reactive_power_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    der_reactive_power_vector_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    node_voltage_magnitude_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    node_voltage_magnitude_vector_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    node_voltage_angle_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    branch_power_magnitude_vector_1: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    branch_power_magnitude_vector_1_per_unit: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    branch_active_power_vector_1: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    branch_active_power_vector_1_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    branch_reactive_power_vector_1: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    branch_reactive_power_vector_1_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    branch_power_magnitude_vector_2: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    branch_power_magnitude_vector_2_per_unit: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    branch_active_power_vector_2: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    branch_active_power_vector_2_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    branch_reactive_power_vector_2: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    branch_reactive_power_vector_2_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=3)]
    loss_active: Optional[base_model.get_dataframe_annotation(float)]
    loss_reactive: Optional[base_model.get_dataframe_annotation(float)]


class ThermalGridOperationRunResults(base_model.BaseModel):
    der_thermal_power_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    der_thermal_power_vector_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    node_head_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    node_head_vector_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    branch_flow_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    branch_flow_vector_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    pump_power: Optional[base_model.get_dataframe_annotation(float)]


class DERModelSetOperationRunResults(base_model.BaseModel):
    der_active_power_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    der_active_power_vector_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    der_reactive_power_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    der_reactive_power_vector_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    der_thermal_power_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    der_thermal_power_vector_per_unit: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    state_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    control_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    output_vector: Optional[base_model.get_dataframe_annotation(float, column_index_levels=2)]
    # TODO: Add output constraint and disturbance timeseries.


class ElectricGridDLMPRunResults(base_model.BaseModel):
    electric_grid_energy_dlmp_node_active_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    electric_grid_voltage_dlmp_node_active_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    electric_grid_congestion_dlmp_node_active_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    electric_grid_loss_dlmp_node_active_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    electric_grid_total_dlmp_node_active_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    electric_grid_voltage_dlmp_node_reactive_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    electric_grid_congestion_dlmp_node_reactive_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    electric_grid_loss_dlmp_node_reactive_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    electric_grid_energy_dlmp_node_reactive_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    electric_grid_total_dlmp_node_reactive_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]
    electric_grid_energy_dlmp_der_active_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    electric_grid_voltage_dlmp_der_active_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    electric_grid_congestion_dlmp_der_active_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    electric_grid_loss_dlmp_der_active_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    electric_grid_total_dlmp_der_active_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    electric_grid_voltage_dlmp_der_reactive_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    electric_grid_congestion_dlmp_der_reactive_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    electric_grid_loss_dlmp_der_reactive_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    electric_grid_energy_dlmp_der_reactive_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    electric_grid_total_dlmp_der_reactive_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    electric_grid_total_dlmp_price_timeseries: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]


class ThermalGridDLMPRunResults(base_model.BaseModel):
    thermal_grid_energy_dlmp_node_thermal_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    thermal_grid_head_dlmp_node_thermal_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    thermal_grid_congestion_dlmp_node_thermal_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    thermal_grid_pump_dlmp_node_thermal_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    thermal_grid_total_dlmp_node_thermal_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    thermal_grid_energy_dlmp_der_thermal_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    thermal_grid_head_dlmp_der_thermal_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    thermal_grid_congestion_dlmp_der_thermal_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    thermal_grid_pump_dlmp_der_thermal_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    thermal_grid_total_dlmp_der_thermal_power: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=2)
    ]
    thermal_grid_total_dlmp_price_timeseries: Optional[
        base_model.get_dataframe_annotation(float, column_index_levels=3)
    ]


class RunResults(base_model.BaseModel):
    electric_grid_model_index: Optional[model_index.ElectricGridModelIndex]
    thermal_grid_model_index: Optional[model_index.ThermalGridModelIndex]
    der_model_set_index: model_index.DERModelSetIndex
    electric_grid_operation_results: ElectricGridOperationRunResults
    thermal_grid_operation_results: ThermalGridOperationRunResults
    der_operation_results: DERModelSetOperationRunResults
    electric_grid_dlmp_results: ElectricGridDLMPRunResults
    thermal_grid_dlmp_results: ThermalGridDLMPRunResults
