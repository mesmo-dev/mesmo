"""Results data models."""
from typing import Optional
import pandas as pd

from mesmo.data_models import base_model, model_index
from mesmo import utils


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


class Results(base_model.BaseModel):
    pass
