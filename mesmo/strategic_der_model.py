import numpy as np
import scipy.sparse as sp
import pandas as pd
import typing

from cv2.gapi import op
from holoviews import opts

import mesmo
from mesmo.der_models import DERModelSet
from mesmo.electric_grid_models import LinearElectricGridModelSet, ElectricGridModelDefault, PowerFlowSolutionFixedPoint


class StrategicElectricGridDLMPResults(mesmo.utils.ResultsBase):
    strategic_electric_grid_energy_dlmp_node_active_power: pd.DataFrame
    strategic_electric_grid_voltage_dlmp_node_active_power: pd.DataFrame
    strategic_electric_grid_congestion_dlmp_node_active_power: pd.DataFrame
    strategic_electric_grid_loss_dlmp_node_active_power: pd.DataFrame
    strategic_electric_grid_total_dlmp_node_active_power: pd.DataFrame
    strategic_electric_grid_voltage_dlmp_node_reactive_power: pd.DataFrame
    strategic_electric_grid_congestion_dlmp_node_reactive_power: pd.DataFrame
    strategic_electric_grid_loss_dlmp_node_reactive_power: pd.DataFrame
    strategic_electric_grid_energy_dlmp_node_reactive_power: pd.DataFrame
    strategic_electric_grid_total_dlmp_node_reactive_power: pd.DataFrame
    strategic_electric_grid_energy_dlmp_der_active_power: pd.DataFrame
    strategic_electric_grid_voltage_dlmp_der_active_power: pd.DataFrame
    strategic_electric_grid_congestion_dlmp_der_active_power: pd.DataFrame
    strategic_electric_grid_loss_dlmp_der_active_power: pd.DataFrame
    strategic_electric_grid_total_dlmp_der_active_power: pd.DataFrame
    strategic_electric_grid_voltage_dlmp_der_reactive_power: pd.DataFrame
    strategic_electric_grid_congestion_dlmp_der_reactive_power: pd.DataFrame
    strategic_electric_grid_loss_dlmp_der_reactive_power: pd.DataFrame
    strategic_electric_grid_energy_dlmp_der_reactive_power: pd.DataFrame
    strategic_electric_grid_total_dlmp_der_reactive_power: pd.DataFrame
    strategic_electric_grid_total_dlmp_price_timeseries: pd.DataFrame
    der_marginal_price_offers: pd.DataFrame
    strategic_der_marginal_price_offers: pd.DataFrame


class StrategicMarket(object):
    def __init__(
            self,
            scenario_name: str,
            strategic_der: str
    ):
        electric_grid_model = ElectricGridModelDefault(scenario_name)
        power_flow_solution = PowerFlowSolutionFixedPoint(electric_grid_model)
        self.linear_electric_grid_model_set = LinearElectricGridModelSet(electric_grid_model, power_flow_solution)
        self.der_model_set = DERModelSet(scenario_name)

        if len(self.der_model_set.electric_ders) > 0:
            self.timestep_interval_hours = (self.der_model_set.timesteps[1] -
                                            self.der_model_set.timesteps[0]) / pd.Timedelta('1h')


    def electric_grid_constraints(self,
                                  optimization_problem: mesmo.utils.OptimizationProblem,
                                  ):




