"""DRO data interface."""

import numpy as np
import pandas as pd
import pathlib

import mesmo


class DRODataSet(object):

    energy_price: pd.DataFrame
    contingency_reserve_price: pd.DataFrame
    forecast_price_raw: pd.DataFrame
    dro_base_data: pd.DataFrame
    energy_price_deviation: pd.DataFrame
    contingency_reserve_price_deviation: pd.DataFrame

    mean_energy_price: float
    variance_energy_price: float
    mean_contingency_price: float
    variance_contingency_price: float

    def __init__(
        self,
        data_path: pathlib.Path,
    ):

        self.forecast_price_raw = pd.read_csv((data_path / "price_forecast_2021_26_07.csv"))
        self.variance_building_disturbance = pd.read_csv((data_path / "flex_building_disturbance_timeseries_var.csv"))
        self.mean_building_disturbance = pd.read_csv((data_path / "flex_building_disturbance_timeseries_mean.csv"))
        self.max_building_disturbance = pd.read_csv((data_path / "flex_building_disturbance_timeseries_max.csv"))
        self.min_building_disturbance = pd.read_csv((data_path / "flex_building_disturbance_timeseries_min.csv"))
        self.energy_price = self.forecast_price_raw["USEP($/MWh)"] / 100
        self.contingency_reserve_price = self.forecast_price_raw["Contingency($/MWh)"] / 100
        self.dro_base_data = pd.read_csv((data_path / "dro_base_data.csv"))
        self.mean_energy_price = self.energy_price.mean()
        self.variance_energy_price = (self.energy_price - self.mean_energy_price).var()
        self.mean_contingency_price = self.contingency_reserve_price.mean()
        self.variance_contingency_price = (self.contingency_reserve_price - self.mean_contingency_price).var()
        self.energy_price_deviation = self.energy_price - self.mean_energy_price
        self.contingency_reserve_price_deviation = self.contingency_reserve_price - self.mean_contingency_price


class DROAmbiguitySet(object):

    gamma: np.array
    delta_lower_bound: np.array
    delta_upper_bound: np.array
    epsilon_upper_bound: np.array

    def __init__(
        self,
        scenario_name: str,
        optimization_problem_stage_2: mesmo.solutions.OptimizationProblem,
        delta_index: np.ndarray,
        dro_data_set: DRODataSet,
        stage_2_delta_disturbances: pd.DataFrame,
    ):

        # Obtain DER model set.
        der_model_set = mesmo.der_models.DERModelSet(scenario_name)

        # Instantiate parameters.
        self.gamma = pd.Series(0.0, index=delta_index)
        self.delta_lower_bound = pd.Series(0.0, index=delta_index)
        self.delta_upper_bound = pd.Series(0.0, index=delta_index)
        self.epsilon_upper_bound = pd.Series(0.0, index=delta_index)

        # Store variable dimensions for debugging.
        self.delta_variables = optimization_problem_stage_2.variables.loc[delta_index].dropna(axis="columns", how="all")

        # To-DO replace with proper data per time_step
        # Obtain energy price parameters.
        delta_index_selected = mesmo.utils.get_index(
            optimization_problem_stage_2.variables,
            name="price_uncertainty_vector",
            price_type="energy",
            timestep=der_model_set.timesteps,
        )
        self.gamma.loc[delta_index_selected] = 1 * dro_data_set.variance_energy_price
        self.delta_lower_bound.loc[delta_index_selected] = 1 * dro_data_set.energy_price_deviation.min()
        self.delta_upper_bound.loc[delta_index_selected] = 1 * dro_data_set.energy_price_deviation.max()
        self.epsilon_upper_bound.loc[delta_index_selected] = 1 * max(
            dro_data_set.energy_price_deviation.min() ** 2, dro_data_set.energy_price_deviation.max() ** 2
        )

        # Obtain up-reserve price parameters.
        delta_index_selected = mesmo.utils.get_index(
            optimization_problem_stage_2.variables,
            name="price_uncertainty_vector",
            price_type="up_reserve",
            timestep=der_model_set.timesteps,
        )
        self.gamma.loc[delta_index_selected] = dro_data_set.variance_contingency_price
        self.delta_lower_bound.loc[delta_index_selected] = dro_data_set.contingency_reserve_price_deviation.min()
        self.delta_upper_bound.loc[delta_index_selected] = dro_data_set.contingency_reserve_price_deviation.max()
        self.epsilon_upper_bound.loc[delta_index_selected] = max(
            dro_data_set.contingency_reserve_price_deviation.min() ** 2,
            dro_data_set.contingency_reserve_price_deviation.max() ** 2,
        )

        # Obtain down-reserve price parameters.
        delta_index_selected = mesmo.utils.get_index(
            optimization_problem_stage_2.variables,
            name="price_uncertainty_vector",
            price_type="down_reserve",
            timestep=der_model_set.timesteps,
        )
        self.gamma.loc[delta_index_selected] = dro_data_set.variance_contingency_price
        self.delta_lower_bound.loc[delta_index_selected] = dro_data_set.contingency_reserve_price_deviation.min()
        self.delta_upper_bound.loc[delta_index_selected] = dro_data_set.contingency_reserve_price_deviation.max()
        self.epsilon_upper_bound.loc[delta_index_selected] = max(
            dro_data_set.contingency_reserve_price_deviation.min() ** 2,
            dro_data_set.contingency_reserve_price_deviation.max() ** 2,
        )

        # Obtain DER uncertainty parameters.
        delta_index_selected = mesmo.utils.get_index(
            optimization_problem_stage_2.variables,
            name="disturbance_uncertainty_vector",
            timestep=der_model_set.timesteps,
            disturbance=der_model_set.disturbances,
        )
        self.gamma[delta_index_selected] = 1
        self.delta_lower_bound.loc[delta_index_selected] = -4
        self.delta_upper_bound.loc[delta_index_selected] = 4
        self.epsilon_upper_bound.loc[delta_index_selected] = 16


def main():
    pass


if __name__ == "__main__":
    main()
