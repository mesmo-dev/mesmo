"""DRO problem stage 2."""

import numpy as np
import pandas as pd
import pathlib
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp

import mesmo
from dro_data_interface import DRODataSet
from dro_stage_1 import Stage1


class Stage2(object):
    def __init__(self, scenario_name, dro_data_set, enable_electric_grid_model=True):

        mesmo.utils.logger.info("Initializing stage 2 problem...")

        # Settings.
        self.scenarios_stage_1 = ["no_reserve_stage_1", "up_reserve_stage_1", "down_reserve_stage_1"]
        self.scenarios_stage_2 = ["no_reserve_stage_2", "up_reserve_stage_2", "down_reserve_stage_2"]

        # Obtain price data object.
        price_data = mesmo.data_interface.PriceData(scenario_name)

        # Obtain DER & grid model objects.
        self.der_model_set = mesmo.der_models.DERModelSet(scenario_name)
        linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(scenario_name)

        # Manipulate building model to avoid over-consumption for up-reserves.
        for der_name, der_model in self.der_model_set.der_models.items():
            if isinstance(der_model, mesmo.der_models.FlexibleBuildingModel):
                der_model.output_maximum_timeseries.loc[
                    :, der_model.output_maximum_timeseries.columns.str.contains("_heat_")
                ] = 0.0
                der_model.output_maximum_timeseries.iloc[
                    -1, der_model.output_maximum_timeseries.columns.str.contains("_cool_")
                ] = 0.0
                der_model.output_minimum_timeseries.iloc[
                    -1, der_model.output_minimum_timeseries.columns.str.contains("_air_flow")
                ] = 0.0

        # Select disturbances to be represented in delta and define mapping matrix.
        for der_name, der_model in self.der_model_set.flexible_der_models.items():
            if isinstance(der_model, mesmo.der_models.FlexibleBuildingModel):
                der_model.delta_disturbances = pd.Index(["ambient_air_temperature", "irradiation"], name="delta_name")
                der_model.disturbance_delta_mapping_matrix = pd.DataFrame(
                    0.0, index=der_model.disturbances, columns=der_model.delta_disturbances
                )
                der_model.disturbance_delta_mapping_matrix.loc[
                    [
                        "irradiation_east",
                        "irradiation_horizontal",
                        "irradiation_north",
                        "irradiation_south",
                        "irradiation_west",
                    ],
                    "irradiation",
                ] = 1.0
                der_model.disturbance_delta_mapping_matrix.loc[
                    "ambient_air_temperature", "ambient_air_temperature"
                ] = 1.0
            elif isinstance(der_model, (mesmo.der_models.FlexibleLoadModel, mesmo.der_models.FlexibleGeneratorModel)):
                der_model.delta_disturbances = der_model.disturbances.rename("delta_name")
                der_model.disturbance_delta_mapping_matrix = pd.DataFrame(
                    (
                        np.eye(len(der_model.disturbances))
                        # Normalization factor, because disturbance value in state space model is normalized.
                        / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)
                    ),
                    index=der_model.disturbances,
                    columns=der_model.delta_disturbances,
                )
                # Raise warning if active_power_nominal is zero.
                if der_model.active_power_nominal != 0.0:
                    mesmo.utils.logger.warning(
                        f"Possible DER definition error: `active_power_nominal` of DER {der_name} is zero."
                    )
            else:
                der_model.delta_disturbances = der_model.disturbances.rename("delta_name")
                der_model.disturbance_delta_mapping_matrix = pd.DataFrame(
                    np.eye(len(der_model.disturbances)),
                    index=der_model.disturbances,
                    columns=der_model.delta_disturbances,
                )

        # Collect delta disturbance entries.
        self.delta_disturbances = pd.MultiIndex.from_tuples(
            [
                (der_name, delta_disturbance)
                for der_name in self.der_model_set.flexible_der_names
                for delta_disturbance in self.der_model_set.der_models[der_name].delta_disturbances
            ]
        )

        # Instantiate optimization problem.
        self.optimization_problem = mesmo.solutions.OptimizationProblem()

        # Re-define stage 1 variables.
        # - This is needed to retain the appropriate dimensions for the combined problem.
        self.der_model_set.define_optimization_variables(self.optimization_problem, scenarios=self.scenarios_stage_1)
        if enable_electric_grid_model:
            linear_electric_grid_model_set.define_optimization_variables(
                self.optimization_problem, scenarios=self.scenarios_stage_1
            )
        self.optimization_problem.define_variable(
            "energy_stage_1", timestep=self.der_model_set.timesteps, scenario=["no_reserve_stage_1"]
        )
        self.optimization_problem.define_variable(
            "up_reserve_stage_1", timestep=self.der_model_set.timesteps, scenario=["up_reserve_stage_1"]
        )
        self.optimization_problem.define_variable(
            "down_reserve_stage_1", timestep=self.der_model_set.timesteps, scenario=["down_reserve_stage_1"]
        )

        # Define DER problem.
        # - Note that custom DER constraints are defined below.
        self.der_model_set.define_optimization_variables(self.optimization_problem, scenarios=self.scenarios_stage_2)
        self.der_model_set.define_optimization_parameters(
            self.optimization_problem, price_data, scenarios=self.scenarios_stage_2
        )

        # Define electric grid problem.
        if enable_electric_grid_model:
            linear_electric_grid_model_set.define_optimization_variables(
                self.optimization_problem, scenarios=self.scenarios_stage_2
            )
            linear_electric_grid_model_set.define_optimization_parameters(
                self.optimization_problem, price_data, scenarios=self.scenarios_stage_2
            )
            linear_electric_grid_model_set.define_optimization_constraints(
                self.optimization_problem, scenarios=self.scenarios_stage_2
            )

        # Define additional variables.
        self.optimization_problem.define_variable(
            "energy_deviation_up_stage_2", scenario=self.scenarios_stage_2, timestep=self.der_model_set.timesteps
        )
        self.optimization_problem.define_variable(
            "energy_deviation_down_stage_2", scenario=self.scenarios_stage_2, timestep=self.der_model_set.timesteps
        )
        self.optimization_problem.define_variable(
            "price_uncertainty_vector",
            scenario=["delta"],
            price_type=["energy", "up_reserve", "down_reserve"],
            timestep=self.der_model_set.timesteps,
        )
        self.optimization_problem.define_variable(
            "disturbance_uncertainty_vector",
            scenario=["delta"],
            disturbance=self.delta_disturbances,
            timestep=self.der_model_set.timesteps,
        )

        # Define additional parameters.
        self.optimization_problem.define_parameter(
            "delta_disturbance_matrix",
            sp.block_diag(
                [
                    (der_model.disturbance_matrix @ der_model.disturbance_delta_mapping_matrix).values
                    for der_name, der_model in self.der_model_set.flexible_der_models.items()
                ]
            ),
        )
        self.optimization_problem.define_parameter(
            "delta_disturbance_output_matrix",
            sp.block_diag(
                [
                    (der_model.disturbance_output_matrix @ der_model.disturbance_delta_mapping_matrix).values
                    for der_name, der_model in self.der_model_set.flexible_der_models.items()
                ]
            ),
        )

        # Power balance equations.
        self.optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="energy_stage_1", timestep=self.der_model_set.timesteps)),
            (
                "variable",
                1.0,
                dict(
                    name="energy_deviation_up_stage_2",
                    scenario=["no_reserve_stage_2"],
                    timestep=self.der_model_set.timesteps,
                ),
            ),
            (
                "variable",
                -1.0,
                dict(
                    name="energy_deviation_down_stage_2",
                    scenario=["no_reserve_stage_2"],
                    timestep=self.der_model_set.timesteps,
                ),
            ),
            "==",
            (
                "variable",
                (
                    -1.0
                    * np.array([np.real(linear_electric_grid_model_set.electric_grid_model.der_power_vector_reference)])
                ),
                dict(
                    name="der_active_power_vector",
                    scenario=["no_reserve_stage_2"],
                    timestep=self.der_model_set.timesteps,
                    der=linear_electric_grid_model_set.electric_grid_model.ders,
                ),
            ),
            broadcast="timestep",
        )
        self.optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="energy_stage_1", timestep=self.der_model_set.timesteps)),
            (
                "variable",
                1.0,
                dict(
                    name="energy_deviation_up_stage_2",
                    scenario=["up_reserve_stage_2"],
                    timestep=self.der_model_set.timesteps,
                ),
            ),
            (
                "variable",
                -1.0,
                dict(
                    name="energy_deviation_down_stage_2",
                    scenario=["up_reserve_stage_2"],
                    timestep=self.der_model_set.timesteps,
                ),
            ),
            ("variable", 1.0, dict(name="up_reserve_stage_1", timestep=self.der_model_set.timesteps)),
            "==",
            (
                "variable",
                (
                    -1.0
                    * np.array([np.real(linear_electric_grid_model_set.electric_grid_model.der_power_vector_reference)])
                ),
                dict(
                    name="der_active_power_vector",
                    scenario=["up_reserve_stage_2"],
                    timestep=self.der_model_set.timesteps,
                    der=linear_electric_grid_model_set.electric_grid_model.ders,
                ),
            ),
            broadcast="timestep",
        )
        self.optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="energy_stage_1", timestep=self.der_model_set.timesteps)),
            (
                "variable",
                1.0,
                dict(
                    name="energy_deviation_up_stage_2",
                    scenario=["down_reserve_stage_2"],
                    timestep=self.der_model_set.timesteps,
                ),
            ),
            (
                "variable",
                -1.0,
                dict(
                    name="energy_deviation_down_stage_2",
                    scenario=["down_reserve_stage_2"],
                    timestep=self.der_model_set.timesteps,
                ),
            ),
            ("variable", 1.0, dict(name="down_reserve_stage_1", timestep=self.der_model_set.timesteps)),
            "==",
            (
                "variable",
                (
                    -1.0
                    * np.array([np.real(linear_electric_grid_model_set.electric_grid_model.der_power_vector_reference)])
                ),
                dict(
                    name="der_active_power_vector",
                    scenario=["down_reserve_stage_2"],
                    timestep=self.der_model_set.timesteps,
                    der=linear_electric_grid_model_set.electric_grid_model.ders,
                ),
            ),
            broadcast="timestep",
        )

        self.optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="energy_deviation_up_stage_2", timestep=self.der_model_set.timesteps)),
            ">=",
            ("constant", 0),
            broadcast="timestep",
        )
        self.optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="energy_deviation_down_stage_2", timestep=self.der_model_set.timesteps)),
            ">=",
            ("constant", 0),
            broadcast="timestep",
        )

        # DER constraints.
        # - The following has been copied from `mesmo.der_models.DERModelSet.define_optimization_constraints()`.
        # - Only state and output equations have been modified to consider the disturbance uncertainty variable.

        # Define DER model constraints.
        # Initial state.
        # - For states which represent storage state of charge, initial state of charge is final state of charge.
        if any(self.der_model_set.states.isin(self.der_model_set.storage_states)):
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1.0,
                    dict(
                        name="state_vector",
                        scenario=self.scenarios_stage_2,
                        timestep=self.der_model_set.timesteps[0],
                        state=self.der_model_set.states[
                            self.der_model_set.states.isin(self.der_model_set.storage_states)
                        ],
                    ),
                ),
                "==",
                (
                    "variable",
                    1.0,
                    dict(
                        name="state_vector",
                        scenario=self.scenarios_stage_2,
                        timestep=self.der_model_set.timesteps[-1],
                        state=self.der_model_set.states[
                            self.der_model_set.states.isin(self.der_model_set.storage_states)
                        ],
                    ),
                ),
                broadcast="scenario",
            )
        # - For other states, set initial state according to the initial state vector.
        if any(~self.der_model_set.states.isin(self.der_model_set.storage_states)):
            self.optimization_problem.define_constraint(
                ("constant", "state_vector_initial", dict(scenario=self.scenarios_stage_2)),
                "==",
                (
                    "variable",
                    1.0,
                    dict(
                        name="state_vector",
                        scenario=self.scenarios_stage_2,
                        timestep=self.der_model_set.timesteps[0],
                        state=self.der_model_set.states[
                            ~self.der_model_set.states.isin(self.der_model_set.storage_states)
                        ],
                    ),
                ),
                broadcast="scenario",
            )

        for scenario in self.scenarios_stage_2:

            # State equation.
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1.0,
                    dict(name="state_vector", scenario=[scenario], timestep=self.der_model_set.timesteps[1:]),
                ),
                "==",
                (
                    "variable",
                    "state_matrix",
                    dict(name="state_vector", scenario=[scenario], timestep=self.der_model_set.timesteps[:-1]),
                ),
                (
                    "variable",
                    "control_matrix",
                    dict(name="control_vector", scenario=[scenario], timestep=self.der_model_set.timesteps[:-1]),
                ),
                ("constant", "disturbance_state_equation", dict(scenario=[scenario])),
                (
                    "variable",
                    "delta_disturbance_matrix",
                    dict(
                        name="disturbance_uncertainty_vector",
                        scenario=["delta"],
                        timestep=self.der_model_set.timesteps[:-1],
                    ),
                ),
                broadcast=["timestep"],
            )

            # Output equation.
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1.0,
                    dict(name="output_vector", scenario=[scenario], timestep=self.der_model_set.timesteps),
                ),
                "==",
                (
                    "variable",
                    "state_output_matrix",
                    dict(name="state_vector", scenario=[scenario], timestep=self.der_model_set.timesteps),
                ),
                (
                    "variable",
                    "control_output_matrix",
                    dict(name="control_vector", scenario=[scenario], timestep=self.der_model_set.timesteps),
                ),
                ("constant", "disturbance_output_equation", dict(scenario=[scenario])),
                (
                    "variable",
                    "delta_disturbance_output_matrix",
                    dict(
                        name="disturbance_uncertainty_vector", scenario=["delta"], timestep=self.der_model_set.timesteps
                    ),
                ),
                broadcast=["timestep"],
            )

        # Output limits.
        self.optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(name="output_vector", scenario=self.scenarios_stage_2, timestep=self.der_model_set.timesteps),
            ),
            ">=",
            ("constant", "output_minimum_timeseries", dict(scenario=self.scenarios_stage_2)),
            broadcast=["timestep", "scenario"],
        )
        self.optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(name="output_vector", scenario=self.scenarios_stage_2, timestep=self.der_model_set.timesteps),
            ),
            "<=",
            ("constant", "output_maximum_timeseries", dict(scenario=self.scenarios_stage_2)),
            broadcast=["timestep", "scenario"],
        )

        # Define connection constraints.
        if len(self.der_model_set.electric_ders) > 0:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1.0,
                    dict(
                        name="der_active_power_vector",
                        scenario=self.scenarios_stage_2,
                        timestep=self.der_model_set.timesteps,
                        der=self.der_model_set.electric_ders,
                    ),
                ),
                "==",
                ("constant", "active_power_constant", dict(scenario=self.scenarios_stage_2)),
                (
                    "variable",
                    "mapping_active_power_by_output",
                    dict(name="output_vector", scenario=self.scenarios_stage_2, timestep=self.der_model_set.timesteps),
                ),
                broadcast=["timestep", "scenario"],
            )
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1.0,
                    dict(
                        name="der_reactive_power_vector",
                        scenario=self.scenarios_stage_2,
                        timestep=self.der_model_set.timesteps,
                        der=self.der_model_set.electric_ders,
                    ),
                ),
                "==",
                ("constant", "reactive_power_constant", dict(scenario=self.scenarios_stage_2)),
                (
                    "variable",
                    "mapping_reactive_power_by_output",
                    dict(name="output_vector", scenario=self.scenarios_stage_2, timestep=self.der_model_set.timesteps),
                ),
                broadcast=["timestep", "scenario"],
            )
        if len(self.der_model_set.thermal_ders) > 0:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1.0,
                    dict(
                        name="der_thermal_power_vector",
                        scenario=self.scenarios_stage_2,
                        timestep=self.der_model_set.timesteps,
                        der=self.der_model_set.thermal_ders,
                    ),
                ),
                "==",
                ("constant", "thermal_power_constant", dict(scenario=self.scenarios_stage_2)),
                (
                    "variable",
                    "mapping_thermal_power_by_output",
                    dict(name="output_vector", scenario=self.scenarios_stage_2, timestep=self.der_model_set.timesteps),
                ),
                broadcast=["timestep", "scenario"],
            )

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (self.der_model_set.timesteps[1] - self.der_model_set.timesteps[0]) / pd.Timedelta(
            "1h"
        )

        # Define objective.
        self.optimization_problem.define_objective(
            (
                "variable",
                -1.0,
                dict(name="energy_stage_1", timestep=self.der_model_set.timesteps),
                dict(name="price_uncertainty_vector", price_type="energy", timestep=self.der_model_set.timesteps),
            ),
            (
                "variable",
                dro_data_set.dro_base_data.at[0, "prob_up_reserve_activated"],
                dict(name="up_reserve_stage_1", timestep=self.der_model_set.timesteps),
                dict(name="price_uncertainty_vector", price_type="up_reserve", timestep=self.der_model_set.timesteps),
            ),
            (
                "variable",
                dro_data_set.dro_base_data.at[0, "prob_down_reserve_activated"],
                dict(name="down_reserve_stage_1", timestep=self.der_model_set.timesteps),
                dict(name="price_uncertainty_vector", price_type="down_reserve", timestep=self.der_model_set.timesteps),
            ),
            (
                "variable",
                (-1.0 * dro_data_set.dro_base_data.at[0, "penalty_energy_deviation ($/kWh)"] * timestep_interval_hours),
                dict(name="energy_deviation_up_stage_2", timestep=self.der_model_set.timesteps),
            ),
            (
                "variable",
                (-1.0 * dro_data_set.dro_base_data.at[0, "penalty_energy_deviation ($/kWh)"] * timestep_interval_hours),
                dict(name="energy_deviation_down_stage_2", timestep=self.der_model_set.timesteps),
            ),
            (
                "variable",
                (-0.01 * timestep_interval_hours),
                dict(name="der_active_power_vector", timestep=self.der_model_set.timesteps),
            ),
        )

        # Obtain DRO matrices.
        self.stage_1_index = mesmo.utils.get_index(self.optimization_problem.variables, scenario=self.scenarios_stage_1)
        self.stage_2_index = mesmo.utils.get_index(self.optimization_problem.variables, scenario=self.scenarios_stage_2)
        self.delta_index = mesmo.utils.get_index(self.optimization_problem.variables, scenario=["delta"])
        a_matrix = self.optimization_problem.get_a_matrix()
        self.r_matrix_2_stage_1 = a_matrix[:, self.stage_1_index]
        self.r_matrix_2_stage_2 = a_matrix[:, self.stage_2_index]
        self.r_matrix_2_delta = a_matrix[:, self.delta_index]
        self.t_vector = self.optimization_problem.get_b_vector()
        q_matrix = self.optimization_problem.get_q_matrix()
        self.w_matrix_2_stage_1_delta = q_matrix[np.ix_(self.stage_1_index, self.delta_index)]
        c_vector = self.optimization_problem.get_c_vector()
        self.w_vector_2_stage_2 = c_vector[:, self.stage_2_index]


def main():

    # Settings.
    scenario_name = "paper_2021_zhang_dro"
    mesmo.data_interface.recreate_database()

    # Get results path.
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Get data.
    dro_data_set = DRODataSet((pathlib.Path(__file__).parent / "dro_data"))

    # Get standard form of stage 1.
    stage_1 = Stage1(scenario_name, dro_data_set)

    # Instantiate optimization problem.
    optimization_problem = mesmo.solutions.OptimizationProblem()

    # Define optimization problem.
    optimization_problem.define_variable("x_vector", index=range(len(stage_1.optimization_problem.variables)))
    optimization_problem.define_constraint(
        (
            "variable",
            stage_1.r_matrix_1_stage_1,
            dict(name="x_vector", index=range(len(stage_1.optimization_problem.variables))),
        ),
        "<=",
        ("constant", stage_1.t_vector_1),
    )
    optimization_problem.define_objective(
        (
            "variable",
            -1.0 * stage_1.w_vector_1_stage_1,
            dict(name="x_vector", index=range(len(stage_1.optimization_problem.variables))),
        )
    )

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results_stage_1 = stage_1.optimization_problem.get_results(optimization_problem.x_vector)
    energy_stage_1 = results_stage_1["energy_stage_1"]
    up_reserve_stage_1 = results_stage_1["up_reserve_stage_1"]
    down_reserve_stage_1 = results_stage_1["down_reserve_stage_1"]

    stage_2 = Stage2(scenario_name, dro_data_set)

    # Re-instantiate optimization problem.
    optimization_problem = mesmo.solutions.OptimizationProblem()

    # Define optimization problem.
    optimization_problem.define_variable("stage_1_vector", index=stage_2.stage_1_index)
    optimization_problem.define_variable("stage_2_vector", index=stage_2.stage_2_index)
    optimization_problem.define_variable("delta_vector", index=stage_2.delta_index)
    optimization_problem.define_constraint(
        ("variable", stage_2.r_matrix_2_stage_1, dict(name="stage_1_vector")),
        ("variable", stage_2.r_matrix_2_stage_2, dict(name="stage_2_vector")),
        ("variable", stage_2.r_matrix_2_delta, dict(name="delta_vector")),
        "<=",
        ("constant", stage_2.t_vector),
    )
    optimization_problem.define_constraint(("variable", 1.0, dict(name="delta_vector")), "==", ("constant", 0.0))
    optimization_problem.define_constraint(
        (
            "variable",
            1.0,
            dict(
                name="stage_1_vector",
                index=mesmo.utils.get_index(stage_2.optimization_problem.variables, name="energy_stage_1"),
            ),
        ),
        "==",
        ("constant", energy_stage_1.values),
    )
    optimization_problem.define_constraint(
        (
            "variable",
            1.0,
            dict(
                name="stage_1_vector",
                index=mesmo.utils.get_index(stage_2.optimization_problem.variables, name="up_reserve_stage_1"),
            ),
        ),
        "==",
        ("constant", up_reserve_stage_1.values),
    )
    optimization_problem.define_constraint(
        (
            "variable",
            1.0,
            dict(
                name="stage_1_vector",
                index=mesmo.utils.get_index(stage_2.optimization_problem.variables, name="down_reserve_stage_1"),
            ),
        ),
        "==",
        ("constant", down_reserve_stage_1.values),
    )

    optimization_problem.define_objective(
        ("variable", -1.0 * stage_2.w_matrix_2_stage_1_delta, dict(name="stage_1_vector"), dict(name="delta_vector")),
        ("variable", -1.0 * stage_2.w_vector_2_stage_2, dict(name="stage_2_vector")),
    )

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results_stage_2 = stage_2.optimization_problem.get_results(optimization_problem.x_vector)
    energy_deviation_up_stage_2 = results_stage_2["energy_deviation_up_stage_2"]
    energy_deviation_down_stage_2 = results_stage_2["energy_deviation_down_stage_2"]

    # Plot some results.
    figure = go.Figure()
    for (scenario_index, scenario) in enumerate(stage_2.scenarios_stage_2):
        figure.add_scatter(
            x=stage_2.der_model_set.timesteps,
            y=energy_deviation_up_stage_2.loc[:, scenario].values,
            name=f"deviation_up_{scenario}",
            line=go.scatter.Line(shape="hv", width=6 - scenario_index, dash="dot"),
        )
        figure.add_scatter(
            x=stage_2.der_model_set.timesteps,
            y=-1.0 * energy_deviation_down_stage_2.loc[:, scenario].values,
            name=f"deviation_down_{scenario}",
            line=go.scatter.Line(shape="hv", width=6 - scenario_index, dash="dot"),
        )
    mesmo.utils.write_figure_plotly(figure, (results_path / f"0_power_balance"))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
