"""DRO problem stage 1."""

import numpy as np
import pandas as pd
import pathlib
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp

import mesmo
from dro_data_interface import DRODataSet


class Stage1(object):
    def __init__(self, scenario_name, dro_data_set, enable_electric_grid_model=True):

        mesmo.utils.logger.info("Initializing stage 1 problem...")

        # Settings.
        self.scenarios_stage_1 = ["no_reserve_stage_1", "up_reserve_stage_1", "down_reserve_stage_1"]

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

        # Instantiate optimization problem.
        self.optimization_problem = mesmo.solutions.OptimizationProblem()

        # Define DER problem.
        self.der_model_set.define_optimization_variables(self.optimization_problem, scenarios=self.scenarios_stage_1)
        self.der_model_set.define_optimization_parameters(
            self.optimization_problem, price_data, scenarios=self.scenarios_stage_1
        )
        self.der_model_set.define_optimization_constraints(self.optimization_problem, scenarios=self.scenarios_stage_1)

        # Define electric grid problem.
        if enable_electric_grid_model:
            linear_electric_grid_model_set.define_optimization_variables(
                self.optimization_problem, scenarios=self.scenarios_stage_1
            )
            linear_electric_grid_model_set.define_optimization_parameters(
                self.optimization_problem, price_data, scenarios=self.scenarios_stage_1
            )
            linear_electric_grid_model_set.define_optimization_constraints(
                self.optimization_problem, scenarios=self.scenarios_stage_1
            )

        # Define additional variables.
        self.optimization_problem.define_variable(
            "energy_stage_1", timestep=self.der_model_set.timesteps, scenario=["no_reserve_stage_1"]
        )
        self.optimization_problem.define_variable(
            "up_reserve_stage_1", timestep=self.der_model_set.timesteps, scenario=["up_reserve_stage_1"]
        )
        self.optimization_problem.define_variable(
            "down_reserve_stage_1", timestep=self.der_model_set.timesteps, scenario=["down_reserve_stage_1"]
        )

        self.optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="up_reserve_stage_1")), ">=", ("constant", 0.0)
        )
        self.optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="down_reserve_stage_1")), ">=", ("constant", 0.0)
        )

        # Define power balance constraints.
        self.optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="energy_stage_1", timestep=self.der_model_set.timesteps)),
            "==",
            (
                "variable",
                (
                    -1.0
                    * np.array([np.real(linear_electric_grid_model_set.electric_grid_model.der_power_vector_reference)])
                ),
                dict(
                    name="der_active_power_vector",
                    scenario="no_reserve_stage_1",
                    timestep=self.der_model_set.timesteps,
                    der=self.der_model_set.ders,
                ),
            ),
            broadcast="timestep",
        )
        self.optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="energy_stage_1", timestep=self.der_model_set.timesteps)),
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
                    scenario="up_reserve_stage_1",
                    timestep=self.der_model_set.timesteps,
                    der=self.der_model_set.ders,
                ),
            ),
            broadcast="timestep",
        )
        self.optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="energy_stage_1", timestep=self.der_model_set.timesteps)),
            ("variable", -1.0, dict(name="down_reserve_stage_1", timestep=self.der_model_set.timesteps)),
            "==",
            (
                "variable",
                (
                    -1.0
                    * np.array([np.real(linear_electric_grid_model_set.electric_grid_model.der_power_vector_reference)])
                ),
                dict(
                    name="der_active_power_vector",
                    scenario="down_reserve_stage_1",
                    timestep=self.der_model_set.timesteps,
                    der=self.der_model_set.ders,
                ),
            ),
            broadcast="timestep",
        )

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (self.der_model_set.timesteps[1] - self.der_model_set.timesteps[0]) / pd.Timedelta(
            "1h"
        )

        # Obtain energy price timeseries.
        # TODO: Slice price data properly according to selected timestamps.
        price_timeseries_energy = dro_data_set.energy_price[0 : len(self.der_model_set.timesteps)].to_numpy()
        price_timeseries_reserve = dro_data_set.contingency_reserve_price[
            0 : len(self.der_model_set.timesteps)
        ].to_numpy()

        # Define objective.
        # Active power cost / revenue.
        # - Cost for load / demand, revenue for generation / supply.
        self.optimization_problem.define_objective(
            (
                "variable",
                (-1.0 * np.array([price_timeseries_energy]) * timestep_interval_hours),
                dict(name="energy_stage_1", timestep=self.der_model_set.timesteps),
            ),
            (
                "variable",
                (
                    +1.0
                    * dro_data_set.dro_base_data["prob_up_reserve_bidded"].values
                    * np.array([price_timeseries_reserve])
                    * timestep_interval_hours
                ),
                dict(name="up_reserve_stage_1", timestep=self.der_model_set.timesteps),
            ),
            (
                "variable",
                (
                    +1.0
                    * dro_data_set.dro_base_data["prob_down_reserve_bidded"].values
                    * np.array([price_timeseries_reserve])
                    * timestep_interval_hours
                ),
                dict(name="up_reserve_stage_1", timestep=self.der_model_set.timesteps),
            ),
        )

        # Obtain standard form matrix / vector representation.
        self.r_matrix_1_stage_1 = self.optimization_problem.get_a_matrix()
        self.t_vector_1 = self.optimization_problem.get_b_vector()
        self.w_vector_1_stage_1 = self.optimization_problem.get_c_vector()


def main():

    # Settings.
    scenario_name = "paper_2021_zhang_dro"
    mesmo.data_interface.recreate_database()

    # Obtain data.
    dro_data_set = DRODataSet((pathlib.Path(__file__).parent / "dro_data"))

    # Get results path.
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

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
    stage_1.optimization_problem.results = stage_1.optimization_problem.get_results(optimization_problem.x_vector)
    results = stage_1.der_model_set.get_optimization_results(stage_1.optimization_problem, stage_1.scenarios_stage_1)

    pd.DataFrame(optimization_problem.x_vector).to_csv((results_path / f"s_1_vector_det.csv"))

    # Obtain reserve results.
    energy_stage_1 = pd.Series(
        stage_1.optimization_problem.results["energy_stage_1"].values.ravel(), index=stage_1.der_model_set.timesteps
    )
    up_reserve_stage_1 = pd.Series(
        stage_1.optimization_problem.results["up_reserve_stage_1"].values.ravel(), index=stage_1.der_model_set.timesteps
    )
    down_reserve_stage_1 = pd.Series(
        stage_1.optimization_problem.results["down_reserve_stage_1"].values.ravel(),
        index=stage_1.der_model_set.timesteps,
    )

    #
    pd.DataFrame(energy_stage_1).to_csv((results_path / f"energy_det.csv"))
    pd.DataFrame(up_reserve_stage_1).to_csv((results_path / f"up_reserve_det.csv"))
    pd.DataFrame(down_reserve_stage_1).to_csv((results_path / f"down_reserve_det.csv"))
    objective_det = {"objective_value": [optimization_problem.objective]}
    objective_det_df = pd.DataFrame(data=objective_det)
    objective_det_df.to_csv((results_path / f"objective_det.csv"))

    # Plot some results.
    figure = go.Figure()
    figure.add_scatter(
        x=energy_stage_1.index,
        y=energy_stage_1.values,
        name="no_reserve_stage_1",
        line=go.scatter.Line(shape="hv", width=5, dash="dot"),
    )
    figure.add_scatter(
        x=up_reserve_stage_1.index,
        y=up_reserve_stage_1.values,
        name="up_reserve_stage_1",
        line=go.scatter.Line(shape="hv", width=4, dash="dot"),
    )
    figure.add_scatter(
        x=down_reserve_stage_1.index,
        y=down_reserve_stage_1.values,
        name="down_reserve_stage_1",
        line=go.scatter.Line(shape="hv", width=3, dash="dot"),
    )
    figure.add_scatter(
        x=up_reserve_stage_1.index,
        y=(energy_stage_1 + up_reserve_stage_1).values,
        name="no_reserve + up_reserve",
        line=go.scatter.Line(shape="hv", width=2, dash="dot"),
    )
    figure.add_scatter(
        x=up_reserve_stage_1.index,
        y=(energy_stage_1 - down_reserve_stage_1).values,
        name="no_reserve - down_reserve",
        line=go.scatter.Line(shape="hv", width=1, dash="dot"),
    )
    figure.update_layout(
        title=f"Power balance",
        xaxis=go.layout.XAxis(tickformat="%H:%M"),
        legend=go.layout.Legend(x=0.01, xanchor="auto", y=0.3, yanchor="auto"),
    )
    # figure.show()
    mesmo.utils.write_figure_plotly(figure, (results_path / f"0_power_balance"))

    for der_name, der_model in stage_1.der_model_set.flexible_der_models.items():

        for output in der_model.outputs:
            figure = go.Figure()
            figure.add_scatter(
                x=der_model.output_maximum_timeseries.index,
                y=der_model.output_maximum_timeseries.loc[:, output].values,
                name="Maximum bound",
                line=go.scatter.Line(shape="hv"),
            )
            figure.add_scatter(
                x=der_model.output_minimum_timeseries.index,
                y=der_model.output_minimum_timeseries.loc[:, output].values,
                name="Minimum bound",
                line=go.scatter.Line(shape="hv"),
            )
            for number, stochastic_scenario in enumerate(stage_1.scenarios_stage_1):
                if number == 0:
                    figure.add_scatter(
                        x=results.output_vector.index,
                        y=results.output_vector.loc[:, [(stochastic_scenario, (der_name, output))]].values.ravel(),
                        name=f"optimal value in {stochastic_scenario} scenario",
                        line=go.scatter.Line(shape="hv", width=number + 5),
                    )
                elif number == 1:
                    figure.add_scatter(
                        x=results.output_vector.index,
                        y=results.output_vector.loc[:, [(stochastic_scenario, (der_name, output))]].values.ravel(),
                        name=f"optimal value in {stochastic_scenario} scenario",
                        line=go.scatter.Line(shape="hv", width=number + 4, dash="dashdot"),
                    )
                else:
                    figure.add_scatter(
                        x=results.output_vector.index,
                        y=results.output_vector.loc[:, [(stochastic_scenario, (der_name, output))]].values.ravel(),
                        name=f"optimal value in {stochastic_scenario} scenario",
                        line=go.scatter.Line(shape="hv", width=number + 3, dash="dot"),
                    )
            figure.update_layout(
                title=f"DER: ({der_model.der_type}, {der_name}) / Output: {output}",
                xaxis=go.layout.XAxis(tickformat="%H:%M"),
                legend=go.layout.Legend(x=0.01, xanchor="auto", y=0.3, yanchor="auto", font=dict(size=9)),
            )
            # figure.show()
            mesmo.utils.write_figure_plotly(
                figure, (results_path / f"der_{der_model.der_type}_{der_name}_output_{output}")
            )

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
