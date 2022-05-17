"""Stochastic optimization (SO) problem equivalent for DRO problem."""

import numpy as np
import pandas as pd
import pathlib
import plotly.express as px
import plotly.graph_objects as go

import mesmo
from dro_stage_1 import Stage1
from dro_stage_2 import Stage2
from dro_data_interface import DRODataSet
from dro_data_interface import DROAmbiguitySet


class ScenarioGeneration(object):

    delta: np.array
    probability: float

    def __init__(self, scenario_number: int, delta_indices_stage2: pd.Index, dro_ambiguity_set: DROAmbiguitySet):

        self.delta = np.zeros((len(delta_indices_stage2), scenario_number))
        self.probability = 1 / scenario_number
        for index in range(len(delta_indices_stage2)):
            self.delta[index, :] = np.random.normal(0, 1e-6 * dro_ambiguity_set.gamma.iloc[index], scenario_number)


def main():

    scenario_name = "paper_2021_zhang_dro"
    enable_electric_grid_model = True
    mesmo.data_interface.recreate_database()
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)

    # Get results path.
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Obtain data objects.
    dro_data_set = DRODataSet((pathlib.Path(__file__).parent / "dro_data"))

    # Obtain individual stage optimization problems.
    stage_1 = Stage1(scenario_name, dro_data_set, enable_electric_grid_model=enable_electric_grid_model)
    stage_2 = Stage2(scenario_name, dro_data_set, enable_electric_grid_model=enable_electric_grid_model)

    # Define scenario sets stage 1.
    price_categories = ["energy", "up_reserve", "down_reserve"]
    uncertainty_index = price_categories + stage_2.delta_disturbances.to_list()

    so_test_number = 1
    scenario_number = 1000
    objective_array = np.zeros(so_test_number)

    dro_ambiguity_set = DROAmbiguitySet(
        scenario_name, stage_2.optimization_problem, stage_2.delta_index, dro_data_set, stage_2.delta_disturbances
    )

    scenario_uncertainty = ScenarioGeneration(scenario_number, stage_2.delta_index, dro_ambiguity_set)

    for test_index in range(so_test_number):

        mesmo.utils.logger.info(f"Cheap SO: {test_index} out of {so_test_number}")

        # Instantiate optimization problem.
        optimization_problem = mesmo.solutions.OptimizationProblem()

        optimization_problem.define_variable(
            "stage_1_vector", stage_1_index=range(len(stage_1.optimization_problem.variables))
        )
        optimization_problem.define_variable(
            "stage_2_vector", scenario_index=range(scenario_number), stage_2_index=range(len(stage_2.stage_2_index))
        )
        optimization_problem.define_variable(
            "delta",
            uncertainty_index=uncertainty_index,
            timestep=der_model_set.timesteps,
            scenario_index=range(scenario_number),
        )

        optimization_problem.define_constraint(
            ("variable", stage_1.r_matrix_1_stage_1, dict(name="stage_1_vector")),
            "<=",
            ("constant", stage_1.t_vector_1),
        )
        optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="delta")),
            "==",
            ("constant", scenario_uncertainty.delta.ravel()),
        )

        # TODO: replace with broadcasting (index order for delta/s2 not consistent yet)
        for scenario_i in range(scenario_number):
            optimization_problem.define_constraint(
                ("variable", stage_2.r_matrix_2_stage_1, dict(name="stage_1_vector")),
                ("variable", stage_2.r_matrix_2_stage_2, dict(name="stage_2_vector", scenario_index=scenario_i)),
                ("variable", stage_2.r_matrix_2_delta, dict(name="delta", scenario_index=scenario_i)),
                "<=",
                ("constant", stage_2.t_vector),
            )

        optimization_problem.define_objective(
            ("variable", -1.0 * stage_1.w_vector_1_stage_1, dict(name="stage_1_vector")),
            (
                "variable",
                (
                    -1.0
                    * scenario_uncertainty.probability
                    * stage_2.w_matrix_2_stage_1_delta.toarray().repeat(scenario_number, axis=1)
                ),
                dict(name="stage_1_vector"),
                dict(name="delta"),
            ),
            (
                "variable",
                (-1.0 * np.concatenate([stage_2.w_vector_2_stage_2] * scenario_number, axis=1)),
                dict(name="stage_2_vector"),
            ),
        )

        optimization_problem.solve()
        objective_array[test_index] = optimization_problem.objective

    x_vector_value = optimization_problem.x_vector
    stage_1_index = mesmo.utils.get_index(optimization_problem.variables, name="stage_1_vector")
    stage_1_results = x_vector_value[stage_1_index]
    pd.DataFrame(stage_1_results).to_csv((results_path / f"s_1_vector_so.csv"))

    objective_so = {"objective_value": objective_array}

    objective_so_df = pd.DataFrame(data=objective_so)
    objective_so_df.to_csv((results_path / f"objective_so.csv"))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
