"""MESMO tutorial: Example 2."""

import numpy as np
import os
import pathlib
import plotly.graph_objects as go

import mesmo


def main():

    # Settings.
    scenario_name = "tutorial_example"
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain problems.
    problem_nominal = mesmo.problems.NominalOperationProblem(scenario_name)
    problem_optimal = mesmo.problems.OptimalOperationProblem(scenario_name)

    # Solve nominal operation problem & get results.
    problem_nominal.solve()
    results_nominal = problem_nominal.get_results()

    # Add peak shaving constraint.
    # - DER active power vector optimization variables is per-unit, i.e. scaled by DER reference power vector.
    #   Therefore it is multiplied below to 'unscale' the optimization variable.
    for timestep in problem_optimal.timesteps:
        problem_optimal.optimization_problem.define_constraint(
            (
                "variable",
                np.array([np.real(problem_optimal.electric_grid_model.der_power_vector_reference)]),
                dict(name="der_active_power_vector", timestep=timestep),
            ),
            ">=",
            ("constant", 0.95 * np.min(np.sum(results_nominal.der_active_power_vector, axis=1))),
        )

    # Solve optimal operation problem & get results.
    problem_optimal.solve()
    results_optimal = problem_optimal.get_results()

    # Plot comparison.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=results_nominal.der_active_power_vector.index,
            y=np.abs(np.sum(results_nominal.der_active_power_vector, axis=1)),
            name="Nominal",
            line=go.scatter.Line(shape="hv"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=results_optimal.der_active_power_vector.index,
            y=np.abs(np.sum(results_optimal.der_active_power_vector, axis=1)),
            name="Optimal",
            line=go.scatter.Line(shape="hv"),
        )
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "comparison"))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
