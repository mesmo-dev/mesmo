"""MESMO tutorial: Example 1."""

import plotly.express as px
import mesmo


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data and models.
    price_data = mesmo.data_interface.PriceData(scenario_name)
    linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(scenario_name)
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = mesmo.utils.OptimizationProblem()

    # Define optimization problem.
    linear_electric_grid_model_set.define_optimization_problem(optimization_problem, price_data)
    der_model_set.define_optimization_problem(optimization_problem, price_data)

    # Solve optimization problem.
    optimization_problem.solve()

    # Get results.
    results = mesmo.problems.Results()
    results.update(linear_electric_grid_model_set.get_optimization_results(optimization_problem))
    results.update(der_model_set.get_optimization_results(optimization_problem))
    results.update(linear_electric_grid_model_set.get_optimization_dlmps(optimization_problem, price_data))

    # Store results.
    results.save(results_path)

    # Plot some results.
    figure = px.line(results.branch_power_magnitude_vector_1.loc[:, ('line', '1', 1)].rename('Line 1; phase 1'))
    mesmo.utils.write_figure_plotly(figure, results_path / 'branch_power_line_1_phase_1')

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
