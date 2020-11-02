"""Project SITEM scenario evaluation script."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.plots
import fledge.problems
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_district25'
    results_path = fledge.utils.get_results_path('run_sitem_baseline', scenario_name)
    plot_grid = False

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain nominal operation problem & run simulation.
    problem = fledge.problems.NominalOperationProblem(scenario_name)
    problem.solve()
    results_private_ev_bus = problem.get_results()

    # Remove bus chargers and re-run simulation.
    for der_name in problem.der_model_set.der_models:
        if type(problem.der_model_set.der_models[der_name]) is fledge.der_models.FixedEVChargerModel:
            if 'bus_charger' in der_name:
                problem.der_model_set.der_models[der_name].active_power_nominal_timeseries *= 0
                problem.der_model_set.der_models[der_name].reactive_power_nominal_timeseries *= 0
    problem.solve()
    results_private_ev = problem.get_results()

    # Remove private EV chargers and re-run simulation.
    for der_name in problem.der_model_set.der_models:
        if type(problem.der_model_set.der_models[der_name]) is fledge.der_models.FixedEVChargerModel:
            problem.der_model_set.der_models[der_name].active_power_nominal_timeseries *= 0
            problem.der_model_set.der_models[der_name].reactive_power_nominal_timeseries *= 0
    problem.solve()
    results_baseline = problem.get_results()

    # Store results to CSV.
    results_private_ev_bus.to_csv(results_path)

    # Obtain electric grid graph.
    electric_grid_graph = fledge.plots.ElectricGridGraph(scenario_name)

    # Plot utilization.
    fledge.plots.plot_line_utilization(
        problem.electric_grid_model,
        electric_grid_graph,
        results_private_ev_bus['branch_power_1_magnitude_per_unit'].max() * 100.0,
        results_path,
        label='Maximum',
        value_unit='%',
        horizontal_line_value=100.0
    )
    fledge.plots.plot_transformer_utilization(
        problem.electric_grid_model,
        electric_grid_graph,
        results_private_ev_bus['branch_power_1_magnitude_per_unit'].max() * 100.0,
        results_path,
        label='Maximum',
        value_unit='%',
        horizontal_line_value=100.0
    )
    fledge.plots.plot_node_utilization(
        problem.electric_grid_model,
        electric_grid_graph,
        (results_private_ev_bus['node_voltage_magnitude_per_unit'].min() - 1.0) * - 100.0,
        results_path,
        label='Maximum',
        value_unit='%',
        suffix='drop',
        horizontal_line_value=5.0
    )

    # Plot utilization on grid layout.
    if plot_grid:
        fledge.plots.plot_grid_transformer_utilization(
            problem.electric_grid_model,
            electric_grid_graph,
            results_private_ev_bus['branch_power_1_magnitude_per_unit'] * 100.0,
            results_path,
            vmin=20.0,
            vmax=120.0,
            value_unit='%',
            make_video=True
        )
        fledge.plots.plot_grid_line_utilization(
            problem.electric_grid_model,
            electric_grid_graph,
            results_private_ev_bus['branch_power_1_magnitude_per_unit'] * 100.0,
            results_path,
            vmin=20.0,
            vmax=120.0,
            value_unit='%',
            make_video=True
        )
        fledge.plots.plot_grid_node_utilization(
            problem.electric_grid_model,
            electric_grid_graph,
            (results_private_ev_bus['node_voltage_magnitude_per_unit'] - 1.0) * -100.0,
            results_path,
            vmin=0.0,
            vmax=10.0,
            value_unit='%',
            suffix='drop',
            make_video=True
        )

    # Plot demand timeseries.
    fledge.plots.plot_total_active_power(
        {
            'Baseload + private EV + bus charging': results_private_ev_bus['der_power_vector'],
            'Baseload + private EV': results_private_ev['der_power_vector'],
            'Baseload': results_baseline['der_power_vector']
        },
        results_path
    )

    # Plot line utilization histogram.
    fledge.plots.plot_line_utilization_histogram(
        {
            'Baseload': results_baseline['branch_power_1_magnitude_per_unit'],
            'Baseload + private EV': results_private_ev['branch_power_1_magnitude_per_unit'],
            'Baseload + private EV + bus charging': results_private_ev_bus['branch_power_1_magnitude_per_unit']
        },
        results_path
    )

    # Plot line utilization cumulative.
    fledge.plots.plot_line_utilization_histogram_cumulative(
        {
            'Baseload': results_baseline['branch_power_1_magnitude_per_unit'],
            'Baseload + private EV': results_private_ev['branch_power_1_magnitude_per_unit'],
            'Baseload + private EV + bus charging': results_private_ev_bus['branch_power_1_magnitude_per_unit']
        },
        results_path
    )

    # Plot transformer utilization histogram.
    fledge.plots.plot_transformer_utilization_histogram(
        {
            'Baseload': results_baseline['branch_power_1_magnitude_per_unit'],
            'Baseload + private EV': results_private_ev['branch_power_1_magnitude_per_unit'],
            'Baseload + private EV + bus charging': results_private_ev_bus['branch_power_1_magnitude_per_unit']
        },
        results_path,
        selected_columns=(
            problem.electric_grid_model.branches[
                problem.electric_grid_model.branches.get_level_values('branch_name').str.contains('22kV')
            ]
        )
    )

    # Plot transformer utilization cumulative.
    fledge.plots.plot_transformer_utilization_histogram_cumulative(
        {
            'Baseload': results_baseline['branch_power_1_magnitude_per_unit'],
            'Baseload + private EV': results_private_ev['branch_power_1_magnitude_per_unit'],
            'Baseload + private EV + bus charging': results_private_ev_bus['branch_power_1_magnitude_per_unit']
        },
        results_path,
        selected_columns=(
            problem.electric_grid_model.branches[
                problem.electric_grid_model.branches.get_level_values('branch_name').str.contains('22kV')
            ]
        )
    )

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
