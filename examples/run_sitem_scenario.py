"""Project SITEM scenario evaluation script."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import re

import fledge.data_interface
import fledge.plots
import fledge.problems
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_district25'
    results_path = fledge.utils.get_results_path('run_sitem_baseline', scenario_name)
    plot_detailed_grid = True

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain nominal operation problem & solution.
    problem = fledge.problems.NominalOperationProblem(scenario_name)
    problem.solve()
    results = problem.get_results()

    # Obtain additional results.
    branch_power_vector_magnitude_per_unit = (
        (np.abs(results['branch_power_vector_1']) + np.abs(results['branch_power_vector_2'])) / 2
        / problem.electric_grid_model.branch_power_vector_magnitude_reference
    )
    branch_power_vector_magnitude_per_unit.loc['maximum', :] = branch_power_vector_magnitude_per_unit.max(axis='rows')
    node_voltage_vector_magnitude_per_unit = (
        np.abs(results['node_voltage_vector'])
        / np.abs(problem.electric_grid_model.node_voltage_vector_reference)
    )
    node_voltage_vector_magnitude_per_unit.loc['maximum', :] = node_voltage_vector_magnitude_per_unit.max(axis='rows')
    node_voltage_vector_magnitude_per_unit.loc['minimum', :] = node_voltage_vector_magnitude_per_unit.min(axis='rows')
    results.update({
        'branch_power_vector_magnitude_per_unit': branch_power_vector_magnitude_per_unit,
        'node_voltage_vector_magnitude_per_unit': node_voltage_vector_magnitude_per_unit
    })

    # Print results.
    print(results)

    # Store results to CSV.
    results.to_csv(results_path)

    # Obtain electric grid graph.
    electric_grid_graph = fledge.plots.ElectricGridGraph(scenario_name)

    # Plot electric grid transformer utilization.
    fledge.plots.plot_electric_grid_transformer_utilization(
        problem.electric_grid_model,
        electric_grid_graph,
        branch_power_vector_magnitude_per_unit,
        results_path,
        make_video=True
    )

    # Plot electric grid line utilization.
    if plot_detailed_grid:
        fledge.plots.plot_electric_grid_line_utilization(
            problem.electric_grid_model,
            electric_grid_graph,
            branch_power_vector_magnitude_per_unit,
            results_path,
            make_video=True
        )

    # Plot electric grid nodes voltage drop.
    if plot_detailed_grid:
        fledge.plots.plot_electric_grid_node_voltage_drop(
            problem.electric_grid_model,
            electric_grid_graph,
            node_voltage_vector_magnitude_per_unit,
            results_path,
            make_video=True
        )

    # Plot some results.
    plt.title('Line utilization [%]')
    plt.bar(
        range(len(problem.electric_grid_model.lines)),
        100.0 * branch_power_vector_magnitude_per_unit.loc['maximum', problem.electric_grid_model.lines]
    )
    plt.hlines(100.0, -0.5, len(problem.electric_grid_model.lines) - 0.5, colors='red')
    plt.xticks(
        range(len(problem.electric_grid_model.lines)),
        problem.electric_grid_model.lines,
        rotation=45,
        ha='right'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    # plt.show()
    plt.close()

    plt.title('Transformer utilization [%]')
    plt.bar(
        range(len(problem.electric_grid_model.transformers)),
        100.0 * branch_power_vector_magnitude_per_unit.loc['maximum', problem.electric_grid_model.transformers]
    )
    plt.hlines(100.0, -0.5, len(problem.electric_grid_model.transformers) - 0.5, colors='red')
    plt.xticks(
        range(len(problem.electric_grid_model.transformers)),
        problem.electric_grid_model.transformers,
        rotation=45,
        ha='right'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    # plt.show()
    plt.close()

    plt.title('Maximum voltage drop [%]')
    plt.bar(
        range(len(problem.electric_grid_model.nodes)),
        100.0 * (node_voltage_vector_magnitude_per_unit.loc['minimum', :] - 1.0)
    )
    plt.hlines(-5.0, -0.5, len(problem.electric_grid_model.nodes) - 0.5, colors='red')
    plt.xticks(
        range(len(problem.electric_grid_model.nodes)),
        problem.electric_grid_model.nodes,
        rotation=45,
        ha='right'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    # plt.show()
    plt.close()

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
