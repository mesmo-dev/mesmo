"""Project SITEM baseline scenario evaluation script."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import fledge.problems
import fledge.utils


def main():

    # Settings.
    scenario_name = 'ema_sample_grid'
    results_path = fledge.utils.get_results_path('run_sitem_baseline', scenario_name)

    # Obtain nominal operation problem & solution.
    problem = fledge.problems.NominalOperationProblem(scenario_name)
    problem.solve()
    results = problem.get_results()

    # Obtain additional results.
    branch_power_vector_magnitude_relative = (
        (np.abs(results['branch_power_vector_1']) + np.abs(results['branch_power_vector_2'])) / 2
        / problem.electric_grid_model.branch_power_vector_magnitude_reference
    )
    branch_power_vector_magnitude_relative.loc['maximum', :] = branch_power_vector_magnitude_relative.max(axis='rows')
    node_voltage_vector_magnitude_per_unit = (
        np.abs(results['node_voltage_vector'])
        / np.abs(problem.electric_grid_model.node_voltage_vector_reference)
    )
    node_voltage_vector_magnitude_per_unit.loc['maximum', :] = node_voltage_vector_magnitude_per_unit.max(axis='rows')
    node_voltage_vector_magnitude_per_unit.loc['minimum', :] = node_voltage_vector_magnitude_per_unit.min(axis='rows')
    results.update({
        'branch_power_vector_magnitude_relative': branch_power_vector_magnitude_relative,
        'node_voltage_vector_magnitude_per_unit': node_voltage_vector_magnitude_per_unit
    })

    # Print results.
    print(results)

    # Plot some results.
    plt.title('Branch utilization [%]')
    plt.bar(
        range(len(problem.electric_grid_model.branches)),
        100.0 * branch_power_vector_magnitude_relative.loc['maximum', :]
    )
    plt.hlines(100.0, -0.5, len(problem.electric_grid_model.branches) - 0.5, colors='red')
    plt.xticks(
        range(len(problem.electric_grid_model.branches)),
        problem.electric_grid_model.branches,
        rotation=45,
        ha='right'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    plt.show()

    plt.title('Maximum voltage rise [%]')
    plt.bar(
        range(len(problem.electric_grid_model.nodes)),
        100.0 * (node_voltage_vector_magnitude_per_unit.loc['maximum', :] - 1.0)
    )
    plt.hlines(5.0, -0.5, len(problem.electric_grid_model.nodes) - 0.5, colors='red')
    plt.xticks(
        range(len(problem.electric_grid_model.nodes)),
        problem.electric_grid_model.nodes,
        rotation=45,
        ha='right'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    plt.show()

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
    plt.show()

    # Store results to CSV.
    results.to_csv(results_path)

    # Print results path.
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
