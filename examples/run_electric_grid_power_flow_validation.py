"""Example script for testing / validating the electric grid power flow solution."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import fledge.config
import fledge.data_interface
import fledge.electric_grid_models
import fledge.utils


def main():

    # Settings.
    scenario_name = fledge.config.config['tests']['scenario_name']
    results_path = fledge.utils.get_results_path('run_electric_grid_power_flow_validation', scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain electric grid model.
    electric_grid_model_default = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    electric_grid_model_opendss = fledge.electric_grid_models.ElectricGridModelOpenDSS(scenario_name)

    # Obtain power solution.
    power_flow_solution_fixed_point = (
        fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model_default)
    )
    power_flow_solution_opendss = (
        fledge.electric_grid_models.PowerFlowSolutionOpenDSS(electric_grid_model_opendss)
    )

    # Obtain results.
    node_voltage_vector_magnitude_fixed_point = (
        np.abs(power_flow_solution_fixed_point.node_voltage_vector)
    )
    node_voltage_vector_magnitude_opendss = (
        np.abs(power_flow_solution_opendss.node_voltage_vector)
    )
    node_voltage_vector_magnitude_per_unit_fixed_point = (
        node_voltage_vector_magnitude_fixed_point / np.abs(electric_grid_model_default.node_voltage_vector_reference)
    )
    node_voltage_vector_magnitude_per_unit_opendss = (
        node_voltage_vector_magnitude_opendss / np.abs(electric_grid_model_default.node_voltage_vector_reference)
    )
    branch_power_vector_1_magnitude_fixed_point = (
        np.abs(power_flow_solution_fixed_point.branch_power_vector_1)
    )
    branch_power_vector_1_magnitude_opendss = (
        np.abs(power_flow_solution_opendss.branch_power_vector_1)
    )
    branch_power_vector_1_magnitude_per_unit_fixed_point = (
        branch_power_vector_1_magnitude_fixed_point / electric_grid_model_default.branch_power_vector_magnitude_reference
    )
    branch_power_vector_1_magnitude_per_unit_opendss = (
        branch_power_vector_1_magnitude_opendss / electric_grid_model_default.branch_power_vector_magnitude_reference
    )
    branch_power_vector_2_magnitude_fixed_point = (
        np.abs(power_flow_solution_fixed_point.branch_power_vector_2)
    )
    branch_power_vector_2_magnitude_opendss = (
        np.abs(power_flow_solution_opendss.branch_power_vector_2)
    )
    branch_power_vector_2_magnitude_per_unit_fixed_point = (
        branch_power_vector_2_magnitude_fixed_point / electric_grid_model_default.branch_power_vector_magnitude_reference
    )
    branch_power_vector_2_magnitude_per_unit_opendss = (
        branch_power_vector_2_magnitude_opendss / electric_grid_model_default.branch_power_vector_magnitude_reference
    )

    # Plot results.
    plt.title('Node voltage magnitude [p.u.]')
    plt.scatter(
        range(len(electric_grid_model_default.nodes)),
        node_voltage_vector_magnitude_per_unit_fixed_point,
        marker=4
    )
    plt.scatter(
        range(len(electric_grid_model_default.nodes)),
        node_voltage_vector_magnitude_per_unit_opendss,
        marker=5
    )
    plt.xticks(
        range(len(electric_grid_model_default.nodes)),
        electric_grid_model_default.nodes,
        rotation=45,
        ha='right'
    )
    plt.legend(['Fixed point', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    plt.show()
    plt.close()

    plt.title('Branch power (direction 1) magnitude [p.u.]')
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        branch_power_vector_1_magnitude_per_unit_fixed_point,
        marker=4
    )
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        branch_power_vector_1_magnitude_per_unit_opendss,
        marker=5
    )
    plt.xticks(
        range(len(electric_grid_model_default.branches)),
        electric_grid_model_default.branches,
        rotation=45,
        ha='right'
    )
    plt.legend(['Fixed point', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    plt.show()
    plt.close()

    plt.title('Branch power (direction 2) magnitude [p.u.]')
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        branch_power_vector_2_magnitude_per_unit_fixed_point,
        marker=4
    )
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        branch_power_vector_2_magnitude_per_unit_opendss,
        marker=5
    )
    plt.xticks(
        range(len(electric_grid_model_default.branches)),
        electric_grid_model_default.branches,
        rotation=45,
        ha='right'
    )
    plt.legend(['Fixed point', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    plt.show()
    plt.close()

    plt.title('Total power loss [VA]')
    plt.scatter(
        [1, 2],
        [
            np.real(power_flow_solution_fixed_point.loss),
            np.imag(power_flow_solution_fixed_point.loss)
        ],
        marker=4
    )
    plt.scatter(
        [1, 2],
        [
            np.real(power_flow_solution_opendss.loss),
            np.imag(power_flow_solution_opendss.loss)
        ],
        marker=5
    )
    plt.xticks(
        [0.5, 1, 2, 2.5],
        ['', 'active', 'reactive', ''],
        rotation=45,
        ha='right'
    )
    plt.legend(['Fixed point', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    plt.show()
    plt.close()

    # Print results path.
    os.startfile(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
