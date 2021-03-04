"""Script for testing the local linear electric grid model."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy

import fledge


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    results_path = fledge.utils.get_results_path(__file__, scenario_name)
    power_multipliers = np.arange(-0.9, 2.3, 0.2)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain model.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

    # Obtain power flow solution for nominal power conditions.
    power_flow_solution_initial = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)

    # Obtain shorthands for no-source matrices and vectors.
    node_admittance_matrix_no_source = (
        electric_grid_model.node_admittance_matrix[np.ix_(
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        )].toarray()
    )
    node_voltage_no_source = (
        power_flow_solution_initial.node_voltage_vector[
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        ]
    )
    nodes_no_source_count = np.sum(electric_grid_model.nodes.get_level_values('node_type') == 'no_source')
    nodes_source_count = np.sum(electric_grid_model.nodes.get_level_values('node_type') == 'source')

    # Define utility function for complex block matrix.
    def complex_block(matrix: np.ndarray):
        return np.block([
            [np.real(matrix), -np.imag(matrix)],
            [np.imag(matrix), np.real(matrix)]
        ])

    # Derive sensitivity matrix.
    # - Based on: https://doi.org/10.1109/ALLERTON.2015.7447032
    N_matrix = (
        np.diag(np.concatenate((
            np.ones(nodes_no_source_count),
            -np.ones(nodes_no_source_count)
        )))
    )
    R_matrix = (
        np.block([
            [
                np.diag(np.cos(np.angle(node_voltage_no_source))),
                -1.0 * np.diag(np.abs(node_voltage_no_source) * np.sin(np.angle(node_voltage_no_source)))
            ],
            [
                np.diag(np.sin(np.angle(node_voltage_no_source))),
                np.diag(np.abs(node_voltage_no_source) * np.cos(np.angle(node_voltage_no_source)))
            ]
        ])
    )
    A_matrix = (
        (
            complex_block(np.diag(np.conj(node_admittance_matrix_no_source @ np.transpose([node_voltage_no_source])).ravel()))
            + complex_block(np.diag(node_voltage_no_source)) @ N_matrix @ complex_block(node_admittance_matrix_no_source)
        )
        @ R_matrix
    )
    sensitivity_matrix = scipy.linalg.inv(A_matrix)

    # Split sensitivity matrix.
    sensitivity_voltage_magnitude_by_der_power_active = (
        scipy.linalg.block_diag(
            np.zeros((nodes_source_count, nodes_source_count)),
            sensitivity_matrix[:nodes_no_source_count, :nodes_no_source_count]
        )
        @ electric_grid_model.der_incidence_wye_matrix
        + scipy.linalg.block_diag(
            np.zeros((nodes_source_count, nodes_source_count)),
            sensitivity_matrix[:nodes_no_source_count, :nodes_no_source_count]
        )
        @ electric_grid_model.der_incidence_delta_matrix
    )
    sensitivity_voltage_magnitude_by_der_power_reactive = (
        scipy.linalg.block_diag(
            np.zeros((nodes_source_count, nodes_source_count)),
            sensitivity_matrix[:nodes_no_source_count, nodes_no_source_count:]
        )
        @ electric_grid_model.der_incidence_wye_matrix
        + scipy.linalg.block_diag(
            np.zeros((nodes_source_count, nodes_source_count)),
            sensitivity_matrix[:nodes_no_source_count, nodes_no_source_count:]
        )
        @ electric_grid_model.der_incidence_delta_matrix
    )

    # Instantiate results variables.
    der_power_vector_active = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.ders, dtype=np.float)
    )
    der_power_vector_reactive = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.ders, dtype=np.float)
    )
    der_power_vector_active_change = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.ders, dtype=np.float)
    )
    der_power_vector_reactive_change = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.ders, dtype=np.float)
    )
    node_voltage_vector_magnitude_power_flow = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.nodes, dtype=np.float)
    )
    node_voltage_vector_magnitude_linear_model = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.nodes, dtype=np.float)
    )

    # Obtain DER power / change.
    der_power_vector_active.loc[:, :] = (
        np.transpose([power_multipliers])
        @ np.array([np.real(power_flow_solution_initial.der_power_vector)])
    )
    der_power_vector_reactive.loc[:, :] = (
        np.transpose([power_multipliers])
        @ np.array([np.imag(power_flow_solution_initial.der_power_vector)])
    )
    der_power_vector_active_change.loc[:, :] = (
        np.transpose([power_multipliers - 1])
        @ np.array([np.real(power_flow_solution_initial.der_power_vector)])
    )
    der_power_vector_reactive_change.loc[:, :] = (
        np.transpose([power_multipliers - 1])
        @ np.array([np.imag(power_flow_solution_initial.der_power_vector)])
    )

    # Obtain power flow solutions.
    power_flow_solutions = (
        fledge.utils.starmap(
            fledge.electric_grid_models.PowerFlowSolutionFixedPoint,
            [(electric_grid_model, row) for row in (der_power_vector_active + 1.0j * der_power_vector_reactive).values]
        )
    )
    power_flow_solutions = dict(zip(power_multipliers, power_flow_solutions))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions[power_multiplier]
        node_voltage_vector_magnitude_power_flow.loc[power_multiplier, :] = np.abs(power_flow_solution.node_voltage_vector)

    # Obtain linear model solutions.
    node_voltage_vector_magnitude_linear_model.loc[:, :] = (
        np.transpose([np.abs(power_flow_solution_initial.node_voltage_vector)] * len(power_multipliers))
        + sensitivity_voltage_magnitude_by_der_power_active @ np.transpose(der_power_vector_active_change.values)
        + sensitivity_voltage_magnitude_by_der_power_reactive @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()

    # Instantiate error variables.
    node_voltage_vector_magnitude_error = (
        pd.Series(index=power_multipliers, dtype=np.float)
    )

    # Obtain error values.
    node_voltage_vector_magnitude_error = (
        100.0 * (
            (node_voltage_vector_magnitude_linear_model - node_voltage_vector_magnitude_power_flow)
            / node_voltage_vector_magnitude_power_flow
        ).mean(axis='columns')
    )

    # Obtain error table.
    linear_electric_grid_model_error = (
        pd.DataFrame(
            [
                node_voltage_vector_magnitude_error,
            ],
            index=[
                'node_voltage_vector_magnitude_error',
            ]
        )
    )
    linear_electric_grid_model_error = linear_electric_grid_model_error.round(2)

    # Print results.
    print(f"der_power_vector_active =\n{der_power_vector_active}")
    print(f"der_power_vector_reactive =\n{der_power_vector_reactive}")
    print(f"der_power_vector_active_change =\n{der_power_vector_active_change}")
    print(f"der_power_vector_reactive_change =\n{der_power_vector_reactive_change}")
    print(f"node_voltage_vector_magnitude_power_flow =\n{node_voltage_vector_magnitude_power_flow}")
    print(f"node_voltage_vector_magnitude_linear_model =\n{node_voltage_vector_magnitude_linear_model}")
    print(f"linear_electric_grid_model_error =\n{linear_electric_grid_model_error}")

    # Store results as CSV.
    der_power_vector_active.to_csv(os.path.join(results_path, 'der_power_vector_active.csv'))
    der_power_vector_reactive.to_csv(os.path.join(results_path, 'der_power_vector_reactive.csv'))
    der_power_vector_active_change.to_csv(os.path.join(results_path, 'der_power_vector_active_change.csv'))
    der_power_vector_reactive_change.to_csv(os.path.join(results_path, 'der_power_vector_reactive_change.csv'))
    node_voltage_vector_magnitude_power_flow.to_csv(os.path.join(results_path, 'node_voltage_vector_magnitude_power_flow.csv'))
    node_voltage_vector_magnitude_linear_model.to_csv(os.path.join(results_path, 'node_voltage_vector_magnitude_linear_model.csv'))
    linear_electric_grid_model_error.to_csv(os.path.join(results_path, 'linear_electric_grid_model_error.csv'))

    # Plot results.

    # Voltage magnitude.
    for node_index, node in enumerate(electric_grid_model.nodes):
        plt.plot(power_multipliers, node_voltage_vector_magnitude_power_flow.loc[:, node], label='Power flow')
        plt.plot(power_multipliers, node_voltage_vector_magnitude_linear_model.loc[:, node], label='Linear model')
        plt.scatter([0.0], [abs(electric_grid_model.node_voltage_vector_reference[node_index])], label='No load')
        plt.scatter([1.0], [abs(power_flow_solution_initial.node_voltage_vector[node_index])], label='Initial point')
        plt.legend()
        plt.title(f"Voltage magnitude [V] for\n (node_type, node_name, phase): {node}")
        plt.savefig(os.path.join(results_path, f'voltage_magnitude_{node}.png'))
        # plt.show()
        plt.close()

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
