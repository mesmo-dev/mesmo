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
    power_multipliers = np.arange(-0.2, 1.2, 0.1)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain electric grid models.
    electric_grid_model_default = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    electric_grid_model_opendss = fledge.electric_grid_models.ElectricGridModelOpenDSS(scenario_name)

    # Obtain nominal power flow solutions.
    power_flow_solution_fledge_nominal = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model_default)
    power_flow_solution_opendss_nominal = fledge.electric_grid_models.PowerFlowSolutionOpenDSS(electric_grid_model_opendss)

    # Instantiate results variables.
    der_power_vector = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.ders, dtype=np.float)
    )
    node_voltage_vector_fledge = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.nodes, dtype=np.complex)
    )
    node_voltage_vector_opendss = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.nodes, dtype=np.complex)
    )
    node_voltage_vector_magnitude_fledge = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.nodes, dtype=np.float)
    )
    node_voltage_vector_magnitude_opendss = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.nodes, dtype=np.float)
    )
    branch_power_vector_1_magnitude_fledge = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.branches, dtype=np.float)
    )
    branch_power_vector_1_magnitude_opendss = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.branches, dtype=np.float)
    )
    branch_power_vector_2_magnitude_fledge = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.branches, dtype=np.float)
    )
    branch_power_vector_2_magnitude_opendss = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.branches, dtype=np.float)
    )
    loss_active_fledge = (
        pd.Series(index=power_multipliers, dtype=np.float)
    )
    loss_active_opendss = (
        pd.Series(index=power_multipliers, dtype=np.float)
    )
    loss_reactive_fledge = (
        pd.Series(index=power_multipliers, dtype=np.float)
    )
    loss_reactive_opendss = (
        pd.Series(index=power_multipliers, dtype=np.float)
    )

    # Obtain DER power / change.
    der_power_vector.loc[:, :] = (
        np.transpose([power_multipliers])
        @ np.array([np.real(electric_grid_model_default.der_power_vector_reference)])
    )

    # Obtain OpenDSS solutions.
    power_flow_solutions_fledge = (
        fledge.utils.starmap(
            fledge.electric_grid_models.PowerFlowSolutionFixedPoint,
            [(electric_grid_model_default, row) for row in der_power_vector.values]
        )
    )
    power_flow_solutions_fledge = dict(zip(power_multipliers, power_flow_solutions_fledge))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions_fledge[power_multiplier]
        node_voltage_vector_fledge.loc[power_multiplier, :] = power_flow_solution.node_voltage_vector
        node_voltage_vector_magnitude_fledge.loc[power_multiplier, :] = np.abs(power_flow_solution.node_voltage_vector)
        branch_power_vector_1_magnitude_fledge.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_1)
        branch_power_vector_2_magnitude_fledge.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_2)
        loss_active_fledge.loc[power_multiplier] = np.real(power_flow_solution.loss)
        loss_reactive_fledge.loc[power_multiplier] = np.imag(power_flow_solution.loss)

    power_flow_solutions_opendss = (
        fledge.utils.starmap(
            fledge.electric_grid_models.PowerFlowSolutionOpenDSS,
            [(electric_grid_model_opendss, row) for row in der_power_vector.values]
        )
    )
    power_flow_solutions_opendss = dict(zip(power_multipliers, power_flow_solutions_opendss))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions_opendss[power_multiplier]
        node_voltage_vector_opendss.loc[power_multiplier, :] = power_flow_solution.node_voltage_vector
        node_voltage_vector_magnitude_opendss.loc[power_multiplier, :] = np.abs(power_flow_solution.node_voltage_vector)
        branch_power_vector_1_magnitude_opendss.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_1)
        branch_power_vector_2_magnitude_opendss.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_2)
        loss_active_opendss.loc[power_multiplier] = np.real(power_flow_solution.loss)
        loss_reactive_opendss.loc[power_multiplier] = np.imag(power_flow_solution.loss)

    # Obtain error values.
    node_voltage_vector_error = (
        100.0 * (
            (node_voltage_vector_fledge - node_voltage_vector_opendss)
            / node_voltage_vector_opendss
        ).abs().mean(axis='columns')
    )
    node_voltage_vector_magnitude_error = (
        100.0 * (
            (node_voltage_vector_magnitude_fledge - node_voltage_vector_magnitude_opendss)
            / node_voltage_vector_magnitude_opendss
        ).mean(axis='columns')
    )
    branch_power_vector_1_magnitude_error = (
        100.0 * (
            (branch_power_vector_1_magnitude_fledge - branch_power_vector_1_magnitude_opendss)
            / branch_power_vector_1_magnitude_opendss
        ).mean(axis='columns')
    )
    branch_power_vector_2_magnitude_error = (
        100.0 * (
            (branch_power_vector_2_magnitude_fledge - branch_power_vector_2_magnitude_opendss)
            / branch_power_vector_2_magnitude_opendss
        ).mean(axis='columns')
    )
    loss_active_error = (
        100.0 * (
            (loss_active_fledge - loss_active_opendss)
            / loss_active_opendss
        )
    )
    loss_reactive_error = (
        100.0 * (
            (loss_reactive_fledge - loss_reactive_opendss)
            / loss_reactive_opendss
        )
    )

    # Obtain error table.
    power_flow_solution_error = (
        pd.DataFrame(
            [
                node_voltage_vector_error,
                node_voltage_vector_magnitude_error,
                branch_power_vector_1_magnitude_error,
                branch_power_vector_2_magnitude_error,
                loss_active_error,
                loss_reactive_error
            ],
            index=[
                'node_voltage_vector_error',
                'node_voltage_vector_magnitude_error',
                'branch_power_vector_1_magnitude_error',
                'branch_power_vector_2_magnitude_error',
                'loss_active_error',
                'loss_reactive_error'
            ]
        )
    )
    power_flow_solution_error = power_flow_solution_error.round(2)

    # Compile results.
    results = fledge.data_interface.ResultsDict(
        der_power_vector=der_power_vector,
        node_voltage_vector_fledge=node_voltage_vector_fledge,
        node_voltage_vector_opendss=node_voltage_vector_opendss,
        node_voltage_vector_magnitude_fledge=node_voltage_vector_magnitude_fledge,
        node_voltage_vector_magnitude_opendss=node_voltage_vector_magnitude_opendss,
        branch_power_vector_1_magnitude_fledge=branch_power_vector_1_magnitude_fledge,
        branch_power_vector_1_magnitude_opendss=branch_power_vector_1_magnitude_opendss,
        branch_power_vector_2_magnitude_fledge=branch_power_vector_2_magnitude_fledge,
        branch_power_vector_2_magnitude_opendss=branch_power_vector_2_magnitude_opendss,
        loss_active_fledge=loss_active_fledge,
        loss_active_opendss=loss_active_opendss,
        loss_reactive_fledge=loss_reactive_fledge,
        loss_reactive_opendss=loss_reactive_opendss,
        power_flow_solution_error=power_flow_solution_error
    )

    # Print results.
    print(results)

    # Store results as CSV.
    results.to_csv(results_path)

    # Plot results.

    # Nominal OpenDSS solution comparison.
    plt.title('Node voltage magnitude [p.u.]')
    plt.scatter(
        range(len(electric_grid_model_default.nodes)),
        (
            np.abs(power_flow_solution_fledge_nominal.node_voltage_vector)
            / np.abs(electric_grid_model_default.node_voltage_vector_reference)
        ),
        marker=4
    )
    plt.scatter(
        range(len(electric_grid_model_default.nodes)),
        (
            np.abs(power_flow_solution_opendss_nominal.node_voltage_vector)
            / np.abs(electric_grid_model_default.node_voltage_vector_reference)
        ),
        marker=5
    )
    plt.xticks(
        range(len(electric_grid_model_default.nodes)),
        electric_grid_model_default.nodes,
        rotation=45,
        ha='right'
    )
    plt.legend(['FLEDGE', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'nominal_voltage_magnitude.png'))
    # plt.show()
    plt.close()

    plt.title('Branch power (direction 1) magnitude [p.u.]')
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        (
            np.abs(power_flow_solution_fledge_nominal.branch_power_vector_1)
            / electric_grid_model_default.branch_power_vector_magnitude_reference
        ),
        marker=4
    )
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        (
            np.abs(power_flow_solution_opendss_nominal.branch_power_vector_1)
            / electric_grid_model_default.branch_power_vector_magnitude_reference
        ),
        marker=5
    )
    plt.xticks(
        range(len(electric_grid_model_default.branches)),
        electric_grid_model_default.branches,
        rotation=45,
        ha='right'
    )
    plt.legend(['FLEDGE', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'nominal_branch_power_1_magnitude.png'))
    # plt.show()
    plt.close()

    plt.title('Branch power (direction 2) magnitude [p.u.]')
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        (
            np.abs(power_flow_solution_fledge_nominal.branch_power_vector_2)
            / electric_grid_model_default.branch_power_vector_magnitude_reference
        ),
        marker=4
    )
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        (
            np.abs(power_flow_solution_opendss_nominal.branch_power_vector_2)
            / electric_grid_model_default.branch_power_vector_magnitude_reference
        ),
        marker=5
    )
    plt.xticks(
        range(len(electric_grid_model_default.branches)),
        electric_grid_model_default.branches,
        rotation=45,
        ha='right'
    )
    plt.legend(['FLEDGE', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'nominal_branch_power_2_magnitude.png'))
    # plt.show()
    plt.close()

    plt.title('Total power loss [VA]')
    plt.scatter(
        [1, 2],
        [
            np.real(power_flow_solution_fledge_nominal.loss),
            np.imag(power_flow_solution_fledge_nominal.loss)
        ],
        marker=4
    )
    plt.scatter(
        [1, 2],
        [
            np.real(power_flow_solution_opendss_nominal.loss),
            np.imag(power_flow_solution_opendss_nominal.loss)
        ],
        marker=5
    )
    plt.xticks(
        [0.5, 1, 2, 2.5],
        ['', 'active', 'reactive', ''],
        rotation=45,
        ha='right'
    )
    plt.legend(['FLEDGE', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'nominal_loss.png'))
    # plt.show()
    plt.close()

    # Voltage magnitude.
    for node_index, node in enumerate(electric_grid_model_default.nodes):
        plt.plot(power_multipliers, node_voltage_vector_magnitude_opendss.loc[:, node], label='OpenDSS')
        plt.plot(power_multipliers, node_voltage_vector_magnitude_fledge.loc[:, node], label='FLEDGE')
        plt.legend()
        plt.title(f"Voltage magnitude [V] for\n (node_type, node_name, phase): {node}")
        plt.savefig(os.path.join(results_path, f'voltage_magnitude_{node}.png'))
        # plt.show()
        plt.close()

    # Branch flow.
    for branch_index, branch in enumerate(electric_grid_model_default.branches):
        plt.plot(power_multipliers, branch_power_vector_1_magnitude_opendss.loc[:, branch], label='OpenDSS')
        plt.plot(power_multipliers, branch_power_vector_1_magnitude_fledge.loc[:, branch], label='FLEDGE')
        plt.legend()
        plt.title(f"Branch power 1 magnitude [VA] for\n (branch_type, branch_name, phase): {branch}")
        plt.savefig(os.path.join(results_path, f'branch_power_1_{branch}.png'))
        # plt.show()
        plt.close()

        plt.plot(power_multipliers, branch_power_vector_2_magnitude_opendss.loc[:, branch], label='OpenDSS')
        plt.plot(power_multipliers, branch_power_vector_2_magnitude_fledge.loc[:, branch], label='FLEDGE')
        plt.legend()
        plt.title(f"Branch power 2 magnitude [VA] for\n (branch_type, branch_name, phase): {branch}")
        plt.savefig(os.path.join(results_path, f'branch_power_2_{branch}.png'))
        # plt.show()
        plt.close()

    # Loss.
    plt.plot(power_multipliers, loss_active_opendss, label='OpenDSS')
    plt.plot(power_multipliers, loss_active_fledge, label='FLEDGE')
    plt.legend()
    plt.title("Total loss active [W]")
    plt.savefig(os.path.join(results_path, f'loss_active.png'))
    # plt.show()
    plt.close()

    plt.plot(power_multipliers, loss_reactive_opendss, label='OpenDSS')
    plt.plot(power_multipliers, loss_reactive_fledge, label='FLEDGE')
    plt.legend()
    plt.title("Total loss reactive [VAr]")
    plt.savefig(os.path.join(results_path, f'loss_reactive.png'))
    # plt.show()
    plt.close()

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
