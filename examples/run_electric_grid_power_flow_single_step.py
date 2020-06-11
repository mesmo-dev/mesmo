"""Example script for setting up and solving an single step electric grid power flow problem."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import fledge.data_interface
import fledge.electric_grid_models
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    results_path = fledge.utils.get_results_path('run_electric_grid_power_flow_single_step', scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain electric grid model.
    # - The ElectricGridModelDefault object defines index sets for node names / branch names / der names / phases /
    #   node types / branch types, the nodal admittance / transformation matrices, branch admittance /
    #   incidence matrices, DER incidence matrices and no load voltage vector as well as nominal power vector.
    # - The model is created for the electric grid which is defined for the given scenario in the data directory.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

    # Obtain the nominal DER power vector.
    der_power_vector_nominal = electric_grid_model.der_power_vector_nominal

    # Obtain power flow solution.
    # - The PowerFlowSolutionFixedPoint obtains the solution for nodal voltage vector / branch power vector
    #   and total loss (all complex valued) for the given DER power vector.
    power_flow_solution = (
        fledge.electric_grid_models.PowerFlowSolutionFixedPoint(
            electric_grid_model,
            der_power_vector_nominal
        )
    )

    # Obtain results (as numpy arrays).
    der_power_vector = power_flow_solution.der_power_vector  # The DER power vector is also stored in the solution.
    node_voltage_vector = power_flow_solution.node_voltage_vector
    branch_power_vector_1 = power_flow_solution.branch_power_vector_1
    branch_power_vector_2 = power_flow_solution.branch_power_vector_2
    loss = power_flow_solution.loss

    # Print results.
    print(f"der_power_vector = \n{der_power_vector}")
    print(f"node_voltage_vector = \n{node_voltage_vector}")
    print(f"branch_power_vector_1 = \n{branch_power_vector_1}")
    print(f"branch_power_vector_2 = \n{branch_power_vector_2}")
    print(f"loss = {loss}")

    # Save results to CSV.
    np.savetxt(os.path.join(results_path, f'der_power_vector.csv'), der_power_vector, delimiter=',')
    np.savetxt(os.path.join(results_path, f'node_voltage_vector.csv'), node_voltage_vector, delimiter=',')
    np.savetxt(os.path.join(results_path, f'branch_power_vector_1.csv'), branch_power_vector_1, delimiter=',')
    np.savetxt(os.path.join(results_path, f'branch_power_vector_2.csv'), branch_power_vector_2, delimiter=',')
    np.savetxt(os.path.join(results_path, f'loss.csv'), loss, delimiter=',')

    # Plot some results.
    plt.title('DER active power [kW]')
    plt.bar(range(len(electric_grid_model.ders)), np.real(der_power_vector) / 1e3)
    plt.xticks(range(len(electric_grid_model.ders)), electric_grid_model.ders, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    plt.show()
    plt.title('DER reactive power [kVAr]')
    plt.bar(range(len(electric_grid_model.ders)), np.imag(der_power_vector) / 1e3)
    plt.xticks(range(len(electric_grid_model.ders)), electric_grid_model.ders, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    plt.show()
    plt.title('Nodal voltage magnitude [kV]')
    plt.bar(range(len(electric_grid_model.nodes)), np.abs(node_voltage_vector.ravel()) / 1e3)
    plt.xticks(range(len(electric_grid_model.nodes)), electric_grid_model.nodes, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    plt.show()
    plt.title('Branch apparent power flow (direction 1) [kVA]')
    plt.bar(range(len(electric_grid_model.branches)), np.abs(branch_power_vector_1.ravel()) / 1e3)
    plt.xticks(range(len(electric_grid_model.branches)), electric_grid_model.branches, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    plt.show()

    # Print results path.
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
