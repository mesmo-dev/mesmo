"""Project SITEM linear model setup script."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import scipy.sparse
import re

import fledge.data_interface
import fledge.electric_grid_models
import fledge.problems
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_all'
    results_path = fledge.utils.get_results_path('run_sitem_linear_model_setup', scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain linear model.
    linear_electric_grid_model = fledge.electric_grid_models.LinearElectricGridModelGlobal(scenario_name)

    # Obtain EV charging station incidence matrix.
    # - Dimension of EV charger incidence matrix: (nodes, EV chargers)
    # - There is no EV charging station at the source / reference / slack node, which therefore is sliced off.
    # - The incidence matrix assumes EV charger index order according to `ev_charger_index` given below.
    ev_charger_incidence_matrix = (
        scipy.sparse.eye(len(linear_electric_grid_model.electric_grid_model.nodes)).tocsr()
    )
    ev_charger_incidence_matrix = (
        ev_charger_incidence_matrix[
            :,
            fledge.utils.get_index(linear_electric_grid_model.electric_grid_model.nodes, node_type='no_source')
        ]
    )
    ev_charger_index = (
        linear_electric_grid_model.electric_grid_model.nodes[
            fledge.utils.get_index(linear_electric_grid_model.electric_grid_model.nodes, node_type='no_source')
        ].get_level_values('node_name')
    )

    # Obtain linear model matrices.
    # - Multiplying EV charger incidence directly into the sensitivity matrices for convenience.
    # - Assuming that EV chargers are balanced / wye-connected and only consume active power.
    # - Using only branch power direction "1" / "from source node", which is sufficient if there is no reverse flow.
    # - Branches are all lines and transformers.
    # - Dimension of branch power sensitivity matrix: (branches, EV chargers)
    # - Dimension of voltage sensitivity matrix: (nodes, EV chargers)
    sensitivity_branch_power_magnitude_by_ev_charger_power = (
        linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_power_wye_active
        @ ev_charger_incidence_matrix
    )
    sensitivity_voltage_magnitude_by_ev_charger_power = (
        linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_wye_active
        @ ev_charger_incidence_matrix
    )

    # Obtain reference vectors.
    # - Defines the reference branch power / voltage at nominal loading conditions without EV charging.
    branch_power_magnitude_reference = (
        np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1)
    )
    voltage_magnitude_reference = (
        np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector)
    )

    # Obtain reference vectors as timeseries.
    # - Defines the reference branch power / voltage depending on the load for each time of the day.
    # - Based on the current scenario definition, the first timestep is 12am and the interval is 1h.
    nominal_operation_problem = fledge.problems.NominalOperationProblem(scenario_name)
    nominal_operation_problem.solve()
    power_flow_solution_timeseries = nominal_operation_problem.get_results()
    branch_power_magnitude_reference_timeseries = (
        np.abs(power_flow_solution_timeseries['branch_power_vector_1'])
    )
    voltage_magnitude_reference_timeseries = (
        np.abs(power_flow_solution_timeseries['node_voltage_vector'])
    )

    # Obtain constraint vectors.
    # - Assuming that lines / transformers can be loaded to 105% and voltage can drop to 95%.
    # - In the synthetic grid, some of the 66/22kV transformers are already loaded up to 104% with nominal load.
    branch_power_vector_magnitude_maximum = (
        1.05
        * linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference
    )
    voltage_vector_magnitude_minimum = (
        0.95
        * linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference
    )

    # Store to CSVs.
    np.savetxt(
        os.path.join(results_path, 'ev_charger_incidence_matrix.csv'),
        ev_charger_incidence_matrix.toarray(),
        delimiter=','
    )
    ev_charger_index.to_frame().to_csv(os.path.join(results_path, 'ev_charger_index.csv'), header=False, index=False)
    np.savetxt(
        os.path.join(results_path, 'sensitivity_branch_power_magnitude_by_ev_charger_power.csv'),
        sensitivity_branch_power_magnitude_by_ev_charger_power.toarray(),
        delimiter=','
    )
    np.savetxt(
        os.path.join(results_path, 'sensitivity_voltage_magnitude_by_ev_charger_power.csv'),
        sensitivity_voltage_magnitude_by_ev_charger_power.toarray(),
        delimiter=','
    )
    np.savetxt(
        os.path.join(results_path, 'branch_power_magnitude_reference.csv'),
        branch_power_magnitude_reference,
        delimiter=','
    )
    np.savetxt(
        os.path.join(results_path, 'voltage_magnitude_reference.csv'),
        voltage_magnitude_reference,
        delimiter=','
    )
    np.savetxt(
        os.path.join(results_path, 'branch_power_magnitude_reference_timeseries.csv'),
        branch_power_magnitude_reference_timeseries,
        delimiter=','
    )
    np.savetxt(
        os.path.join(results_path, 'voltage_magnitude_reference_timeseries.csv'),
        voltage_magnitude_reference_timeseries,
        delimiter=','
    )
    np.savetxt(
        os.path.join(results_path, 'branch_power_vector_magnitude_maximum.csv'),
        branch_power_vector_magnitude_maximum,
        delimiter=','
    )
    np.savetxt(
        os.path.join(results_path, 'voltage_vector_magnitude_minimum.csv'),
        voltage_vector_magnitude_minimum,
        delimiter=','
    )

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
