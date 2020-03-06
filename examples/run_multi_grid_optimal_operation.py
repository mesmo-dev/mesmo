"""Example script for setting up and solving a multi-grid optimal operation problem."""

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

import cobmo.database_interface
import fledge.config
import fledge.database_interface
import fledge.der_models
import fledge.linear_electric_grid_models
import fledge.electric_grid_models
import fledge.power_flow_solvers
import fledge.thermal_grid_models


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'
    results_path = (
        os.path.join(
            fledge.config.results_path,
            f'run_multi_grid_optimal_operation_{fledge.config.timestamp}'
        )
    )

    # Instantiate results directory.
    os.mkdir(results_path)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.database_interface.recreate_database()
    cobmo.database_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.database_interface.ScenarioData(scenario_name)
    price_data = fledge.database_interface.PriceData(scenario_name)

    # Obtain price timeseries.
    price_name = 'energy'
    price_timeseries = price_data.price_timeseries_dict[price_name]

    # Obtain models.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModel(scenario_name)
    power_flow_solution = fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.linear_electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
    thermal_power_flow_solution = fledge.thermal_grid_models.ThermalPowerFlowSolution(thermal_grid_model)
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define linear electric grid model constraints.
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define thermal grid model variables.
    thermal_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define thermal grid model constraints.
    thermal_grid_model.define_optimization_constraints(
        optimization_problem,
        thermal_power_flow_solution,
        scenario_data.timesteps
    )

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem
    )

    # Define constraints for the connection with the DER power vector of the electric and thermal grids.
    der_model_set.define_optimization_connection_grid(
        optimization_problem,
        power_flow_solution,
        electric_grid_model,
        thermal_power_flow_solution,
        thermal_grid_model
    )

    # Define limit constraints.

    # Electric grid.

    # Voltage.
    voltage_magnitude_vector = (  # Define shorthand.
        lambda node:
        np.abs(power_flow_solution.node_voltage_vector.ravel()[electric_grid_model.nodes.get_loc(node)])
    )
    optimization_problem.voltage_magnitude_vector_minimum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        electric_grid_model.nodes.to_list(),
        rule=lambda optimization_problem, timestep, *node: (
            optimization_problem.voltage_magnitude_vector_change[timestep, node]
            + voltage_magnitude_vector(node)
            >=
            0.5 * voltage_magnitude_vector(node)
        )
    )
    optimization_problem.voltage_magnitude_vector_maximum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        electric_grid_model.nodes.to_list(),
        rule=lambda optimization_problem, timestep, *node: (
            optimization_problem.voltage_magnitude_vector_change[timestep, node]
            + voltage_magnitude_vector(node)
            <=
            1.5 * voltage_magnitude_vector(node)
        )
    )

    # Branch flows.
    branch_power_vector_1_squared = (  # Define shorthand.
        lambda branch:
        np.abs(power_flow_solution.branch_power_vector_1.ravel()[electric_grid_model.branches.get_loc(branch)] ** 2)
    )
    optimization_problem.branch_power_vector_1_squared_maximum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        electric_grid_model.branches.to_list(),
        rule=lambda optimization_problem, timestep, *branch: (
            optimization_problem.branch_power_vector_1_squared_change[timestep, branch]
            + branch_power_vector_1_squared(branch)
            <=
            1.5 * branch_power_vector_1_squared(branch)
        )
    )
    branch_power_vector_2_squared = (  # Define shorthand.
        lambda branch:
        np.abs(power_flow_solution.branch_power_vector_2.ravel()[electric_grid_model.branches.get_loc(branch)] ** 2)
    )
    optimization_problem.branch_power_vector_2_squared_maximum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        electric_grid_model.branches.to_list(),
        rule=lambda optimization_problem, timestep, *branch: (
            optimization_problem.branch_power_vector_2_squared_change[timestep, branch]
            + branch_power_vector_2_squared(branch)
            <=
            1.5 * branch_power_vector_2_squared(branch)
        )
    )

    # Thermal grid.

    # Node head.
    node_head_vector = (  # Define shorthand.
        lambda node:
        thermal_power_flow_solution.node_head_vector.ravel()[thermal_grid_model.nodes.get_loc(node)]
    )
    optimization_problem.node_head_vector_minimum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        thermal_grid_model.nodes.to_list(),
        rule=lambda optimization_problem, timestep, *node: (
            optimization_problem.node_head_vector[timestep, node]
            # + node_head_vector(node)
            >=
            1.5 * node_head_vector(node)
        )
    )
    # Branch flow.
    branch_flow_vector = (  # Define shorthand.
        lambda branch:
        thermal_power_flow_solution.branch_flow_vector.ravel()[thermal_grid_model.branches.get_loc(branch)]
    )
    optimization_problem.branch_flow_vector_maximum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        thermal_grid_model.branches.to_list(),
        rule=lambda optimization_problem, timestep, branch: (  # This will not work if `branches` becomes MultiIndex.
            optimization_problem.branch_flow_vector[timestep, branch]
            # + branch_flow_vector(branch)
            <=
            1.5 * branch_flow_vector(branch)
        )
    )

    # Define electric grid objective.
    # TODO: Not considering loss costs due to unrealiable loss model.
    # if optimization_problem.find_component('objective') is None:
    #     optimization_problem.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
    # optimization_problem.objective.expr += (
    #     sum(
    #         price_timeseries.at[timestep, 'price_value']
    #         * (
    #             optimization_problem.loss_active_change[timestep]
    #             + np.sum(np.real(power_flow_solution.loss))
    #         )
    #         for timestep in scenario_data.timesteps
    #     )
    # )

    # Define objective.
    thermal_grid_model.define_optimization_objective(
        optimization_problem,
        thermal_power_flow_solution,
        price_timeseries,
        scenario_data.timesteps
    )

    # Define DER objective.
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_timeseries
    )

    # Solve optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    optimization_solver = pyo.SolverFactory(fledge.config.solver_name)
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")
    # optimization_problem.display()

    # Obtain results.
    (
        der_active_power_vector,
        der_reactive_power_vector,
        voltage_magnitude_vector,
        branch_power_vector_1_squared,
        branch_power_vector_2_squared,
        loss_active,
        loss_reactive
    ) = linear_electric_grid_model.get_optimization_results(
        optimization_problem,
        power_flow_solution,
        scenario_data.timesteps,
        in_per_unit=False,
        with_mean=True
    )
    (
        der_thermal_power_vector,
        node_head_vector,
        branch_flow_vector,
        pump_power
    ) = thermal_grid_model.get_optimization_results(
        optimization_problem,
        thermal_power_flow_solution,
        scenario_data.timesteps,
        in_per_unit=False,
        with_mean=True
    )

    # Print results.
    print(f"der_active_power_vector = \n{der_active_power_vector.to_string()}")
    print(f"der_reactive_power_vector = \n{der_reactive_power_vector.to_string()}")
    print(f"voltage_magnitude_vector = \n{voltage_magnitude_vector.to_string()}")
    print(f"branch_power_vector_1_squared = \n{branch_power_vector_1_squared.to_string()}")
    print(f"branch_power_vector_2_squared = \n{branch_power_vector_2_squared.to_string()}")
    print(f"loss_active = \n{loss_active.to_string()}")
    print(f"loss_reactive = \n{loss_reactive.to_string()}")
    print(f"der_thermal_power_vector = \n{der_thermal_power_vector.to_string()}")
    print(f"node_head_vector = \n{node_head_vector.to_string()}")
    print(f"branch_flow_vector = \n{branch_flow_vector.to_string()}")
    print(f"pump_power = \n{pump_power.to_string()}")

    # Store results as CSV.
    der_active_power_vector.to_csv(os.path.join(results_path, 'der_active_power_vector.csv'))
    der_reactive_power_vector.to_csv(os.path.join(results_path, 'der_reactive_power_vector.csv'))
    voltage_magnitude_vector.to_csv(os.path.join(results_path, 'voltage_magnitude_vector.csv'))
    branch_power_vector_1_squared.to_csv(os.path.join(results_path, 'branch_power_vector_1_squared.csv'))
    branch_power_vector_2_squared.to_csv(os.path.join(results_path, 'branch_power_vector_2_squared.csv'))
    loss_active.to_csv(os.path.join(results_path, 'loss_active.csv'))
    loss_reactive.to_csv(os.path.join(results_path, 'loss_reactive.csv'))
    der_thermal_power_vector.to_csv(os.path.join(results_path, 'der_thermal_power_vector.csv'))
    node_head_vector.to_csv(os.path.join(results_path, 'node_head_vector.csv'))
    branch_flow_vector.to_csv(os.path.join(results_path, 'branch_flow_vector.csv'))
    pump_power.to_csv(os.path.join(results_path, 'pump_power.csv'))

    # Obtain duals.
    voltage_magnitude_vector_minimum_dual = (
        pd.DataFrame(columns=electric_grid_model.nodes, index=scenario_data.timesteps, dtype=np.float)
    )
    voltage_magnitude_vector_maximum_dual = (
        pd.DataFrame(columns=electric_grid_model.nodes, index=scenario_data.timesteps, dtype=np.float)
    )
    branch_power_vector_1_squared_maximum_dual = (
        pd.DataFrame(columns=electric_grid_model.branches, index=scenario_data.timesteps, dtype=np.float)
    )
    branch_power_vector_2_squared_maximum_dual = (
        pd.DataFrame(columns=electric_grid_model.branches, index=scenario_data.timesteps, dtype=np.float)
    )
    node_head_vector_minimum_dual = (
        pd.DataFrame(columns=thermal_grid_model.nodes, index=scenario_data.timesteps, dtype=np.float)
    )
    branch_flow_vector_maximum_dual = (
        pd.DataFrame(columns=thermal_grid_model.branches, index=scenario_data.timesteps, dtype=np.float)
    )

    for timestep in scenario_data.timesteps:

        for node_index, node in enumerate(electric_grid_model.nodes):
            voltage_magnitude_vector_minimum_dual.at[timestep, node] = (
                optimization_problem.dual[
                    optimization_problem.voltage_magnitude_vector_minimum_constraint[timestep, node]
                ]
            )
            voltage_magnitude_vector_maximum_dual.at[timestep, node] = (
                optimization_problem.dual[
                    optimization_problem.voltage_magnitude_vector_maximum_constraint[timestep, node]
                ]
            )

        for branch_index, branch in enumerate(electric_grid_model.branches):
            branch_power_vector_1_squared_maximum_dual.at[timestep, branch] = (
                optimization_problem.dual[
                    optimization_problem.branch_power_vector_1_squared_maximum_constraint[timestep, branch]
                ]
            )
            branch_power_vector_2_squared_maximum_dual.at[timestep, branch] = (
                optimization_problem.dual[
                    optimization_problem.branch_power_vector_2_squared_maximum_constraint[timestep, branch]
                ]
            )

        for node_index, node in enumerate(thermal_grid_model.nodes):
            node_head_vector_minimum_dual.at[timestep, node] = (
                optimization_problem.dual[
                    optimization_problem.node_head_vector_minimum_constraint[timestep, node]
                ]
            )

        for branch_index, branch in enumerate(thermal_grid_model.branches):
            branch_flow_vector_maximum_dual.at[timestep, branch] = (
                optimization_problem.dual[
                    optimization_problem.branch_flow_vector_maximum_constraint[timestep, branch]
                ]
            )

    # Print duals.
    print(f"voltage_magnitude_vector_minimum_dual = \n{voltage_magnitude_vector_minimum_dual.to_string()}")
    print(f"voltage_magnitude_vector_maximum_dual = \n{voltage_magnitude_vector_maximum_dual.to_string()}")
    print(f"branch_power_vector_1_squared_maximum_dual = \n{branch_power_vector_1_squared_maximum_dual.to_string()}")
    print(f"branch_power_vector_2_squared_maximum_dual = \n{branch_power_vector_2_squared_maximum_dual.to_string()}")
    print(f"node_head_vector_minimum_dual = \n{node_head_vector_minimum_dual.to_string()}")
    print(f"branch_flow_vector_maximum_dual = \n{branch_flow_vector_maximum_dual.to_string()}")

    # Obtain DLMPs.
    voltage_magnitude_vector_minimum_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    voltage_magnitude_vector_maximum_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    branch_power_vector_1_squared_maximum_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    branch_power_vector_2_squared_maximum_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    loss_active_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    loss_reactive_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    node_head_vector_minimum_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    branch_flow_vector_maximum_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    pump_power_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )

    electric_grid_energy_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    electric_grid_voltage_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    electric_grid_congestion_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    electric_grid_loss_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    thermal_grid_energy_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    thermal_grid_head_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    thermal_grid_congestion_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    thermal_grid_pump_dlmp = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )

    for timestep in scenario_data.timesteps:
        voltage_magnitude_vector_minimum_dlmp.loc[timestep, :] = (
            (
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active.transpose()
                @ np.transpose([voltage_magnitude_vector_minimum_dual.loc[timestep, :].values])
            ).ravel()
        )
        voltage_magnitude_vector_maximum_dlmp.loc[timestep, :] = (
            (
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active.transpose()
                @ np.transpose([voltage_magnitude_vector_maximum_dual.loc[timestep, :].values])
            ).ravel()
        )
        branch_power_vector_1_squared_maximum_dlmp.loc[timestep, :] = (
            (
                linear_electric_grid_model.sensitivity_branch_power_1_by_der_power_active.transpose()
                @ np.transpose([branch_power_vector_1_squared_maximum_dual.loc[timestep, :].values])
            ).ravel()
        )
        branch_power_vector_2_squared_maximum_dlmp.loc[timestep, :] = (
            (
                linear_electric_grid_model.sensitivity_branch_power_2_by_der_power_active.transpose()
                @ np.transpose([branch_power_vector_2_squared_maximum_dual.loc[timestep, :].values])
            ).ravel()
        )
        loss_active_dlmp.loc[timestep, :] = (
            linear_electric_grid_model.sensitivity_loss_active_by_der_power_active.ravel()
            * price_timeseries.at[timestep, 'price_value']
        )
        loss_reactive_dlmp.loc[timestep, :] = (
            -1.0
            * linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active.ravel()
            * price_timeseries.at[timestep, 'price_value']
        )
        node_head_vector_minimum_dlmp.loc[timestep, :] = (
            (
                thermal_grid_model.sensitivity_node_head_by_der_power.transpose()
                @ np.transpose([node_head_vector_minimum_dual.loc[timestep, :].values])
            ).ravel()
            / thermal_grid_model.cooling_plant_efficiency
        )
        branch_flow_vector_maximum_dlmp.loc[timestep, :] = (
            (
                thermal_grid_model.sensitivity_branch_flow_by_der_power.transpose()
                @ np.transpose([branch_flow_vector_maximum_dual.loc[timestep, :].values])
            ).ravel()
            / thermal_grid_model.cooling_plant_efficiency
        )
        pump_power_dlmp.loc[timestep, :] = (
            -1.0
            * thermal_grid_model.sensitivity_pump_power_by_der_power.ravel()
            * price_timeseries.at[timestep, 'price_value']
        )

        electric_grid_energy_dlmp.loc[timestep, :] = (
            price_timeseries.at[timestep, 'price_value']
        )
        thermal_grid_energy_dlmp.loc[timestep, :] = (
            price_timeseries.at[timestep, 'price_value']
            / thermal_grid_model.cooling_plant_efficiency
        )
    electric_grid_voltage_dlmp = (
        voltage_magnitude_vector_minimum_dlmp
        + voltage_magnitude_vector_maximum_dlmp
    )
    electric_grid_congestion_dlmp = (
        branch_power_vector_1_squared_maximum_dlmp
        + branch_power_vector_2_squared_maximum_dlmp
    )
    electric_grid_loss_dlmp = (
        loss_active_dlmp
        + loss_reactive_dlmp
    )
    thermal_grid_head_dlmp = (
        node_head_vector_minimum_dlmp
    )
    thermal_grid_congestion_dlmp = (
        branch_flow_vector_maximum_dlmp
    )
    thermal_grid_pump_dlmp = (
        pump_power_dlmp
    )

    # Print DLMPs.
    print(f"voltage_magnitude_vector_minimum_dlmp = \n{voltage_magnitude_vector_minimum_dlmp.to_string()}")
    print(f"voltage_magnitude_vector_maximum_dlmp = \n{voltage_magnitude_vector_maximum_dlmp.to_string()}")
    print(f"branch_power_vector_1_squared_maximum_dlmp = \n{branch_power_vector_1_squared_maximum_dlmp.to_string()}")
    print(f"branch_power_vector_2_squared_maximum_dlmp = \n{branch_power_vector_2_squared_maximum_dlmp.to_string()}")
    print(f"loss_active_dlmp = \n{loss_active_dlmp.to_string()}")
    print(f"loss_reactive_dlmp = \n{loss_reactive_dlmp.to_string()}")
    print(f"node_head_vector_minimum_dlmp = \n{node_head_vector_minimum_dlmp.to_string()}")
    print(f"branch_flow_vector_maximum_dlmp = \n{branch_flow_vector_maximum_dlmp.to_string()}")
    print(f"pump_power_dlmp = \n{pump_power_dlmp.to_string()}")

    print(f"electric_grid_energy_dlmp = \n{electric_grid_energy_dlmp.to_string()}")
    print(f"electric_grid_voltage_dlmp = \n{electric_grid_voltage_dlmp.to_string()}")
    print(f"electric_grid_congestion_dlmp = \n{electric_grid_congestion_dlmp.to_string()}")
    print(f"electric_grid_loss_dlmp = \n{electric_grid_loss_dlmp.to_string()}")
    print(f"thermal_grid_energy_dlmp = \n{thermal_grid_energy_dlmp.to_string()}")
    print(f"thermal_grid_head_dlmp = \n{thermal_grid_head_dlmp.to_string()}")
    print(f"thermal_grid_congestion_dlmp = \n{thermal_grid_congestion_dlmp.to_string()}")
    print(f"thermal_grid_pump_dlmp = \n{thermal_grid_pump_dlmp.to_string()}")

    # Obtain complete DLMPs.
    electric_grid_dlmp = (
        pd.concat(
            [
                electric_grid_energy_dlmp,
                electric_grid_loss_dlmp,
                electric_grid_voltage_dlmp,
                electric_grid_congestion_dlmp
            ],
            axis='columns',
            keys=['energy', 'loss', 'voltage', 'congestion'],
            names=['dlmp_type']
        )
    )
    thermal_grid_dlmp = (
        pd.concat(
            [
                thermal_grid_energy_dlmp,
                thermal_grid_pump_dlmp,
                thermal_grid_head_dlmp,
                thermal_grid_congestion_dlmp
            ],
            axis='columns',
            keys=['energy', 'pump', 'head', 'congestion'],
            names=['dlmp_type']
        )
    )

    # Plot thermal grid DLMPs.
    colors = list(color['color'] for color in matplotlib.rcParams['axes.prop_cycle'])
    for der in thermal_grid_model.ders:
        fig, (ax1, lax) = plt.subplots(ncols=2, figsize=[7.8, 2.6], gridspec_kw={"width_ratios": [100, 1]})
        ax1.set_title(f'Flexible building "{der[1]}"')
        ax1.stackplot(
            scenario_data.timesteps,
            thermal_grid_dlmp.loc[:, (slice(None), *der)].droplevel(['der_type', 'der_name'], axis='columns').T,
            labels=['Energy', 'Pumping', 'Head', 'Congest.'],
            colors=[colors[0], colors[1], colors[2], colors[3]]
        )
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price [S$/MWh]')
        # ax1.set_ylim((0.0, 10.0))
        ax2 = plt.twinx(ax1)
        ax2.plot(
            der_thermal_power_vector.loc[:, der].abs() / 1000000,
            label='Thrm. pw.',
            drawstyle='steps-post',
            color='darkgrey',
            linewidth=3
        )
        ax2.plot(
            der_active_power_vector.loc[:, der].abs() / 1000000,
            label='Active pw.',
            drawstyle='steps-post',
            color='black',
            linewidth=1.5
        )
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_xlim((scenario_data.timesteps[0], scenario_data.timesteps[-1]))
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Power [MW]')
        ax2.set_ylim((0.0, 20.0))  # TODO: Document modifications for Thermal Electric DLMP paper
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        lax.legend((*h1, *h2), (*l1, *l2), borderaxespad=0)
        lax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'thermal_grid_dlmp_{der}.pdf'))
        plt.close()


if __name__ == '__main__':
    main()
