"""Run script for reproducing results of the Paper: 'Distribution Locational Marginal Pricing for Combined Thermal
and Electric Grid Operation', available at: <https://doi.org/10.36227/techrxiv.11918712.v1>.
"""

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
import fledge.electric_grid_models
import fledge.thermal_grid_models


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'
    scenario = 1  # Choices: 1 (unconstrained operation), 2 (constrained branch flow), 3 (constrained pressure head).
    results_path = (
        os.path.join(
            fledge.config.config['paths']['results'],
            f'paper_2020_dlmp_combined_thermal_electric_scenario_{scenario}_{fledge.config.get_timestamp()}'
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
    price_type = 'singapore_wholesale'
    price_timeseries = price_data.price_timeseries_dict[price_type]

    # Obtain models.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
    thermal_grid_model.ets_head_loss = 0.0  # TODO: Document modifications for Thermal Electric DLMP paper
    thermal_grid_model.cooling_plant_efficiency = 10.0  # TODO: Document modifications for Thermal Electric DLMP paper
    thermal_power_flow_solution = fledge.thermal_grid_models.ThermalPowerFlowSolution(thermal_grid_model)
    linear_thermal_grid_model = (
        fledge.thermal_grid_models.LinearThermalGridModel(
            thermal_grid_model,
            thermal_power_flow_solution
        )
    )
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define linear electric grid model constraints.
    voltage_magnitude_vector_minimum = 0.5 * np.abs(power_flow_solution.node_voltage_vector)
    voltage_magnitude_vector_maximum = 1.5 * np.abs(power_flow_solution.node_voltage_vector)
    branch_power_vector_squared_maximum = 1.5 * np.abs(power_flow_solution.branch_power_vector_1 ** 2)
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps,
        voltage_magnitude_vector_minimum=voltage_magnitude_vector_minimum,
        voltage_magnitude_vector_maximum=voltage_magnitude_vector_maximum,
        branch_power_vector_squared_maximum=branch_power_vector_squared_maximum
    )

    # Define thermal grid model variables.
    linear_thermal_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define thermal grid model constraints.
    node_head_vector_minimum = 1.5 * thermal_power_flow_solution.node_head_vector
    branch_flow_vector_maximum = 1.5 * thermal_power_flow_solution.branch_flow_vector
    # Modify limits for scenarios.
    if scenario == 1:
        pass
    elif scenario == 2:
        branch_flow_vector_maximum[thermal_grid_model.branches.get_loc('4')] *= 0.1 / 1.5
    elif scenario == 3:
        node_head_vector_minimum[thermal_grid_model.nodes.get_loc(('no_source', '15'))] *= 0.1 / 1.5
    else:
        ValueError(f"Invalid scenario: {scenario}")
    linear_thermal_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps,
        node_head_vector_minimum=node_head_vector_minimum,
        branch_flow_vector_maximum=branch_flow_vector_maximum
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

    # Define objective.
    linear_thermal_grid_model.define_optimization_objective(
        optimization_problem,
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
    optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.config['optimization']['show_solver_output'])
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")
    # optimization_problem.display()

    # Obtain results.
    results = (
        linear_electric_grid_model.get_optimization_results(
            optimization_problem,
            power_flow_solution,
            scenario_data.timesteps,
            in_per_unit=False,
            with_mean=True
        )
    )
    results.update(
        linear_thermal_grid_model.get_optimization_results(
            optimization_problem,
            scenario_data.timesteps,
            in_per_unit=False,
            with_mean=True
        )
    )

    # Print results.
    print(results)

    # Store results as CSV.
    results.to_csv(results_path)

    # Obtain DLMPs.
    (
        voltage_magnitude_vector_minimum_dlmp,
        voltage_magnitude_vector_maximum_dlmp,
        branch_power_vector_1_squared_maximum_dlmp,
        branch_power_vector_2_squared_maximum_dlmp,
        loss_active_dlmp,
        loss_reactive_dlmp,
        electric_grid_energy_dlmp,
        electric_grid_voltage_dlmp,
        electric_grid_congestion_dlmp,
        electric_grid_loss_dlmp
    ) = linear_electric_grid_model.get_optimization_dlmps(
        optimization_problem,
        price_timeseries,
        scenario_data.timesteps
    )
    (
        node_head_vector_minimum_dlmp,
        branch_flow_vector_maximum_dlmp,
        pump_power_dlmp,
        thermal_grid_energy_dlmp,
        thermal_grid_head_dlmp,
        thermal_grid_congestion_dlmp,
        thermal_grid_pump_dlmp
    ) = linear_thermal_grid_model.get_optimization_dlmps(
        optimization_problem,
        price_timeseries,
        scenario_data.timesteps
    )

    # Print DLMPs.
    print(f"voltage_magnitude_vector_minimum_dlmp = \n{voltage_magnitude_vector_minimum_dlmp}")
    print(f"voltage_magnitude_vector_maximum_dlmp = \n{voltage_magnitude_vector_maximum_dlmp}")
    print(f"branch_power_vector_1_squared_maximum_dlmp = \n{branch_power_vector_1_squared_maximum_dlmp}")
    print(f"branch_power_vector_2_squared_maximum_dlmp = \n{branch_power_vector_2_squared_maximum_dlmp}")
    print(f"loss_active_dlmp = \n{loss_active_dlmp}")
    print(f"loss_reactive_dlmp = \n{loss_reactive_dlmp}")
    print(f"node_head_vector_minimum_dlmp = \n{node_head_vector_minimum_dlmp}")
    print(f"branch_flow_vector_maximum_dlmp = \n{branch_flow_vector_maximum_dlmp}")
    print(f"pump_power_dlmp = \n{pump_power_dlmp}")
    print(f"electric_grid_energy_dlmp = \n{electric_grid_energy_dlmp}")
    print(f"electric_grid_voltage_dlmp = \n{electric_grid_voltage_dlmp}")
    print(f"electric_grid_congestion_dlmp = \n{electric_grid_congestion_dlmp}")
    print(f"electric_grid_loss_dlmp = \n{electric_grid_loss_dlmp}")
    print(f"thermal_grid_energy_dlmp = \n{thermal_grid_energy_dlmp}")
    print(f"thermal_grid_head_dlmp = \n{thermal_grid_head_dlmp}")
    print(f"thermal_grid_congestion_dlmp = \n{thermal_grid_congestion_dlmp}")
    print(f"thermal_grid_pump_dlmp = \n{thermal_grid_pump_dlmp}")

    # Store DLMPs as CSV.
    voltage_magnitude_vector_minimum_dlmp.to_csv(os.path.join(results_path, 'voltage_magnitude_vector_minimum_dlmp.csv'))
    voltage_magnitude_vector_maximum_dlmp.to_csv(os.path.join(results_path, 'voltage_magnitude_vector_maximum_dlmp.csv'))
    branch_power_vector_1_squared_maximum_dlmp.to_csv(os.path.join(results_path, 'branch_power_vector_1_squared_maximum_dlmp.csv'))
    branch_power_vector_2_squared_maximum_dlmp.to_csv(os.path.join(results_path, 'branch_power_vector_2_squared_maximum_dlmp.csv'))
    loss_active_dlmp.to_csv(os.path.join(results_path, 'loss_active_dlmp.csv'))
    loss_reactive_dlmp.to_csv(os.path.join(results_path, 'loss_reactive_dlmp.csv'))
    node_head_vector_minimum_dlmp.to_csv(os.path.join(results_path, 'node_head_vector_minimum_dlmp.csv'))
    branch_flow_vector_maximum_dlmp.to_csv(os.path.join(results_path, 'branch_flow_vector_maximum_dlmp.csv'))
    pump_power_dlmp.to_csv(os.path.join(results_path, 'pump_power_dlmp.csv'))
    electric_grid_energy_dlmp.to_csv(os.path.join(results_path, 'electric_grid_energy_dlmp.csv'))
    electric_grid_voltage_dlmp.to_csv(os.path.join(results_path, 'electric_grid_voltage_dlmp.csv'))
    electric_grid_congestion_dlmp.to_csv(os.path.join(results_path, 'electric_grid_congestion_dlmp.csv'))
    electric_grid_loss_dlmp.to_csv(os.path.join(results_path, 'electric_grid_loss_dlmp.csv'))
    thermal_grid_energy_dlmp.to_csv(os.path.join(results_path, 'thermal_grid_energy_dlmp.csv'))
    thermal_grid_head_dlmp.to_csv(os.path.join(results_path, 'thermal_grid_head_dlmp.csv'))
    thermal_grid_congestion_dlmp.to_csv(os.path.join(results_path, 'thermal_grid_congestion_dlmp.csv'))
    thermal_grid_pump_dlmp.to_csv(os.path.join(results_path, 'thermal_grid_pump_dlmp.csv'))

    # Plot thermal grid DLMPs.
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
            results['der_thermal_power_vector'].loc[:, der].abs() / 1000000,
            label='Thrm. pw.',
            drawstyle='steps-post',
            color='darkgrey',
            linewidth=3
        )
        ax2.plot(
            results['der_active_power_vector'].loc[:, der].abs() / 1000000,
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

    # Print results path.
    print("Results are stored in: " + results_path)


if __name__ == '__main__':
    main()
