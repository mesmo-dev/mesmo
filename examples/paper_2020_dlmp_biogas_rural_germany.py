"""Example script for setting up and solving a flexible biogas plant optimal operation problem.
"""

import matplotlib.pyplot as plt
import pyomo.environ as pyo
import pandas as pd
import matplotlib.dates
import networkx as nx
import numpy as np
import os

import fledge.data_interface
import fledge.der_models
import fledge.config
import fledge.electric_grid_models
import fledge.plots
import fledge.utils
import fledge.problems
import bipmo.bipmo.plots


# Settings.
scenario_name = 'cigre_mv_network_with_all_ders'
plots = True  # If True, script may produce plots.

# Obtain results path.
results_path = (
    fledge.utils.get_results_path(f'paper_2020_dlmp_biogas_rural_germany_scenario', scenario_name)
)

# Recreate / overwrite database, to incorporate changes in the CSV files.
fledge.data_interface.recreate_database()
# Obtain scenario data and price timeseries.
scenario_data = fledge.data_interface.ScenarioData(scenario_name)
price_data = fledge.data_interface.PriceData(scenario_name)
price_type = 'EPEX SPOT Power DE Day Ahead'
price_timeseries = price_data.price_timeseries_dict[price_type]

chp_schedule: pd.DataFrame

run_milp = False

for i in range(2):
    if not run_milp:
        if i == 0:
            is_milp = True
        else:
            is_milp = False
    else:
        i = 3  # will stop from iterating again

    # Obtain models.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    # Use base scenario power flow for consistent linear model behavior and per unit values.
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(scenario_name)
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )

    # TODO: define custom constraints on lines
    voltage_magnitude_vector_minimum = (
        scenario_data.scenario['voltage_per_unit_minimum']
        * np.abs(electric_grid_model.node_voltage_vector_reference)
        if pd.notnull(scenario_data.scenario['voltage_per_unit_minimum'])
        else None
    )
    voltage_magnitude_vector_maximum = (
        scenario_data.scenario['voltage_per_unit_maximum']
        * np.abs(electric_grid_model.node_voltage_vector_reference)
        if pd.notnull(scenario_data.scenario['voltage_per_unit_maximum'])
        else None
    )
    branch_power_vector_squared_maximum = (
        scenario_data.scenario['branch_flow_per_unit_maximum']
        * np.abs(electric_grid_model.branch_power_vector_magnitude_reference ** 2)
        if pd.notnull(scenario_data.scenario['branch_flow_per_unit_maximum'])
        else None
    )
    branch_power_vector_squared_maximum[
        fledge.utils.get_index(electric_grid_model.branches, branch_name='Line_2_3')
    ] *= 0.25

    # Get the biogas plant model and set the switches flag accordingly
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    flexible_biogas_plant_model = der_model_set.flexible_der_models['Biogas Plant 9']
    if not is_milp:
        # set the chp_schedule resulting from the milp optimization
        flexible_biogas_plant_model.chp_schedule = chp_schedule
    der_model_set.flexible_der_models[flexible_biogas_plant_model.der_name] = flexible_biogas_plant_model

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps,
        voltage_magnitude_vector_minimum=voltage_magnitude_vector_minimum,
        voltage_magnitude_vector_maximum=voltage_magnitude_vector_maximum,
        branch_power_vector_squared_maximum=branch_power_vector_squared_maximum
    )

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem,
        electric_grid_model=electric_grid_model,
        power_flow_solution=power_flow_solution
    )

    if is_milp:
        # define binary variables for MILP solution
        optimization_problem.binary_variables = pyo.Var(flexible_biogas_plant_model.timesteps,
                                                        [flexible_biogas_plant_model.der_name],
                                                        flexible_biogas_plant_model.switches,
                                                        domain=pyo.Binary)

        for timestep in flexible_biogas_plant_model.timesteps:
            for output in flexible_biogas_plant_model.outputs:
                if 'active_power_Wel' in output:
                    for chp in flexible_biogas_plant_model.CHP_list:
                        if chp in output and any(flexible_biogas_plant_model.switches.str.contains(chp)):
                            optimization_problem.der_model_constraints.add(
                                optimization_problem.output_vector[timestep, flexible_biogas_plant_model.der_name, output]
                                >=
                                flexible_biogas_plant_model.output_minimum_timeseries.at[timestep, output]
                                * optimization_problem.binary_variables[timestep, flexible_biogas_plant_model.der_name, chp + '_switch']
                            )
                            optimization_problem.der_model_constraints.add(
                                optimization_problem.output_vector[timestep, flexible_biogas_plant_model.der_name, output]
                                <=
                                flexible_biogas_plant_model.output_maximum_timeseries.at[timestep, output]
                                * optimization_problem.binary_variables[timestep, flexible_biogas_plant_model.der_name, chp + '_switch']
                            )

    # Define electric grid objective.
    linear_electric_grid_model.define_optimization_objective(
        optimization_problem,
        price_timeseries=price_timeseries,
        timesteps=scenario_data.timesteps
    )

    der_model_set.define_optimization_objective(
        optimization_problem,
        price_timeseries,
        electric_grid_model=electric_grid_model
    )

    # Solve optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.config['optimization']['show_solver_output'])
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")

    if is_milp:
        # get the MILP solution for the biogas plant schedule
        binaries = optimization_problem.binary_variables
        timesteps = flexible_biogas_plant_model.timesteps
        chp_schedule = flexible_biogas_plant_model.chp_schedule
        for timestep in timesteps:
            for chp in flexible_biogas_plant_model.CHP_list:
                chp_schedule.loc[timestep, chp+'_switch'] = \
                    binaries[timestep, flexible_biogas_plant_model.der_name, chp+'_switch'].value


in_per_unit = True

# Obtain results.
results = (
    linear_electric_grid_model.get_optimization_results(
        optimization_problem,
        power_flow_solution,
        scenario_data.timesteps,
        in_per_unit=in_per_unit
    )
)
results.update(
    der_model_set.get_optimization_results(
        optimization_problem
    )
)

# Obtain additional results.
branch_power_vector_magnitude_per_unit = (
    (
        np.sqrt(np.abs(results['branch_power_vector_1_squared']))
        + np.sqrt(np.abs(results['branch_power_vector_2_squared']))
    ) / 2
    # / electric_grid_model.branch_power_vector_magnitude_reference
)
branch_power_vector_magnitude_per_unit.loc['maximum', :] = branch_power_vector_magnitude_per_unit.max(axis='rows')
node_voltage_vector_magnitude_per_unit = (
    np.abs(results['voltage_magnitude_vector'])
    # / np.abs(electric_grid_model.node_voltage_vector_reference)
)
node_voltage_vector_magnitude_per_unit.loc['maximum', :] = node_voltage_vector_magnitude_per_unit.max(axis='rows')
node_voltage_vector_magnitude_per_unit.loc['minimum', :] = node_voltage_vector_magnitude_per_unit.min(axis='rows')
results.update({
    'branch_power_vector_magnitude_per_unit': branch_power_vector_magnitude_per_unit,
    'node_voltage_vector_magnitude_per_unit': node_voltage_vector_magnitude_per_unit
})

# Print results.
print(results)

 # Store results as CSV.
results.to_csv(results_path)

# Obtain DLMPs.
dlmps = (
    linear_electric_grid_model.get_optimization_dlmps(
        optimization_problem,
        price_timeseries,
        scenario_data.timesteps
    )
)

# Print DLMPs.
print(dlmps)

# Store DLMPs as CSV.
dlmps.to_csv(results_path)

if plots:
    # Plot electric grid DLMPs.
    electric_grid_dlmp = (
        pd.concat(
            [
                dlmps['electric_grid_energy_dlmp'],
                dlmps['electric_grid_loss_dlmp'],
                dlmps['electric_grid_voltage_dlmp'],
                dlmps['electric_grid_congestion_dlmp']
            ],
            axis='columns',
            keys=['energy', 'loss', 'voltage', 'congestion'],
            names=['dlmp_type']
        )
    )
    colors = list(color['color'] for color in matplotlib.rcParams['axes.prop_cycle'])
    for der in electric_grid_model.ders:

        # Obtain corresponding node.
        # TODO: Consider delta connected DERs.
        node = (
            electric_grid_model.nodes[
                electric_grid_model.der_incidence_wye_matrix[
                    :, electric_grid_model.ders.get_loc(der)
                ].toarray().ravel() > 0
            ]
        )

        # Create plot.
        fig, (ax1, lax) = plt.subplots(ncols=2, figsize=[7.8, 2.6], gridspec_kw={"width_ratios": [100, 1]})
        ax1.set_title(f"DER {der[1]} ({der[0].replace('_', ' ').capitalize()})")
        ax1.stackplot(
            scenario_data.timesteps,
            (
                electric_grid_dlmp.loc[:, (slice(None), *zip(*node))].groupby('dlmp_type', axis='columns').mean().T
            ),
            labels=['Energy', 'Loss', 'Voltage', 'Congest.'],
            colors=[colors[0], colors[1], colors[2], colors[3]],
            step='post'
        )
        ax1.plot(
            (
                electric_grid_dlmp.loc[
                    :, (slice(None), *zip(*node))
                ].groupby('dlmp_type', axis='columns').mean().sum(axis='columns')
            ),
            label='Total DLMP',
            drawstyle='steps-post',
            color='red',
            linewidth=1.0
        )
        ax1.grid(True)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price [€/MWh]')
        # ax1.set_ylim((0.0, 10.0))
        ax2 = plt.twinx(ax1)
        if der in electric_grid_model.ders:
            ax2.plot(
                results['der_active_power_vector'].loc[:, der].abs() / (1 if in_per_unit else 1e6),
                label='Active pw.',
                drawstyle='steps-post',
                color='black',
                linewidth=1.5
            )
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_xlim((scenario_data.timesteps[0].toordinal(), scenario_data.timesteps[-1].toordinal()))
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Power [p.u.]') if in_per_unit else ax2.set_ylabel('Power [MW]')
        # ax2.set_ylim((0.0, 1.0)) if in_per_unit else ax2.set_ylim((0.0, 30.0))
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        lax.legend((*h1, *h2), (*l1, *l2), borderaxespad=0)
        lax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'electric_grid_der_dlmp_{der}.png'))
        # plt.show()
        plt.close()

    # Obtain graphs.
    electric_grid_graph = fledge.plots.ElectricGridGraph(scenario_name)


    # Plot electric grid DLMPs in grid.
    dlmp_types = [
        'electric_grid_energy_dlmp',
        'electric_grid_voltage_dlmp',
        'electric_grid_congestion_dlmp',
        'electric_grid_loss_dlmp'
    ]
    for timestep in scenario_data.timesteps:
        for dlmp_type in dlmp_types:
            node_color = (
                dlmps[dlmp_type].loc[timestep, :].groupby('node_name').mean().reindex(electric_grid_graph.nodes).values
            )
            plt.title(
                f"{dlmp_type.replace('_', ' ').capitalize().replace('dlmp', 'DLMP')}"
                f" at {timestep.strftime('%H:%M:%S')}"
            )
            nx.draw_networkx_nodes(
                electric_grid_graph,
                pos=electric_grid_graph.node_positions,
                nodelist=(
                    electric_grid_model.nodes[
                        fledge.utils.get_index(electric_grid_model.nodes, node_type='source')
                    ].get_level_values('node_name')[:1].to_list()
                ),
                node_size=150.0,
                node_color='red',
            )
            nx.draw_networkx(
                electric_grid_graph,
                pos=electric_grid_graph.node_positions,
                arrows=False,
                node_size=100.0,
                node_color=node_color,
                edgecolors='black',  # Make node border visible.
            )
            sm = (
                plt.cm.ScalarMappable(
                    norm=plt.Normalize(
                        vmin=np.min(node_color),
                        vmax=np.max(node_color)
                    )
                )
            )
            cb = plt.colorbar(sm, shrink=0.9)
            cb.set_label('Price [€/MWh]')
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, f'{dlmp_type}_{timestep.strftime("%H-%M-%S")}.png'))
            # plt.show()
            plt.close()

    # Plot electric grid line utilization.
    fledge.plots.plot_grid_line_utilization(
        electric_grid_model,
        electric_grid_graph,
        branch_power_vector_magnitude_per_unit * 100.0,
        results_path,
        value_unit='%',
    )

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")

    results = (
        flexible_biogas_plant_model.get_optimization_results(
            optimization_problem, price_timeseries
        )
    )

    print(results)

    if plots:
        bipmo.bipmo.plots.generate_biogas_plant_plots(results, flexible_biogas_plant_model, price_timeseries)
