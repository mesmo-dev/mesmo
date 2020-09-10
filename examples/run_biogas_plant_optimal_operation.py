"""Example script for setting up and solving a flexible biogas plant optimal operation problem.
"""

import matplotlib.pyplot as plt
import pyomo.environ as pyo
import pandas as pd

import fledge.data_interface
import fledge.der_models
import fledge.config
import fledge.utils


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
price_type = 'biogas'
price_timeseries = price_data.price_timeseries_dict[price_type]

run_milp = False
chp_schedule: pd.DataFrame

for i in range(2):
    if not run_milp:
        if i == 0:
            is_milp = True
        else:
            is_milp = False
    else:
        i = 3  # will stop from iterating again

    # Get the biogas plant model and set the switches flag accordingly
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    flexible_biogas_plant_model = der_model_set.flexible_der_models['Biogas Plant 9']
    if not is_milp:
        # set the chp_schedule resulting from the milp optimization
        flexible_biogas_plant_model.chp_schedule = chp_schedule
    der_model_set.flexible_der_models[flexible_biogas_plant_model.der_name] = flexible_biogas_plant_model

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    flexible_biogas_plant_model.define_optimization_variables(optimization_problem)
    flexible_biogas_plant_model.define_optimization_constraints(optimization_problem)

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

    flexible_biogas_plant_model.define_optimization_objective(optimization_problem)

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


results = (
    flexible_biogas_plant_model.get_optimization_results(
        optimization_problem, price_timeseries
    )
)
# Plot results.
if plots:

    for output in flexible_biogas_plant_model.outputs:
        plt.plot(flexible_biogas_plant_model.output_maximum_timeseries[output], label="Maximum", drawstyle='steps-post')
        plt.plot(flexible_biogas_plant_model.output_minimum_timeseries[output], label="Minimum", drawstyle='steps-post')
        plt.plot(results['output_vector'][output], label="Optimal", drawstyle='steps-post')
        plt.legend()
        plt.xlabel('Timesteps (day and hour)')
        plt.title(f"Output: {output}")
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
                 'rotation', 45)
        plt.show()
        plt.close()

    for control in flexible_biogas_plant_model.controls:
        plt.plot(results['control_vector'][control], label="Optimal", drawstyle='steps-post', color='#D55E00')
        plt.legend()
        plt.xlabel('Timesteps (day and hour)')
        plt.title(f"Control: {control}")
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
                 'rotation', 45)
        plt.show()
        plt.close()

    for state in flexible_biogas_plant_model.states:
        plt.plot(results['state_vector'][state], label="Optimal", drawstyle='steps-post', color='#D55E00')
        plt.legend()
        plt.xlabel('Timesteps (day and hour)')
        plt.title(f"State: {state}")
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
                 'rotation', 45)
        plt.show()
        plt.close()

    plt.plot(results['profit_vector']['profit_value'], label="Optimal", drawstyle='steps-post', color='#D55E00')
    plt.legend()
    plt.title('Profit per time interval (euros)')
    plt.setp(plt.gca().xaxis.get_majorticklabels(),
             'rotation', 45)
    plt.xlabel('Timesteps (day and hour)')
    plt.ylabel('Profit (euros)')
    plt.show()
    plt.close()

    plt.plot(price_timeseries['price_value'], drawstyle='steps-post')
    plt.title(f"Price in euros per MWh: {price_type}")
    plt.setp(plt.gca().xaxis.get_majorticklabels(),
             'rotation', 45)
    plt.show()
    plt.close()

    print("Total profit in euros:", sum(results['profit_vector']['profit_value']))
