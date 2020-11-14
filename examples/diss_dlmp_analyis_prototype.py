"""Script for analyzing the impact of the granularity of a price signal"""

import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
import plotly.graph_objects as go

import fledge.config
import fledge.data_interface
import fledge.utils
import fledge.der_models
import fledge.electric_grid_models
import fledge.problems
import fledge.analysis_utils as au


# Global Settings
voltage_min = 0.8
voltage_max = 1.2
path_to_solver_executable = '/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx/cplex'


# def main():
#
#     scenario_name = 'cigre_high_granularity'
#     price_data = get_price_data_for_scenario(scenario_name)
#     # Recreate / overwrite database, to incorporate the new grids that we created
#     fledge.data_interface.recreate_database()
#     results_path = fledge.utils.get_results_path('run_electric_grid_optimal_operation', scenario_name)
#     _, results, dlmps = run_centralized_problem(scenario_name, results_path)
#
#     price_data_dlmps = price_data.copy()
#     price_data_dlmps.price_timeseries = dlmps['electric_grid_total_dlmp_price_timeseries']
#
#     _, results_validation = run_decentralized_problem(scenario_name, results_path, price_data_dlmps)


def main():
    plots = True  # will generate plots if set to True
    regenerate_scenarios = False  # will regenerate the grid input data

    # Define grid / scenario names
    mv_grid_name = 'cigre_mv_network'
    high_granularity_scenario_name = 'cigre_high_granularity'
    low_granularity_scenario_name = 'cigre_low_granularity'
    no_granularity_scenario_name = low_granularity_scenario_name  # use same scenario, only without the actual grid

    # Generate the grids that are needed for different granularity levels (comment out if not needed)
    if regenerate_scenarios:
        path_to_grid_map = 'examples/electric_grid_mapping.csv'
        au.combine_electric_grids(mv_grid_name, path_to_grid_map, high_granularity_scenario_name)
        au.aggregate_electric_grids(mv_grid_name, path_to_grid_map, low_granularity_scenario_name)

    # Recreate / overwrite database, to incorporate the new grids that we created
    fledge.data_interface.recreate_database()

    # Run centralized problems for all granularity levels
    scenarios = [high_granularity_scenario_name, low_granularity_scenario_name]
    # the no granularity scenario is added below in the  decentralized case
    opf_results_dict = {}
    for scenario_name in scenarios:
        results_path = fledge.utils.get_results_path(
            'run_electric_grid_optimal_operation', scenario_name + '_central')
        opt_objective, opf_results, dlmps = run_centralized_problem(scenario_name, results_path)
        opf_results_dict['opt_objective_' + scenario_name + '_central'] = opt_objective
        opf_results_dict['opf_results_' + scenario_name + '_central'] = opf_results
        opf_results_dict['dlmps_' + scenario_name] = dlmps

    # Run decentralized problem based on the DLMPs and one based on wholesale market price (no granularity)
    # TODO: Test scenario with weighted average DLMP instead of MV-level DLMP
    # first the scenarios from above
    price_timeseries_dict = {}
    for scenario_name in scenarios:
        results_path = fledge.utils.get_results_path(
            'run_electric_grid_optimal_operation', scenario_name + '_decentral')
        # Obtain price data
        price_data = get_price_data_for_scenario(scenario_name)
        # Change price time series to dlmps
        dlmps = opf_results_dict['dlmps_' + scenario_name]
        price_data.price_timeseries = dlmps['electric_grid_total_dlmp_price_timeseries']
        opt_objective, opf_results = run_decentralized_problem(
            scenario_name, results_path, price_data)
        opf_results_dict['opt_objective_' + scenario_name + '_decentral'] = opt_objective
        opf_results_dict['opf_results_' + scenario_name + '_decentral'] = opf_results
        price_timeseries_dict[scenario_name] = price_data.price_timeseries

    # then the "no granularity" scenario based on the wholesale market
    scenario_name = no_granularity_scenario_name
    price_data = get_price_data_for_scenario(scenario_name)
    results_path = fledge.utils.get_results_path('run_electric_grid_optimal_operation',
                                                 'cigre_no_granularity' + '_decentral')
    opt_objective, results = run_decentralized_problem(
        scenario_name, results_path, price_data)
    opf_results_dict['opt_objective_' + 'cigre_no_granularity' + '_decentral'] = opt_objective
    opf_results_dict['opf_results_' + 'cigre_no_granularity' + '_decentral'] = opf_results
    price_timeseries_dict['cigre_no_granularity'] = price_data.price_timeseries

    # Get the set points from decentralized problems and calculate nominal power flow
    # now using the entire grid again
    pf_results_dict = {}
    der_costs_revenues_dict = {}
    system_costs = {}
    scenario_name = high_granularity_scenario_name
    for key in opf_results_dict.keys():
        if ('decentral' in key) and ('opf_results' in key):
            results_path = fledge.utils.get_results_path('run_electric_grid_nominal_operation', key + '_power_flow')
            # change the set points of the DERs
            der_model_set = change_der_set_points_based_on_results(scenario_name, opf_results_dict[key])
            # run the nominal power flow and store results in dictionary
            pf_results_dict[get_scenario_string(key)] = run_nominal_operation(scenario_name, der_model_set,
                                                                              results_path)
            # calculate the system costs
            system_costs[get_scenario_string(key)] = calculate_system_costs_at_source_node(scenario_name, der_model_set)
            # calculate individual DER costs / revenues
            der_costs_revenues_dict[get_scenario_string(key)] = \
                price_timeseries_dict[get_scenario_string(key)].loc[:,
                ('active_power', slice(None), der_model_set.der_names)] \
                * pf_results_dict[get_scenario_string(key)]['der_power_magnitude'].values

    # Gather all result data in a dictionary and generate plots
    if plots:
        # Combine all results in one big dictionary
        results_dict = {'opf_results': opf_results_dict,
                        'pf_results': pf_results_dict,
                        'price_timeseries': price_timeseries_dict,
                        'system_costs': system_costs,
                        'der_costs_revenues': der_costs_revenues_dict}
        # Pass to plot function
        results_path = fledge.utils.get_results_path('dlmp_analysis_plots', mv_grid_name)
        generate_result_plots(results_dict, high_granularity_scenario_name, results_path)

    # TODO: repeat the above with higher DER penetration
    # DERs can be modified based on the code example in function change_der_set_points...

    print('Done.')


def run_centralized_problem(
        scenario_name: str,
        results_path: str
) -> [float, fledge.data_interface.ResultsDict, fledge.data_interface.ResultsDict]:
    # This should return the results of the centralized optimization problem (incl. grid)
    # Obtain data
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name, price_type=scenario_data.scenario['price_type'])
    # Formulate central problem
    [der_model_set, electric_grid_model, linear_electric_grid_model, power_flow_solution, optimization_problem] = \
        formulate_electric_grid_optimization_problem(scenario_name)

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

    # Define DER objective.
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_data,
        electric_grid_model=electric_grid_model
    )

    # Solve centralized optimization problem.
    solve_optimization_problem(optimization_problem)

    # Obtain results.
    results = (
        linear_electric_grid_model.get_optimization_results(
            optimization_problem,
            power_flow_solution,
            scenario_data.timesteps,
            in_per_unit=True,
            with_mean=True
        )
    )
    results.update(
        der_model_set.get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    print(results)

    # Store results as CSV.
    results.to_csv(results_path)

    # Obtain DLMPs.
    dlmps = (
        linear_electric_grid_model.get_optimization_dlmps(
            optimization_problem,
            price_data,
            scenario_data.timesteps
        )
    )

    # Print DLMPs.
    print(dlmps)

    # Store DLMPs as CSV.
    dlmps.to_csv(results_path)

    # Print results path.
    # fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")

    return [pyo.value(optimization_problem.objective), results, dlmps]


def formulate_electric_grid_optimization_problem(
        scenario_name: str,
) -> [fledge.der_models.DERModelSet,
      fledge.electric_grid_models.ElectricGridModelDefault,
      fledge.electric_grid_models.LinearElectricGridModelGlobal,
      fledge.electric_grid_models.PowerFlowSolutionFixedPoint,
      pyo.ConcreteModel]:
    # Obtain data.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name, price_type=scenario_data.scenario['price_type'])

    # Obtain models.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Instantiate centralized optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define linear electric grid model constraints.
    voltage_magnitude_vector_minimum = voltage_min * np.abs(electric_grid_model.node_voltage_vector_reference)
    # voltage_magnitude_vector_minimum[
    #     fledge.utils.get_index(electric_grid_model.nodes, node_name='4')
    # ] *= 0.965 / 0.5
    voltage_magnitude_vector_maximum = voltage_max * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_vector_squared_maximum = np.abs(
        electric_grid_model.branch_power_vector_magnitude_reference ** 2)
    # branch_power_vector_squared_maximum[
    #     fledge.utils.get_index(electric_grid_model.branches, branch_type='line', branch_name='2')
    # ] *= 1.2 / 10.0
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps,
        voltage_magnitude_vector_minimum=voltage_magnitude_vector_minimum,
        voltage_magnitude_vector_maximum=voltage_magnitude_vector_maximum,
        branch_power_vector_squared_maximum=branch_power_vector_squared_maximum
    )

    # Define grid  / centralized objective.
    linear_electric_grid_model.define_optimization_objective(
        optimization_problem,
        price_data,
        scenario_data.timesteps
    )

    return [der_model_set, electric_grid_model, linear_electric_grid_model, power_flow_solution, optimization_problem]


def run_decentralized_problem(
        scenario_name: str,
        results_path: str,
        price_data: fledge.data_interface.PriceData
) -> [float, fledge.data_interface.ResultsDict]:
    """
    This function returns the results of the decentralized optimization problem (DERs optimize based on price signal)
    :param scenario_name:
    :param results_path:
    :param dlmps:
    :return: results dictionary of the optimization
    """

    # Obtain all DERs
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Instantiate decentralized DER optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem
    )

    # Define objective (DER operation cost minimization).
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_data
    )

    # Solve decentralized DER optimization problem.
    solve_optimization_problem(optimization_problem)

    # Obtain results.
    results = (
        der_model_set.get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    print(results)
    # Store results as CSV.
    results.to_csv(results_path)

    # Print results path.
    # fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")

    return [pyo.value(optimization_problem.objective), results]


def solve_optimization_problem(
        optimization_problem: pyo.ConcreteModel
):
    # Solve decentralized DER optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
    optimization_solver = pyo.SolverFactory('cplex', executable=path_to_solver_executable)
    optimization_result = optimization_solver.solve(optimization_problem,
                                                    tee=fledge.config.config['optimization']['show_solver_output'])
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")


def run_nominal_operation(
        scenario_name: str,
        der_model_set: fledge.der_models.DERModelSet,
        results_path: str
) -> fledge.data_interface.ResultsDict:
    # run nominal operation problem with the set points from the decentralized problems
    # Formulate nominal operation problem
    problem = fledge.problems.NominalOperationProblem(scenario_name)
    # Update the der model set (with new set points)
    problem.der_model_set = der_model_set
    problem.solve()
    results = problem.get_results()
    problem.electric_grid_model.der_power_vector_reference
    # Print results.
    print(results)
    # Store results as CSV.
    results.to_csv(results_path)

    return results


def increase_der_penetration(
        electric_grid: fledge.data_interface.ElectricGridData
) -> fledge.data_interface.ElectricGridData:
    # TODO: still to be implemented
    # some function to vary the DER penetration per iteration
    # what would be the input? the electric grid?
    pass


def change_der_set_points_based_on_results(
        scenario_name: str,
        results: fledge.data_interface.ResultsDict
) -> fledge.der_models.DERModelSet:
    # Requirements: for every DER model it should return the active and reactive power output based on the results
    grid_data = fledge.data_interface.ElectricGridData(scenario_name)
    # Obtain DER model set
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    for der_name in der_model_set.der_names:
        if der_name in results['output_vector']:
            der_model_set.der_models[der_name].active_power_nominal_timeseries = \
                results['output_vector'].loc[:, (der_name, 'active_power')]
            der_model_set.der_models[der_name].reactive_power_nominal_timeseries = \
                results['output_vector'].loc[:, (der_name, 'reactive_power')]

    return der_model_set


def get_price_data_for_scenario(
        scenario_name: str
) -> fledge.data_interface.PriceData:
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    return fledge.data_interface.PriceData(scenario_name, price_type=scenario_data.scenario['price_type'])


# def calculate_system_costs_at_source_node_comparison(
#         scenario_name: str,
#         der_model_set_new_setpoints: fledge.der_models.DERModelSet
# ) -> float:
# TODO: this currently only returns infeasible problems, that is why I use the workaround below
#     scenario_data = fledge.data_interface.ScenarioData(scenario_name)
#     # Formulate central problem
#     [der_model_set, electric_grid_model, linear_electric_grid_model, power_flow_solution, optimization_problem] = \
#         formulate_electric_grid_optimization_problem(scenario_name)
#
#     # Custom constraints to fix the power output to set points
#     if optimization_problem.find_component('der_model_constraints') is None:
#         optimization_problem.der_model_constraints = pyo.ConstraintList()
#     for der_name in der_model_set.der_names:
#         der_index = int(fledge.utils.get_index(electric_grid_model.ders, der_name=der_name))
#         der = electric_grid_model.ders[der_index]
#         for timestep in scenario_data.timesteps:
#             optimization_problem.der_model_constraints.add(
#                 optimization_problem.der_active_power_vector_change[timestep, der]
#                 ==
#                 der_model_set_new_setpoints.der_models[der_name].active_power_nominal_timeseries[timestep] - np.real(power_flow_solution.der_power_vector[der_index])
#             )
#             optimization_problem.der_model_constraints.add(
#                 optimization_problem.der_reactive_power_vector_change[timestep, der]
#                 ==
#                 der_model_set_new_setpoints.der_models[der_name].reactive_power_nominal_timeseries[timestep] - np.imag(power_flow_solution.der_power_vector[der_index])
#             )
#
#     # Solve centralized optimization problem.
#     solve_optimization_problem(optimization_problem)
#     return optimization_problem.objective.expr()


def calculate_system_costs_at_source_node(
        scenario_name: str,
        der_model_set_new_setpoints: fledge.der_models.DERModelSet
) -> float:
    # This function returns the ...
    # Obtain data
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    timesteps = scenario_data.timesteps

    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution_reference = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)

    power_flow_solutions = get_power_flow_solutions_for_timesteps(
        electric_grid_model,
        der_model_set_new_setpoints,
        timesteps
    )

    # Use power flow solution to formulate objective function
    # first define variables
    price_data = fledge.data_interface.PriceData(scenario_name, price_type=scenario_data.scenario['price_type'])
    # Obtain models.
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution_reference
        )
    )
    # Instantiate centralized optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        timesteps
    )
    # Define objective
    linear_electric_grid_model.define_optimization_objective(
        optimization_problem,
        price_data,
        timesteps
    )

    # calculate the objective by evaluating the expression and repeat for next timestep
    set_electric_grid_optimization_variables_based_on_power_flow(
        optimization_problem,
        electric_grid_model,
        power_flow_solution_reference,
        power_flow_solutions,
        timesteps
    )
    return optimization_problem.objective.expr()[0]


def set_electric_grid_optimization_variables_based_on_power_flow(
        optimization_problem: pyo.ConcreteModel,
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        power_flow_solution_reference,
        power_flow_solutions,
        timesteps
):
    # set objective function variables to zero (power vector change is zero, as the solution is already correct)
    for timestep in timesteps:
        power_flow_solution = power_flow_solutions[timestep]
        loss_active_change = np.real(power_flow_solution.loss) - np.real(power_flow_solution_reference.loss)
        loss_reactive_change = np.imag(power_flow_solution.loss) - np.imag(power_flow_solution_reference.loss)
        optimization_problem.loss_active_change[timestep] = loss_active_change
        optimization_problem.loss_reactive_change[timestep] = loss_reactive_change
        for der_index, der in enumerate(electric_grid_model.ders):
            active_power_change = np.real(power_flow_solution.der_power_vector[der_index]) - \
                                  np.real(power_flow_solution_reference.der_power_vector[der_index])
            reactive_power_change = np.imag(power_flow_solution.der_power_vector[der_index]) - \
                                    np.imag(power_flow_solution_reference.der_power_vector[der_index])
            optimization_problem.der_active_power_vector_change[timestep, der].value = active_power_change
            optimization_problem.der_reactive_power_vector_change[timestep, der].value = reactive_power_change


def get_power_flow_solutions_for_timesteps(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        der_model_set_new_setpoints: fledge.der_models.DERModelSet,
        timesteps
):
    der_power_vector = (
        pd.DataFrame(columns=electric_grid_model.ders, index=timesteps, dtype=np.complex)
    )
    # Obtain nominal DER power vector based on set points from the optimal power flows
    for der in electric_grid_model.ders:
        der_name = der[1]
        der_power_vector.loc[:, der] = (
                der_model_set_new_setpoints.der_models[der_name].active_power_nominal_timeseries
                + (1.0j * der_model_set_new_setpoints.der_models[der_name].reactive_power_nominal_timeseries)
        )
    # use DER power vector to calculate power flow per timestep
    power_flow_solutions = (
        fledge.utils.starmap(
            fledge.electric_grid_models.PowerFlowSolutionFixedPoint,
            zip(
                itertools.repeat(electric_grid_model),
                der_power_vector.values
            )
        )
    )
    return dict(zip(timesteps, power_flow_solutions))


def generate_result_plots(
        results_dict: dict,
        high_granularity_scenario_name: str,
        results_path: str = None
):
    # This function produces the plots:
    # [x] branch power magnitude,
    # [x] node voltage magnitude
    # [x] total losses
    # [x] der dispatch
    # [x] objective result / system costs
    # [x] prices over time
    # [x] branch power magnitude over time per branch
    # [x] node voltage magnitude over time per node
    # [x] der output over time
    # [x] dlmp validation
    # [ ] load flow calculation
    pf_results = results_dict['pf_results']
    system_costs = results_dict['system_costs']
    opf_results = results_dict['opf_results']
    price_timeseries = results_dict['price_timeseries']

    # Get the electric grid model of the high granularity (was the same for all power flow calculations)
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(
        high_granularity_scenario_name)
    scenario_data = fledge.data_interface.ScenarioData(high_granularity_scenario_name)  # only used for timesteps
    der_model_set = fledge.der_models.DERModelSet(high_granularity_scenario_name)

    # Branch power magnitude for both directions and node voltage
    plots = {
        'Branch power (direction 1) magnitude [p.u.]': 'branch_power_1_magnitude_per_unit',
        'Branch power (direction 2) magnitude [p.u.]': 'branch_power_2_magnitude_per_unit',
        'Node voltage magnitude [p.u.]': 'node_voltage_magnitude_per_unit',
        'DER power magnitude [p.u.]': 'der_power_magnitude_per_unit',
    }
    scenarios = get_list_of_scenario_strings(pf_results)
    for plot in plots.keys():
        minimum = None
        maximum = None
        legend = scenarios
        if 'Node voltage' in plot:
            x_index = electric_grid_model.nodes
            minimum = voltage_min * 0.8
            maximum = voltage_max * 1.2
        elif 'Branch power' in plot:
            x_index = electric_grid_model.branches.append(electric_grid_model.transformers)
            minimum = 0.0
            maximum = 1.3
        elif 'DER' in plot:
            x_index = electric_grid_model.ders
            minimum = 0.0
            maximum = 1.3
        for timestep in scenario_data.timesteps:
            plt.title(plot + ' at: ' + timestep.strftime("%H-%M-%S"))
            marker_index = 4
            for scenario_name in scenarios:
                pf_results_scenario_name = [key for key in pf_results.keys() if scenario_name in key][0]
                pf_results_of_scenario = pf_results[pf_results_scenario_name]
                plt.scatter(
                    range(len(x_index)),
                    pf_results_of_scenario[plots[plot]].loc[timestep, :].reindex(x_index).values,
                    marker=marker_index
                )
                marker_index += 1
            plt.xticks(
                range(len(x_index)),
                x_index,
                rotation=45,
                ha='right'
            )
            if minimum is not None:
                plt.ylim((minimum, maximum))
            plt.legend(legend)
            plt.grid()
            plt.tight_layout()
            save_or_show_plot(plot, results_path, timestep)

    # Branch power magnitude for both directions and node voltage over time
    plots = {
        'Branch power (direction 1) magnitude [p.u.] for branch ': 'branch_power_1_magnitude_per_unit',
        'Branch power (direction 2) magnitude [p.u.] for branch ': 'branch_power_2_magnitude_per_unit',
        'Node voltage magnitude [p.u.] for node ': 'node_voltage_magnitude_per_unit',
        'DER power magnitude [p.u.] for ': 'der_power_magnitude_per_unit',
    }
    minimum = None
    maximum = None
    for plot in plots.keys():
        legend = scenarios
        if 'Node voltage' in plot:
            assets = electric_grid_model.nodes
            minimum = voltage_min * 0.8
            maximum = voltage_max * 1.2
        elif 'Branch power' in plot:
            assets = electric_grid_model.branches
            minimum = 0.0
            maximum = 1.3
        elif 'DER' in plot:
            assets = electric_grid_model.ders
            minimum = 0.0
            maximum = 1.3
        for asset in assets:
            plt.title(plot + asset[1])
            marker_index = 4
            for scenario_name in scenarios:
                pf_results_scenario_name = [key for key in pf_results.keys() if scenario_name in key][0]
                pf_results_of_scenario = pf_results[pf_results_scenario_name]
                plt.plot(
                    scenario_data.timesteps,
                    pf_results_of_scenario[plots[plot]].loc[:, (asset[0],  asset[1], slice(None))],
                    marker=marker_index
                )
                marker_index += 1
            if minimum is not None:
                plt.ylim((minimum, maximum))
            plt.legend(legend)
            plt.grid()
            plt.tight_layout()
            save_or_show_plot(plot + asset[1], results_path)

    # Generate plots that are independent of branches / nodes:
    plots = {
        'System losses': 'loss_magnitude',
        'Total system losses': 'loss_magnitude',
    }
    for plot in plots.keys():
        scenarios = get_list_of_scenario_strings(pf_results)
        if 'Total' in plot:
            x_index = ['total']
        elif 'System losses' in plot:
            x_index = scenario_data.timesteps
        plt.title(plot)
        marker_index = 4
        for scenario_name in scenarios:
            pf_results_scenario_name = [key for key in pf_results.keys() if scenario_name in key][0]
            pf_results_of_scenario = pf_results[pf_results_scenario_name]
            if 'Total' in plot:
                y_values = sum(pf_results_of_scenario[plots[plot]].values)
            else:
                y_values = pf_results_of_scenario[plots[plot]].values
            plt.scatter(
                range(len(x_index)),
                y_values,
                marker=marker_index
            )
            marker_index += 1
        plt.xticks(
            range(len(x_index)),
            x_index,
            rotation=45,
            ha='right'
        )
        plt.legend(scenarios)
        plt.grid()
        plt.tight_layout()
        save_or_show_plot(plot, results_path)

    # System costs plots
    plots = {
        'Objective value': 'opt_objective_',
    }
    for plot in plots.keys():
        scenarios = get_list_of_scenario_strings(pf_results)
        if 'Objective value' in plot:
            x_index = ['Central (all)', 'Decentral (all)', 'Costs (at supply node)']

        plt.title(plot)
        marker_index = 4
        for scenario_name in scenarios:
            opt_type = 'central'
            if 'no_granularity' in scenario_name:
                opt_type = 'decentral'
            opf_results_scenario_name = plots[plot] + scenario_name + '_' + opt_type
            opf_results_of_scenario_central = opf_results[opf_results_scenario_name]
            opt_type = 'decentral'
            opf_results_scenario_name = plots[plot] + scenario_name + '_' + opt_type
            opf_results_of_scenario_decentral = opf_results[opf_results_scenario_name]
            system_costs_of_scenario = system_costs[scenario_name]
            y_values = [opf_results_of_scenario_central, opf_results_of_scenario_decentral, system_costs_of_scenario]
            plt.scatter(
                range(len(x_index)),
                y_values,
                marker=marker_index
            )
            marker_index += 1
        plt.xticks(
            range(len(x_index)),
            x_index,
            rotation=45,
            ha='right'
        )
        plt.legend(scenarios)
        plt.grid()
        plt.tight_layout()
        save_or_show_plot(plot, results_path)

    # Price plots
    for der_name in der_model_set.der_names:
        plot = f'Price at {der_name} in EUR per kWh'
        plt.title(plot)
        marker_index = 4
        for key in price_timeseries.keys():
            price_timeseries_at_der = price_timeseries[key].loc[:, ('active_power', slice(None), der_name)].iloc[:, 0]
            x_index = price_timeseries_at_der.index
            plt.scatter(
                range(len(x_index)),
                price_timeseries_at_der.values * 1e3,
                marker=marker_index
            )
            marker_index += 1
        plt.xticks(
            range(len(x_index)),
            x_index,
            rotation=45,
            ha='right'
        )
        plt.legend(scenarios)
        plt.grid()
        plt.tight_layout()
        save_or_show_plot(plot, results_path)

    # DLMP validation plots
    wholesale_price = price_timeseries['cigre_no_granularity']
    price_timeseries_dlmps = price_timeseries['cigre_high_granularity']
    central_opf_results = opf_results['opf_results_cigre_high_granularity_central']
    decentral_opf_results = opf_results['opf_results_cigre_high_granularity_decentral']
    for der_name in der_model_set.der_names:
        if issubclass(type(der_model_set.der_models[der_name]), fledge.der_models.FlexibleDERModel):
            plot_dlmp_validation(
                der_name=der_name,
                price_timeseries=wholesale_price,
                price_timeseries_dlmps=price_timeseries_dlmps,
                results=central_opf_results,
                results_validation=decentral_opf_results,
                results_path=results_path
            )

    if results_path is not None:
        print(f"Plots are stored in: {results_path}")
        fledge.utils.launch(results_path)


def save_or_show_plot(
        plot_name: str,
        results_path: str,
        timestep: pd.Timestamp = None
):
    if results_path is None:
        plt.show()
    else:
        if timestep is None:
            file_name = f'{plot_name}.png'
        else:
            file_name = f'{plot_name}_{timestep.strftime("%Y-%m-%d_%H-%M-%S")}.png'
        file_path = os.path.join(results_path, file_name)
        plt.savefig(file_path)
        print(f'Saved plot: {file_name}', end="\r")
    plt.close()


def get_list_of_scenario_strings(
        pf_results_dict: dict
) -> list:
    scenarios = []
    for key in pf_results_dict.keys():
        scenarios.append(key)
    return scenarios


def get_scenario_string(
        key: str
) -> str:
    return key[len('opf_results_'): -len('_decentral')]


def plot_dlmp_validation(
        der_name: str,
        price_timeseries: pd.DataFrame,
        price_timeseries_dlmps: pd.DataFrame,
        results: fledge.data_interface.ResultsDict,
        results_validation: fledge.data_interface.ResultsDict,
        results_path: str
):
    values_1 = price_timeseries.loc[:, ('active_power', slice(None), der_name)].iloc[:, 0]
    values_2 = price_timeseries_dlmps.loc[:, ('active_power', slice(None), der_name)].iloc[:, 0]
    values_3 = price_timeseries_dlmps.loc[:, ('reactive_power', slice(None), der_name)].iloc[:, 0]

    title = 'Price comparison: ' + der_name
    filename = 'price_timeseries_comparison_' + der_name
    y_label = 'Price'
    value_unit = 'EUR/Wh'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=values_1.index,
        y=values_1.values,
        name='Wholesale price',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_2.index,
        y=values_2.values,
        name='DLMP (active power)',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_3.index,
        y=values_3.values,
        name='DLMP (reactive power)',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.update_layout(
        title=title,
        yaxis_title=f'{y_label} [{value_unit}]',
        xaxis=go.layout.XAxis(tickformat='%H:%M'),
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.5, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + '.png'))

    # Plot: Active power comparison.
    values_1 = -1e-6 * results['output_vector'].loc[:, (der_name, 'active_power')]
    values_2 = -1e-6 * results_validation['output_vector'].loc[:, (der_name, 'active_power')]

    title = 'Active power comparison: ' + der_name
    filename = 'active_power_comparison_' + der_name
    y_label = 'Active power'
    value_unit = 'MW'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=values_1.index,
        y=values_1.values,
        name='Centralized solution',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_2.index,
        y=values_2.values,
        name='DER (decentralized) solution',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.update_layout(
        title=title,
        yaxis_title=f'{y_label} [{value_unit}]',
        xaxis=go.layout.XAxis(tickformat='%H:%M'),
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + '.png'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
