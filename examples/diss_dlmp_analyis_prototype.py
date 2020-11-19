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
voltage_min = 0.9
voltage_max = 1.1
solver = 'default'  #choice: 'cplex', default is currently 'gurobi' (see config.yml)
path_to_solver_executable = '/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx/cplex'
plots = True  # will generate plots if set to True
regenerate_scenario_data = True  # will re-generate the grid input data if set to True


def main():

    # Define grid / scenario names (the exported scenario data will have these names)
    mv_grid_name = 'simple_mv_3node'
    high_granularity_scenario_name = 'simple_high_granularity'
    low_granularity_scenario_name = 'simple_low_granularity'
    no_granularity_scenario_name = low_granularity_scenario_name  # use same scenario, only without the actual grid
    # no_granularity does not generate new scenario data

    der_penetration_scenario_data = {
        'no_penetration': 0.0,
        'low_penetration': 0.5,
        'high_penetration': 1.0,
    }

    # Generate the grids that are needed for different granularity levels (comment out if not needed)
    if regenerate_scenario_data:
        # generate aggregated / combined grids
        path_to_grid_map = 'examples/electric_grid_mapping.csv'
        au.combine_electric_grids(mv_grid_name, path_to_grid_map, high_granularity_scenario_name)
        au.aggregate_electric_grids(mv_grid_name, path_to_grid_map, low_granularity_scenario_name)
        # generate scenario data for DER penetration scenarios
        path_to_der_data = 'examples/additional_electric_grid_ders.csv'
        grid_scenarios = [low_granularity_scenario_name, high_granularity_scenario_name]
        for scenario_name in grid_scenarios:
            for der_penetration in der_penetration_scenario_data.keys():
                au.increase_der_penetration_of_scenario_on_lv_level(
                    scenario_name=scenario_name,
                    path_to_der_data=path_to_der_data,
                    penetration_ratio=der_penetration_scenario_data[der_penetration],
                    new_scenario_name=scenario_name + '_' + der_penetration
                )

    # Recreate / overwrite database, to incorporate the new grids that we created
    fledge.data_interface.recreate_database()
    results_dict = {}
    # Run different granularity / der penetration scenarios and analyze
    for der_penetration in der_penetration_scenario_data:
        granularity_scenario_data = {
            'high_granularity': high_granularity_scenario_name + '_' + der_penetration,
            'high_granularity_mean': high_granularity_scenario_name + '_' + der_penetration,
            'low_granularity': low_granularity_scenario_name + '_' + der_penetration,
            'no_granularity': no_granularity_scenario_name + '_' + der_penetration
        }
        results_dict[der_penetration] = run_dlmp_analysis_for_scenario(
            granularity_scenario_data=granularity_scenario_data,
            der_penetration=der_penetration
        )

    return results_dict


def run_dlmp_analysis_for_scenario(
    granularity_scenario_data: dict,
    der_penetration: str = 'default'
) -> dict:

    # Run centralized problems for all granularity levels
    opf_results_dict = {}
    for granularity_level in granularity_scenario_data.keys():
        if 'no_granularity' in granularity_level:
            # if there is no granularity (only on node), then the problem is formulated as decentral (below)
            continue
        scenario_name = granularity_scenario_data[granularity_level]
        if 'mean' not in granularity_level:
            results_path = fledge.utils.get_results_path(
                'run_electric_grid_optimal_operation', granularity_level + '_central_' + der_penetration)
            opt_objective, opf_results, dlmps = run_centralized_problem(scenario_name, results_path)
            opf_results_dict['opt_objective_' + granularity_level + '_central'] = opt_objective
            opf_results_dict['opf_results_' + granularity_level + '_central'] = opf_results
            opf_results_dict['dlmps_' + granularity_level] = dlmps
        else:
            # for the mean DLMPs, we calculate them based on the the high granularity dlmps
            opf_results_dict['opt_objective_' + granularity_level + '_central'] = None
            opf_results_dict['opf_results_' + granularity_level + '_central'] = None
            scenario_name = granularity_scenario_data['high_granularity']
            mean_dlmps = calculate_mean_dlmps_for_lv_nodes(opf_results_dict['dlmps_' + 'high_granularity'], scenario_name)
            opf_results_dict['dlmps_' + granularity_level] = mean_dlmps

    # Run decentralized problem based on the DLMPs and one based on wholesale market price (no granularity)
    price_timeseries_dict = {}
    for granularity_level in granularity_scenario_data.keys():
        results_path = fledge.utils.get_results_path(
            'run_electric_grid_optimal_operation', granularity_level + '_decentral_' + der_penetration)
        scenario_name = granularity_scenario_data[granularity_level]
        # Obtain price data
        price_data = get_price_data_for_scenario(scenario_name)
        # Change price time series to dlmps
        if 'no_granularity' not in granularity_level:
            # Change price timeseries to DLMPs that were calculated in centralized problem
            dlmps = opf_results_dict['dlmps_' + granularity_level]
            if dlmps is None:
                continue
            price_data.price_timeseries = dlmps['electric_grid_total_dlmp_price_timeseries']
        opt_objective, opf_results = run_decentralized_problem(
            scenario_name, results_path, price_data)
        opf_results_dict['opt_objective_' + granularity_level + '_decentral'] = opt_objective
        opf_results_dict['opf_results_' + granularity_level + '_decentral'] = opf_results
        price_timeseries_dict[granularity_level] = price_data.price_timeseries

    # Get the set points from decentralized problems and calculate nominal power flow
    # now using the entire grid again
    pf_results_dict = {}
    der_costs_revenues_dict = {}
    system_costs = {}
    scenario_name = granularity_scenario_data['high_granularity']
    for key in opf_results_dict.keys():
        if ('decentral' in key) and ('opf_results' in key):
            results_path = fledge.utils.get_results_path('run_electric_grid_nominal_operation', key + '_power_flow_' + der_penetration)
            # change the set points of the DERs
            try:
                der_model_set_new_setpoints = change_der_set_points_based_on_results(scenario_name, opf_results_dict[key])
            except KeyError:
                continue
            # run the nominal power flow and store results in dictionary
            pf_results_dict[get_scenario_string(key)] = run_nominal_operation(scenario_name, der_model_set_new_setpoints,
                                                                              results_path)
            # calculate the system costs
            system_costs[get_scenario_string(key)] = calculate_system_costs_at_source_node(scenario_name, der_model_set_new_setpoints)
            # calculate individual DER costs / revenues
            # TODO: this is currently only working for flexible and fixed loads, no generators, etc.!!!
            der_costs_revenues_dict[get_scenario_string(key)] = \
                price_timeseries_dict[get_scenario_string(key)].loc[:,
                ('active_power', ['fixed_load', 'flexible_load'], slice(None))] \
                * pf_results_dict[get_scenario_string(key)]['der_power_magnitude'].values

    # Gather all result data in a dictionary and generate plots
    results_dict = {}
    if plots:
        # Combine all results in one big dictionary
        results_dict = {'opf_results': opf_results_dict,
                        'pf_results': pf_results_dict,
                        'price_timeseries': price_timeseries_dict,
                        'system_costs': system_costs,
                        'der_costs_revenues': der_costs_revenues_dict}
        # Pass to plot function
        results_path = fledge.utils.get_results_path('dlmp_analysis_plots', der_penetration)
        generate_result_plots(results_dict, granularity_scenario_data, results_path)

    # TODO: repeat the above with higher DER penetration
    # DERs can be modified based on the code example in function change_der_set_points...

    print('Done.')
    return results_dict


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
    feasible = solve_optimization_problem(optimization_problem)
    if not feasible:
        return [None, None, None]

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
    if optimization_problem is None:
        return [None, None]

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
) -> bool:
    # Solve decentralized DER optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    if solver is 'default':
        optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
    else:
        optimization_solver = pyo.SolverFactory('cplex', executable=path_to_solver_executable)
    optimization_result = optimization_solver.solve(optimization_problem,
                                                    tee=fledge.config.config['optimization']['show_solver_output'])
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        print('######### COULD NOT SOLVE OPTIMIZATION #########')
        print(f"Solver termination condition: {optimization_result.solver.termination_condition}")
        return False
        # raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")

    return True


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


def calculate_mean_dlmps_for_lv_nodes(
        dlmps: fledge.data_interface.ResultsDict,
        scenario_name: str
) -> fledge.data_interface.ResultsDict:
    if dlmps is None:
        return None
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    mean_dlmps = dlmps.copy()
    lv_der_names = electric_grid_model.der_names[electric_grid_model.der_names.str.contains('_')]
    # find the MV node names in the LV ders:
    mv_nodes = []
    for der_name in lv_der_names:
        split_der_name = der_name.split('_')
        mv_nodes.append(split_der_name[0])
    mv_nodes = list(set(mv_nodes))  # remove duplicates
    dlmp_type = 'electric_grid_total_dlmp_price_timeseries'
    for mv_node in mv_nodes:
        lv_der_names_at_node = lv_der_names[lv_der_names.str.contains(mv_node + '_')]
        mean_dlmps[dlmp_type].loc[:, ('active_power', slice(None), lv_der_names_at_node)] = \
            mean_dlmps[dlmp_type].loc[:, ('active_power', slice(None), lv_der_names_at_node)].apply(
            lambda x: dlmps[dlmp_type].loc[:, ('active_power', slice(None), lv_der_names_at_node)].mean(axis=1))
        mean_dlmps[dlmp_type].loc[:, ('reactive_power', slice(None), lv_der_names_at_node)] = \
            mean_dlmps[dlmp_type].loc[:, ('reactive_power', slice(None), lv_der_names_at_node)].apply(
            lambda x: dlmps[dlmp_type].loc[:, ('reactive_power', slice(None), lv_der_names_at_node)].mean(axis=1))
    return mean_dlmps


def generate_result_plots(
        results_dict: dict,
        granularity_scenario_data: dict,
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
    # TODO: [ ] load flow calculation comparison --> ask Sebastian
    pf_results = results_dict['pf_results']
    system_costs = results_dict['system_costs']
    opf_results = results_dict['opf_results']
    price_timeseries = results_dict['price_timeseries']

    # Get the electric grid model of the high granularity (was the same for all power flow calculations)
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(
        granularity_scenario_data['high_granularity'])
    scenario_data = fledge.data_interface.ScenarioData(granularity_scenario_data['high_granularity'])  # only used for timesteps

    # Branch power magnitude for both directions and node voltage
    plots = {
        'Branch power (direction 1) magnitude [p.u.]': 'branch_power_vector_1',
        'Branch power (direction 2) magnitude [p.u.]': 'branch_power_vector_2',
        'Node voltage magnitude [p.u.]': 'node_voltage_magnitude_per_unit',
        'DER power magnitude [p.u.]': 'der_power_magnitude_per_unit',
    }
    scenarios = get_list_of_scenario_strings(pf_results)
    for plot in plots.keys():
        minimum = None
        maximum = None
        if 'Node voltage' in plot:
            x_index = electric_grid_model.nodes
            minimum = voltage_min
            maximum = voltage_max
        elif 'Branch power' in plot:
            x_index = electric_grid_model.branches
            minimum = 0.0
            maximum = 1.0
        elif 'DER' in plot:
            x_index = electric_grid_model.ders
            minimum = 0.0
            maximum = 1.0
        for timestep in scenario_data.timesteps:
            plt.figure()
            plt.title(plot + ' at: ' + timestep.strftime("%H-%M-%S"))
            marker_index = 4
            for scenario_name in scenarios:
                pf_results_scenario_name = [key for key in pf_results.keys() if scenario_name in key][0]
                pf_results_of_scenario = pf_results[pf_results_scenario_name]
                if 'Branch power' in plot:
                    y_values = np.abs(pf_results_of_scenario[plots[plot]].loc[timestep, :].reindex(x_index).values ** 2)
                    y_values = y_values / (electric_grid_model.branch_power_vector_magnitude_reference ** 2)
                else:
                    y_values = pf_results_of_scenario[plots[plot]].loc[timestep, :].reindex(x_index).values
                plt.scatter(
                    range(len(x_index)),
                    y_values,
                    marker=marker_index,
                    label=scenario_name
                )
                marker_index += 1
            if (minimum is not None) and (maximum is not None):
                label_min = 'Minimum'
                label_max = 'Maximum'
                plt.plot([range(len(x_index))[0], range(len(x_index))[-1]], [minimum, minimum], 'k-', color='r',
                         label=label_min)
                plt.plot([range(len(x_index))[0], range(len(x_index))[-1]], [maximum, maximum], 'k-', color='r',
                         label=label_max)
                # plt.ylim((minimum, maximum))
            handles, labels = plt.gca().get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            plt.legend(handles, labels)
            plt.xticks(
                range(len(x_index)),
                x_index,
                rotation=45,
                ha='right'
            )
            plt.grid()
            plt.tight_layout()
            save_or_show_plot(plot, results_path, timestep)

    # Branch power magnitude for both directions and node voltage over time
    plots = {
        'Branch power (direction 1) magnitude [p.u.] for ': 'branch_power_vector_1',
        'Branch power (direction 2) magnitude [p.u.] for ': 'branch_power_vector_2',
        'Node voltage magnitude [p.u.] for node ': 'node_voltage_magnitude_per_unit',
        'DER power magnitude [p.u.] for ': 'der_power_magnitude_per_unit',
    }
    minimum = None
    maximum = None
    for plot in plots.keys():
        if 'Node voltage' in plot:
            assets = electric_grid_model.nodes
            minimum = voltage_min
            maximum = voltage_max
        elif 'Branch power' in plot:
            assets = electric_grid_model.branches
            minimum = 0.0
            maximum = 1.0
        elif 'DER' in plot:
            assets = electric_grid_model.ders
            minimum = 0.0
            maximum = 1.0
        for asset in assets:
            plt.figure()
            plt.title(plot + asset[0] + ' :' + asset[1])
            marker_index = 4
            for scenario_name in scenarios:
                pf_results_scenario_name = [key for key in pf_results.keys() if scenario_name in key][0]
                pf_results_of_scenario = pf_results[pf_results_scenario_name]
                if 'Branch power' in plot:
                    y_values = np.abs(pf_results_of_scenario[plots[plot]].loc[:, (asset[0],  asset[1], slice(None))] ** 2)
                    branch_index = fledge.utils.get_index(electric_grid_model.branches, branch_type=asset[0], branch_name=asset[1])
                    y_values = y_values / (electric_grid_model.branch_power_vector_magnitude_reference[branch_index] ** 2)
                else:
                    y_values = pf_results_of_scenario[plots[plot]].loc[:, (asset[0],  asset[1], slice(None))]
                plt.plot(
                    scenario_data.timesteps,
                    y_values,
                    marker=marker_index,
                    label=scenario_name
                )
                marker_index += 1
            if (minimum is not None) and (maximum is not None):
                label_min = 'Minimum'
                label_max = 'Maximum'
                plt.plot([scenario_data.timesteps[0], scenario_data.timesteps[-1]], [minimum, minimum], 'k-', color='r',
                         label=label_min)
                plt.plot([scenario_data.timesteps[0], scenario_data.timesteps[-1]], [maximum, maximum], 'k-', color='r',
                         label=label_max)
            handles, labels = plt.gca().get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            plt.legend(handles, labels)
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
    der_model_set = fledge.der_models.DERModelSet(granularity_scenario_data['high_granularity'])
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

    # DLMPs at time step over all nodes
    plots = {
        'Nodal Prices in EUR per kWh': '',
    }
    for plot in plots.keys():
        if 'Nodal Prices' in plot:
            x_index = der_model_set.der_names
        for timestep in scenario_data.timesteps:
            plt.figure()
            plt.title(plot + ' at: ' + timestep.strftime("%H-%M-%S"))
            marker_index = 4
            for scenario_name in scenarios:
                prices_at_timestep = price_timeseries[scenario_name].loc[timestep, ('active_power', ['fixed_load', 'flexible_load'], slice(None))]
                plt.scatter(
                    range(len(x_index)),
                    prices_at_timestep.values * 1e3,
                    marker=marker_index,
                    label=scenario_name
                )
                marker_index += 1
            handles, labels = plt.gca().get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            plt.legend(handles, labels)
            plt.xticks(
                range(len(x_index)),
                x_index,
                rotation=45,
                ha='right'
            )
            plt.grid()
            plt.tight_layout()
            save_or_show_plot(plot, results_path, timestep)

    # DLMP validation plots
    try:
        wholesale_price = price_timeseries['no_granularity']
        price_timeseries_dlmps = price_timeseries['high_granularity']
        central_opf_results = opf_results['opf_results_high_granularity_central']
        decentral_opf_results = opf_results['opf_results_high_granularity_decentral']
    except KeyError:
        print('Missing data! It seems that one of the optimization problems was infeasible and returned no results. Cannot generate plots!')
        return
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
    results_dict = main()
