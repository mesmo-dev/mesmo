"""Script for analyzing the impact of the granularity of a price signal"""

import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt

import fledge.config
import fledge.data_interface
import fledge.utils
import fledge.der_models
import fledge.electric_grid_models
import fledge.problems
import fledge.analysis_utils as au


# Global Settings
voltage_min = 0.5
voltage_max = 1.5
path_to_solver_executable = '/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx/cplex'


def main():

    plots = True  # will generate plots if set to True
    regenerate_grids = False  # will regenerate the grid input data

    # Define grid / scenario names
    mv_grid_name = 'cigre_mv_network'
    high_granularity_scenario_name = 'cigre_high_granularity'
    low_granularity_scenario_name = 'cigre_low_granularity'
    no_granularity_scenario_name = low_granularity_scenario_name  # use same scenario, only without the actual grid

    # Generate the grids that are needed for different granularity levels (comment out if not needed)
    if regenerate_grids:
        path_to_grid_map = 'examples/electric_grid_mapping.csv'
        au.combine_electric_grids(mv_grid_name, path_to_grid_map, high_granularity_scenario_name)
        au.aggregate_electric_grids(mv_grid_name, path_to_grid_map, low_granularity_scenario_name)

    # Recreate / overwrite database, to incorporate the new grids that we created
    fledge.data_interface.recreate_database()

    # TODO: check if price_sensitivity_coefficient is being used and if it is set correctly?
    # Run centralized problems for all granularity levels
    scenarios = [high_granularity_scenario_name, low_granularity_scenario_name]
    opf_results_dict = {}
    for scenario_name in scenarios:
        results_path = fledge.utils.get_results_path(
            'run_electric_grid_optimal_operation', scenario_name + '_central')
        opt_results, opf_results, dlmps = run_centralized_problem(scenario_name, results_path)
        opf_results_dict['opt_results_' + scenario_name + '_central'] = opt_results
        opf_results_dict['opf_results_' + scenario_name + '_central'] = opf_results
        opf_results_dict['dlmps_' + scenario_name] = dlmps
        # results = run_nominal_operation(scenario_name, results_path)

    # Run decentralized problem based on the DLMPs and one based on wholesale market price (no granularity)
    # TODO: Test scenario with weighted average dlmp instead of MV-level DLMP
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
        opt_results, opf_results = run_decentralized_problem(
            scenario_name, results_path, price_data)
        opf_results_dict['opt_results_' + scenario_name + '_decentral'] = opt_results
        opf_results_dict['opf_results_' + scenario_name + '_decentral'] = opf_results
        price_timeseries_dict[scenario_name] = price_data.price_timeseries

    # then the "no granularity" scenario based on the wholesale market
    scenario_name = no_granularity_scenario_name
    price_data = get_price_data_for_scenario(scenario_name)
    results_path = fledge.utils.get_results_path('run_electric_grid_optimal_operation', scenario_name + '_decentral')
    opt_results, results = run_decentralized_problem(
        scenario_name, results_path, price_data)
    opf_results_dict['opt_results_' + 'cigre_no_granularity' + '_decentral'] = opt_results
    opf_results_dict['opf_results_' + 'cigre_no_granularity' + '_decentral'] = opf_results
    price_timeseries_dict['cigre_no_granularity'] = price_data.price_timeseries

    # Get the set points from decentralized problems and calculate nominal power flow
    # now using the entire grid again
    pf_results_dict = {}
    scenario_name = high_granularity_scenario_name
    for key in opf_results_dict.keys():
        if ('decentral' in key) and ('opf_results' in key):
            results_path = fledge.utils.get_results_path('run_electric_grid_nominal_operation', key + '_power_flow')
            # change the set points of the DERs
            der_model_set = change_der_set_points_based_on_results(scenario_name, opf_results_dict[key])
            # run the nominal power flow and store results in dictionary
            pf_results_dict['pf_results_' + key] = run_nominal_operation(scenario_name, der_model_set, results_path)

    # Gather all result data in a dictionary and generate plots
    if plots:
        # Combine all results in one big dictionary
        results_dict = {'opf_results': opf_results_dict,
                        'pf_results': pf_results_dict,
                        'price_timeseries': price_timeseries_dict}
        # Pass to plot function
        generate_result_plots(results_dict, high_granularity_scenario_name)

    # TODO: repeat the above with higher DER penetration
    # DERs can be modified based on the code example in function change_der_set_points...

    print('Done.')


def run_centralized_problem(
        scenario_name: str,
        results_path: str
) -> [object, fledge.data_interface.ResultsDict, fledge.data_interface.ResultsDict]:
    # This should return the results of the centralized optimization problem (incl. grid)

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
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
    optimization_solver = pyo.SolverFactory('cplex', executable=path_to_solver_executable)
    optimization_result = optimization_solver.solve(optimization_problem,
                                                    tee=fledge.config.config['optimization']['show_solver_output'])
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")

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

    return [optimization_result, results, dlmps]


def run_decentralized_problem(
        scenario_name: str,
        results_path: str,
        price_data: fledge.data_interface.PriceData
) -> [object, fledge.data_interface.ResultsDict]:
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
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
    optimization_solver = pyo.SolverFactory('cplex', executable=path_to_solver_executable)
    optimization_result = optimization_solver.solve(optimization_problem,
                                                    tee=fledge.config.config['optimization']['show_solver_output'])
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")

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

    return [optimization_result, results]


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
    # [] objective result / system costs
    # [] prices over time (see dlmp paper)
    pf_results = results_dict['pf_results']
    opf_results = results_dict['opf_results']
    price_timeseries = results_dict['price_timeseries']

    # Get the electric grid model of the high granularity (was the same for all power flow calculations)
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(
        high_granularity_scenario_name)
    scenario_data = fledge.data_interface.ScenarioData(high_granularity_scenario_name)  # only used for timesteps

    # Branch power magnitude for both directions and node voltage
    plots = {
        'Branch power (direction 1) magnitude [p.u.]': 'branch_power_1_magnitude_per_unit',
        'Branch power (direction 2) magnitude [p.u.]': 'branch_power_2_magnitude_per_unit',
        'Node voltage magnitude [p.u.]': 'node_voltage_magnitude_per_unit',
        'DER power magnitude [p.u.]': 'der_power_magnitude_per_unit',
    }
    for plot in plots.keys():
        scenarios = get_list_of_scenario_strings(pf_results)
        if 'Node' in plot:
            x_index = electric_grid_model.nodes
        elif 'Branch' in plot:
            x_index = electric_grid_model.branches.append(electric_grid_model.transformers)
        elif 'DER' in plot:
            x_index = electric_grid_model.ders
        for timestep in scenario_data.timesteps:
            plt.title(plot)
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
            plt.legend(scenarios)
            plt.grid()
            plt.tight_layout()
            # plt.savefig(os.path.join(results_path, 'nominal_branch_power_1_magnitude.png'))
            plt.show()
            plt.close()

    # TODO: generate objective result plot (system costs)
    # Generate plots that are independent of branches / nodes:
    plots = {
        'System losses': 'loss_magnitude',
        'Total system losses': 'loss_magnitude',
    }
    for plot in plots.keys():
        scenarios = get_list_of_scenario_strings(pf_results)
        if 'Total' in plot:
            x_index = ['total']
        else:
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
        # plt.savefig(os.path.join(results_path, 'nominal_branch_power_1_magnitude.png'))
        plt.show()
        plt.close()

    if results_path is None:
        # show plots - plt.show()
        pass
    else:
        # save figures to path
        pass


def get_list_of_scenario_strings(
        pf_results_dict: dict
) -> list:
    scenarios = []
    for key in pf_results_dict.keys():
        scenarios.append(key[len('pf_results_opf_results_'): -len('_decentral')])
    return scenarios


if __name__ == '__main__':
    main()
