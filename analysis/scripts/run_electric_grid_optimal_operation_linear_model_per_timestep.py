"""
Example script for setting up and solving an electric grid optimal operation problem using a linear model for every
timestep.
"""

import numpy as np
import pandas as pd
import itertools

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.problems
import fledge.utils


def main():

    # ---------------------------------------------------------------------------------------------------------
    # SETTINGS
    scenario_name = 'singapore_6node'
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Power flow voltage and branch flow limits
    voltage_max = 1.1
    voltage_min = 0.9
    branch_flow_max = 1.0
    # Custom constrained branches, branch_name and factor for branch
    constrained_branches = {
        # '2': 0.8
    }

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    print('Loading data...', end='\r')
    # fledge.data_interface.recreate_database()

    # ---------------------------------------------------------------------------------------------------------
    # OBTAIN DATA
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)

    # ---------------------------------------------------------------------------------------------------------
    # PRE-SOLVE
    # Pre-solve optimal operation problem without underlying electric grid to get realistic initial values
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    presolve_results = run_presolve_with_der_models(der_model_set, price_data)
    der_model_set = change_der_set_points_based_on_results(der_model_set, presolve_results)

    # ---------------------------------------------------------------------------------------------------------
    # POWER FLOW AND LINEAR ELECTRIC GRID MODEL
    # Obtain the base power flow, using the values from the presolved optmization given as input as initial dispatch
    # quantities.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solutions_per_timestep = get_power_flow_solutions_per_timestep(
        electric_grid_model=electric_grid_model,
        der_model_set_new_setpoints=der_model_set,
        timesteps=scenario_data.timesteps
    )
    # Get linear electric grid model for all timesteps
    linear_electric_grid_models_per_timestep = get_linear_electric_grid_models_per_timestep(
        electric_grid_model,
        power_flow_solutions_per_timestep,
        scenario_data.timesteps)
    # Get the first linear electric grid model for the next function calls
    linear_electric_grid_model = linear_electric_grid_models_per_timestep[scenario_data.timesteps[0]]

    # Get DER model set
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # ---------------------------------------------------------------------------------------------------------
    # OPTIMAL POWER FLOW
    print('Formulating optimization problem...', end='\r')
    # Instantiate optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define optimization variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )
    der_model_set.define_optimization_variables(
        optimization_problem
    )

    # Define constraints.
    node_voltage_magnitude_vector_minimum = voltage_min * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = voltage_max * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = branch_flow_max * electric_grid_model.branch_power_vector_magnitude_reference
    # Custom constraint on one specific line
    for branch_name in constrained_branches:
        branch_power_magnitude_vector_maximum[
            fledge.utils.get_index(electric_grid_model.branches, branch_name=branch_name)
        ] *= constrained_branches[branch_name]
    for timestep in scenario_data.timesteps:
        linear_electric_grid_models_per_timestep[timestep].define_optimization_constraints(
            optimization_problem=optimization_problem,
            timesteps=scenario_data.timesteps,
            node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
            node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
            branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
        )
    der_model_set.define_optimization_constraints(
        optimization_problem,
        electric_grid_model=electric_grid_model
    )

    # add constraints for DERs
    define_custom_constraints_for_ders(der_model_set)

    # Define objective.
    linear_electric_grid_model.define_optimization_objective(
        optimization_problem,
        price_data,
        scenario_data.timesteps
    )
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_data
    )

    print('Solving optimal power flow...', end='\r')
    # Solve optimization problem.
    optimization_problem.solve()

    # ---------------------------------------------------------------------------------------------------------
    # RESULTS
    # Obtain results.
    results = fledge.problems.Results()
    results.update(
        linear_electric_grid_model.get_optimization_results(
            optimization_problem,
            None,
            scenario_data.timesteps,
        )
    )
    results.update(
        der_model_set.get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    print(results)

    # Store results to CSV.
    results.save(results_path)

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

    # Store DLMPs to CSV.
    dlmps.save(results_path)

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


def define_custom_constraints_for_ders(
        der_model_set: fledge.der_models.DERModelSet
):
    # Add constraint on electricity use of flex building
    for der_name in der_model_set.der_models.keys():
        der_model = der_model_set.der_models[der_name]
        if type(der_model) is fledge.der_models.FlexibleBuildingModel:
            der_model.output_maximum_timeseries['grid_electric_power'] = (
                    (-1) * der_model.active_power_nominal_timeseries
            )
            # Put a constraint on cooling power (= 0) to effectively disable cooling in the HVAC system
            if 'zone_generic_cool_thermal_power_cooling' in der_model.output_maximum_timeseries.columns:
                der_model.output_maximum_timeseries['zone_generic_cool_thermal_power_cooling'] = 0


def run_presolve_with_der_models(
        der_model_set: fledge.der_models.DERModelSet,
        price_data: fledge.data_interface.PriceData
) -> fledge.problems.Results:
    # Pre-solve optimal operation problem without underlying electric grid to get realistic initial values
    # Obtain all DERs
    print('Running pre-solve for der models only...', end='\r')

    # Turn off solver output for pre-solve
    # Store original solver output value
    show_solver_output_original = fledge.config.config['optimization']['show_solver_output']
    fledge.config.config['optimization']['show_solver_output'] = False
    # Instantiate decentralized DER optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem
    )
    # Add constraint on electricity use of flex building
    define_custom_constraints_for_ders(der_model_set)

    # Define objective (DER operation cost minimization).
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_data
    )

    # Solve decentralized DER optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = fledge.problems.Results()
    results.update(
        der_model_set.get_optimization_results(
            optimization_problem
        )
    )
    # Set back to original value
    fledge.config.config['optimization']['show_solver_output'] = show_solver_output_original
    return results


def change_der_set_points_based_on_results(
        der_model_set: fledge.der_models.DERModelSet,
        results: fledge.problems.Results
) -> fledge.der_models.DERModelSet:
    attributes = dir(results)
    if 'der_active_power_vector' in attributes:
        for der_name in der_model_set.der_names:
            der_model = der_model_set.der_models[der_name]
            der_type = der_model.der_type
            der_model.active_power_nominal_timeseries = (
                results.der_active_power_vector[der_type, der_name]
            )
            der_model.reactive_power_nominal_timeseries = (
                results.der_reactive_power_vector[der_type, der_name]
            )
    # If there was no electric grid model in the optimization, get the results based on the output vector
    elif 'output_vector' in attributes:
        for der_name in der_model_set.der_names:
            der_model = der_model_set.der_models[der_name]
            if issubclass(type(der_model), fledge.der_models.FlexibleDERModel):
                if 'active_power' in results.output_vector[der_name].columns:
                    der_model.active_power_nominal_timeseries = (
                        results.output_vector[(der_name, 'active_power')]
                    )
                    der_model.reactive_power_nominal_timeseries = (
                        results.output_vector[(der_name, 'reactive_power')]
                    )
                elif 'grid_electric_power' in results.output_vector[der_name].columns:
                    der_model.active_power_nominal_timeseries = (
                        results.output_vector[(der_name, 'grid_electric_power')]
                    ) * (-1)
                    if type(der_model) is fledge.der_models.FlexibleBuildingModel:
                        power_factor = der_model.power_factor_nominal
                    else:
                        power_factor = 0.95
                    der_model.reactive_power_nominal_timeseries = (
                        results.output_vector[(der_name, 'grid_electric_power')] * np.tan(np.arccos(power_factor))
                    ) * (-1)
    else:
        print('Results object does not contain any data on active power output. ')
        raise ValueError

    return der_model_set


def get_der_power_vector(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        der_models_set: fledge.der_models.DERModelSet,
        timesteps: pd.Index
) -> pd.DataFrame:
    der_power_vector = (
        pd.DataFrame(columns=electric_grid_model.ders, index=timesteps, dtype=np.complex)
    )
    for der in electric_grid_model.ders:
        der_name = der[1]
        der_power_vector.loc[:, der] = (
                der_models_set.der_models[der_name].active_power_nominal_timeseries
                + (1.0j * der_models_set.der_models[der_name].reactive_power_nominal_timeseries)
        )
    return der_power_vector


def get_power_flow_solutions_per_timestep(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        der_model_set_new_setpoints: fledge.der_models.DERModelSet,
        timesteps: pd.Index
):
    der_power_vector = get_der_power_vector(electric_grid_model, der_model_set_new_setpoints, timesteps)
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


def get_linear_electric_grid_models_per_timestep(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        power_flow_solutions: dict,
        timesteps
) -> dict:
    print('Obtaining linear electric grid model for all timesteps...', end='\r')
    # TODO: adapt to Local Approx: LinearElectricGridModelLocal
    linear_electric_grid_models = (
        fledge.utils.starmap(
            fledge.electric_grid_models.LinearElectricGridModelGlobal,
            zip(
                itertools.repeat(electric_grid_model),
                list(power_flow_solutions.values())
            )
        )
    )
    linear_electric_grid_models = dict(zip(timesteps, linear_electric_grid_models))
    # Assign corresponding timestep to the linear electric grid model attribute
    for timestep in timesteps:
        linear_electric_grid_models[timestep].timestep = timestep

    return linear_electric_grid_models


if __name__ == '__main__':
    main()
