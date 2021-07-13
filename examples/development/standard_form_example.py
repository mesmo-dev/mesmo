"""Example script for setting up and solving a standard form flexible DER optimal operation problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp

import fledge


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    # fledge.data_interface.recreate_database()

    # Obtain data.
    der_data = fledge.data_interface.DERData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name, der_data)
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)

    # Obtain model.
    der_model_set = fledge.der_models.DERModelSet(der_data)
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    der_power_vector = pd.DataFrame(
        data=np.array([electric_grid_model.der_power_vector_reference for timestep in der_model_set.timesteps]),
        index=der_model_set.timesteps,
        columns=der_model_set.electric_ders
    )
    power_flow_solution_set = fledge.electric_grid_models.PowerFlowSolutionSet(electric_grid_model, der_power_vector)
    linear_electric_grid_model_set = (
        fledge.electric_grid_models.LinearElectricGridModelSet(
            electric_grid_model,
            power_flow_solution_set,
            linear_electric_grid_model_method=fledge.electric_grid_models.LinearElectricGridModelGlobal
        )
    )

    # Define grid limits.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference

    # Instantiate optimization problem.
    fledge.utils.log_time('standard-form interface')
    fledge.utils.log_time('standard-form problem')
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define linear electric grid model set problem.
    linear_electric_grid_model_set.define_optimization_variables(optimization_problem)
    linear_electric_grid_model_set.define_optimization_parameters(
        optimization_problem,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
    )
    linear_electric_grid_model_set.define_optimization_constraints(optimization_problem)
    linear_electric_grid_model_set.define_optimization_objective(optimization_problem, price_data)

    # Define DER model set problem.
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_parameters(optimization_problem)
    der_model_set.define_optimization_constraints(optimization_problem)
    fledge.utils.log_time('standard-form problem')

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results_1 = fledge.problems.Results()
    results_1.update(linear_electric_grid_model_set.get_optimization_results(optimization_problem))
    results_1.update(linear_electric_grid_model_set.get_optimization_dlmps(optimization_problem, price_data))
    results_1.update(der_model_set.get_optimization_results(optimization_problem))
    objective_1 = optimization_problem.evaluate_objective(optimization_problem.x_vector)
    objective_1_new = linear_electric_grid_model_set.evaluate_optimization_objective(results_1, price_data)
    fledge.utils.log_time('standard-form interface')

    # Instantiate optimization problem.
    fledge.utils.log_time('cvxpy interface')
    fledge.utils.log_time('cvxpy problem')
    optimization_problem_cvxpy = OptimizationProblemCVXPY()

    # Define electric grid model variables.
    optimization_problem_cvxpy.der_active_power_vector = (
        cp.Variable((
            len(linear_electric_grid_model.electric_grid_model.timesteps),
            len(linear_electric_grid_model.electric_grid_model.ders)
        ))
    )
    optimization_problem_cvxpy.der_reactive_power_vector = (
        cp.Variable((
            len(linear_electric_grid_model.electric_grid_model.timesteps),
            len(linear_electric_grid_model.electric_grid_model.ders)
        ))
    )
    optimization_problem_cvxpy.node_voltage_magnitude_vector = (
        cp.Variable((
            len(linear_electric_grid_model.electric_grid_model.timesteps),
            len(linear_electric_grid_model.electric_grid_model.nodes)
        ))
    )
    optimization_problem_cvxpy.branch_power_magnitude_vector_1 = (
        cp.Variable((
            len(linear_electric_grid_model.electric_grid_model.timesteps),
            len(linear_electric_grid_model.electric_grid_model.branches)
        ))
    )
    optimization_problem_cvxpy.branch_power_magnitude_vector_2 = (
        cp.Variable((
            len(linear_electric_grid_model.electric_grid_model.timesteps),
            len(linear_electric_grid_model.electric_grid_model.branches)
        ))
    )
    optimization_problem_cvxpy.loss_active = cp.Variable((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))
    optimization_problem_cvxpy.loss_reactive = cp.Variable((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))

    # Define DER model variables.
    optimization_problem_cvxpy.state_vector = {
        der_model.der_name: cp.Variable((len(der_model.timesteps), len(der_model.states)))
        for der_name, der_model in der_model_set.flexible_der_models.items()
    }
    optimization_problem_cvxpy.control_vector = {
        der_model.der_name: cp.Variable((len(der_model.timesteps), len(der_model.controls)))
        for der_name, der_model in der_model_set.flexible_der_models.items()
    }
    optimization_problem_cvxpy.output_vector = {
        der_model.der_name: cp.Variable((len(der_model.timesteps), len(der_model.outputs)))
        for der_name, der_model in der_model_set.flexible_der_models.items()
    }

    # Define electric grid model constraints.
    timestep_index = slice(None)
    # Define voltage equation.
    optimization_problem_cvxpy.constraints.append(
        optimization_problem_cvxpy.node_voltage_magnitude_vector[timestep_index, :]
        ==
        (
            cp.transpose(
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                @ cp.transpose(cp.multiply(
                    optimization_problem_cvxpy.der_active_power_vector[timestep_index, :],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                @ cp.transpose(cp.multiply(
                    optimization_problem_cvxpy.der_reactive_power_vector[timestep_index, :],
                    np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
            )
            + np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
        )
        / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
    )
    # Define branch flow equation.
    optimization_problem_cvxpy.constraints.append(
        optimization_problem_cvxpy.branch_power_magnitude_vector_1[timestep_index, :]
        ==
        (
            cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                @ cp.transpose(cp.multiply(
                    optimization_problem_cvxpy.der_active_power_vector[timestep_index, :],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
                + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                @ cp.transpose(cp.multiply(
                    optimization_problem_cvxpy.der_reactive_power_vector[timestep_index, :],
                    np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
            )
            + np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1.ravel())])
        )
        / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
    )
    optimization_problem_cvxpy.constraints.append(
        optimization_problem_cvxpy.branch_power_magnitude_vector_2[timestep_index, :]
        ==
        (
            cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                @ cp.transpose(cp.multiply(
                    optimization_problem_cvxpy.der_active_power_vector[timestep_index, :],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
                + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                @ cp.transpose(cp.multiply(
                    optimization_problem_cvxpy.der_reactive_power_vector[timestep_index, :],
                    np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
            )
            + np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2.ravel())])
        )
        / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
    )

    # Define loss equation.
    optimization_problem_cvxpy.constraints.append(
        optimization_problem_cvxpy.loss_active[timestep_index, :]
        ==
        cp.transpose(
            linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
            @ cp.transpose(cp.multiply(
                optimization_problem_cvxpy.der_active_power_vector[timestep_index, :],
                np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
            ) - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
            + linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
            @ cp.transpose(cp.multiply(
                optimization_problem_cvxpy.der_reactive_power_vector[timestep_index, :],
                np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
            ) - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
        )
        + np.real(linear_electric_grid_model.power_flow_solution.loss)
    )
    optimization_problem_cvxpy.constraints.append(
        optimization_problem_cvxpy.loss_reactive[timestep_index, :]
        ==
        cp.transpose(
            linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
            @ cp.transpose(cp.multiply(
                optimization_problem_cvxpy.der_active_power_vector[timestep_index, :],
                np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
            ) - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
            + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
            @ cp.transpose(cp.multiply(
                optimization_problem_cvxpy.der_reactive_power_vector[timestep_index, :],
                np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
            ) - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
        )
        + np.imag(linear_electric_grid_model.power_flow_solution.loss)
    )

    # Define voltage limits.
    optimization_problem_cvxpy.voltage_magnitude_vector_minimum_constraint = (
        optimization_problem_cvxpy.node_voltage_magnitude_vector
        - np.array([node_voltage_magnitude_vector_minimum.ravel()])
        / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
        >=
        0.0
    )
    optimization_problem_cvxpy.voltage_magnitude_vector_maximum_constraint = (
        optimization_problem_cvxpy.node_voltage_magnitude_vector
        - np.array([node_voltage_magnitude_vector_maximum.ravel()])
        / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
        <=
        0.0
    )
    optimization_problem_cvxpy.constraints.append(optimization_problem_cvxpy.voltage_magnitude_vector_maximum_constraint)

    # Define branch flow limits.
    optimization_problem_cvxpy.branch_power_magnitude_vector_1_minimum_constraint = (
        optimization_problem_cvxpy.branch_power_magnitude_vector_1
        + np.array([branch_power_magnitude_vector_maximum.ravel()])
        / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
        >=
        0.0
    )
    optimization_problem_cvxpy.constraints.append(
        optimization_problem_cvxpy.branch_power_magnitude_vector_1_minimum_constraint
    )
    optimization_problem_cvxpy.branch_power_magnitude_vector_1_maximum_constraint = (
        optimization_problem_cvxpy.branch_power_magnitude_vector_1
        - np.array([branch_power_magnitude_vector_maximum.ravel()])
        / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
        <=
        0.0
    )
    optimization_problem_cvxpy.constraints.append(
        optimization_problem_cvxpy.branch_power_magnitude_vector_1_maximum_constraint
    )
    optimization_problem_cvxpy.branch_power_magnitude_vector_2_minimum_constraint = (
        optimization_problem_cvxpy.branch_power_magnitude_vector_2
        + np.array([branch_power_magnitude_vector_maximum.ravel()])
        / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
        >=
        0.0
    )
    optimization_problem_cvxpy.constraints.append(
        optimization_problem_cvxpy.branch_power_magnitude_vector_2_minimum_constraint
    )
    optimization_problem_cvxpy.branch_power_magnitude_vector_2_maximum_constraint = (
        optimization_problem_cvxpy.branch_power_magnitude_vector_2
        - np.array([branch_power_magnitude_vector_maximum.ravel()])
        / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
        <=
        0.0
    )
    optimization_problem_cvxpy.constraints.append(
        optimization_problem_cvxpy.branch_power_magnitude_vector_2_maximum_constraint
    )

    # Define DER model constraints.
    for der_name, der_model in der_model_set.flexible_der_models.items():

        # Define constraints.
        # Initial state.
        # - For states which represent storage state of charge, initial state of charge is final state of charge.
        if any(der_model.states.isin(der_model.storage_states)):
            optimization_problem_cvxpy.constraints.append(
                optimization_problem_cvxpy.state_vector[der_model.der_name][0, der_model.states.isin(der_model.storage_states)]
                ==
                optimization_problem_cvxpy.state_vector[der_model.der_name][-1, der_model.states.isin(der_model.storage_states)]
            )
        # - For other states, set initial state according to the initial state vector.
        if any(~der_model.states.isin(der_model.storage_states)):
            optimization_problem_cvxpy.constraints.append(
                optimization_problem_cvxpy.state_vector[der_model.der_name][0, ~der_model.states.isin(der_model.storage_states)]
                ==
                der_model.state_vector_initial.loc[~der_model.states.isin(der_model.storage_states)].values
            )

        # State equation.
        optimization_problem_cvxpy.constraints.append(
            optimization_problem_cvxpy.state_vector[der_model.der_name][1:, :]
            ==
            cp.transpose(
                der_model.state_matrix.values
                @ cp.transpose(optimization_problem_cvxpy.state_vector[der_model.der_name][:-1, :])
                + der_model.control_matrix.values
                @ cp.transpose(optimization_problem_cvxpy.control_vector[der_model.der_name][:-1, :])
                + der_model.disturbance_matrix.values
                @ np.transpose(der_model.disturbance_timeseries.iloc[:-1, :].values)
            )
        )

        # Output equation.
        optimization_problem_cvxpy.constraints.append(
            optimization_problem_cvxpy.output_vector[der_model.der_name]
            ==
            cp.transpose(
                der_model.state_output_matrix.values
                @ cp.transpose(optimization_problem_cvxpy.state_vector[der_model.der_name])
                + der_model.control_output_matrix.values
                @ cp.transpose(optimization_problem_cvxpy.control_vector[der_model.der_name])
                + der_model.disturbance_output_matrix.values
                @ np.transpose(der_model.disturbance_timeseries.values)
            )
        )

        # Output limits.
        outputs_minimum_infinite = (
            (der_model.output_minimum_timeseries == -np.inf).all()
        )
        optimization_problem_cvxpy.constraints.append(
            optimization_problem_cvxpy.output_vector[der_model.der_name][:, ~outputs_minimum_infinite]
            >=
            der_model.output_minimum_timeseries.loc[:, ~outputs_minimum_infinite].values
        )
        outputs_maximum_infinite = (
            (der_model.output_maximum_timeseries == np.inf).all()
        )
        optimization_problem_cvxpy.constraints.append(
            optimization_problem_cvxpy.output_vector[der_model.der_name][:, ~outputs_maximum_infinite]
            <=
            der_model.output_maximum_timeseries.loc[:, ~outputs_maximum_infinite].values
        )

        # Define connection constraints.
        optimization_problem_cvxpy.constraints.append(
            optimization_problem_cvxpy.der_active_power_vector[:, der_model.electric_grid_der_index]
            ==
            cp.transpose(
                der_model.mapping_active_power_by_output.values
                @ cp.transpose(optimization_problem_cvxpy.output_vector[der_model.der_name])
            )
            / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)
        )
        optimization_problem_cvxpy.constraints.append(
            optimization_problem_cvxpy.der_reactive_power_vector[:, der_model.electric_grid_der_index]
            ==
            cp.transpose(
                der_model.mapping_reactive_power_by_output.values
                @ cp.transpose(optimization_problem_cvxpy.output_vector[der_model.der_name])
            )
            / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)
        )

    # Obtain timestep interval in hours, for conversion of power to energy.
    timestep_interval_hours = (scenario_data.timesteps[1] - scenario_data.timesteps[0]) / pd.Timedelta('1h')

    # Define objective. 
    optimization_problem_cvxpy.objective += (
        (
            np.array([
                price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values[timestep_index]
            ])
            * timestep_interval_hours  # In Wh.
            @ cp.sum(-1.0 * (
                cp.multiply(
                    optimization_problem_cvxpy.der_active_power_vector[timestep_index, :],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                )
            ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
        )
        + ((
            price_data.price_sensitivity_coefficient
            * timestep_interval_hours  # In Wh.
            * cp.sum((
                cp.multiply(
                    optimization_problem_cvxpy.der_active_power_vector[timestep_index, :],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                )
            ) ** 2)
        ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
    )
    optimization_problem_cvxpy.objective += (
        (
            np.array([
                price_data.price_timeseries.loc[:, ('reactive_power', 'source', 'source')].values[timestep_index]
            ])
            * timestep_interval_hours  # In Wh.
            @ cp.sum(-1.0 * (
                cp.multiply(
                    optimization_problem_cvxpy.der_reactive_power_vector[timestep_index, :],
                    np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                )
            ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
        )
        + ((
            price_data.price_sensitivity_coefficient
            * timestep_interval_hours  # In Wh.
            * cp.sum((
                cp.multiply(
                    optimization_problem_cvxpy.der_reactive_power_vector[timestep_index, :],
                    np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                )
            ) ** 2)  # Sum along DERs, i.e. sum for each timestep.
        ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
    )
    optimization_problem_cvxpy.objective += (
        (
            np.array([
                price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values[timestep_index]
            ])
            * timestep_interval_hours  # In Wh.
            @ (
                optimization_problem_cvxpy.loss_active[timestep_index, :]
            )
        )
        + ((
            price_data.price_sensitivity_coefficient
            * timestep_interval_hours  # In Wh.
            * cp.sum((
                optimization_problem_cvxpy.loss_active[timestep_index, :]
            ) ** 2)
        ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
    )
    fledge.utils.log_time('cvxpy problem')

    # Solve optimization problem.
    fledge.utils.log_time('cvxpy solve')
    optimization_problem_cvxpy.solve()
    fledge.utils.log_time('cvxpy solve')

    # Instantiate results variables.
    fledge.utils.log_time('cvxpy get results')
    state_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.states)
    control_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.controls)
    output_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)

    # Obtain results.
    for der_name in der_model_set.flexible_der_names:
        state_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem_cvxpy.state_vector[der_name].value
        )
        control_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem_cvxpy.control_vector[der_name].value
        )
        output_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem_cvxpy.output_vector[der_name].value
        )
    results_2 = (
        fledge.problems.Results(
            state_vector=state_vector,
            control_vector=control_vector,
            output_vector=output_vector
        )
    )
    results_2.update(linear_electric_grid_model.get_optimization_results(optimization_problem_cvxpy))
    fledge.utils.log_time('cvxpy get results')
    fledge.utils.log_time('cvxpy interface')

    # Store results to CSV.
    results_2.save(results_path)

    # Plot results.
    for der_name, der_model in der_model_set.flexible_der_models.items():
        for output in der_model.outputs:

            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=der_model.output_maximum_timeseries.index,
                y=der_model.output_maximum_timeseries.loc[:, output].values,
                name='Maximum',
                line=go.scatter.Line(shape='hv')
            ))
            figure.add_trace(go.Scatter(
                x=der_model.output_minimum_timeseries.index,
                y=der_model.output_minimum_timeseries.loc[:, output].values,
                name='Minimum',
                line=go.scatter.Line(shape='hv')
            ))
            figure.add_trace(go.Scatter(
                x=results_1['output_vector'].index,
                y=results_1['output_vector'].loc[:, (der_name, output)].values,
                name='Optimal (standard form)',
                line=go.scatter.Line(shape='hv', width=4)
            ))
            figure.add_trace(go.Scatter(
                x=results_2['output_vector'].index,
                y=results_2['output_vector'].loc[:, (der_name, output)].values,
                name='Optimal (traditional form)',
                line=go.scatter.Line(shape='hv', width=2)
            ))
            figure.update_layout(
                title=f'DER: {der_name} / Output: {output}',
                xaxis=go.layout.XAxis(tickformat='%H:%M'),
                legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
            )
            # figure.show()
            fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'output_{der_name}_{output}'))

    for der_name, der_model in der_model_set.flexible_der_models.items():
        for disturbance in der_model.disturbances:

            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=der_model.disturbance_timeseries.index,
                y=der_model.disturbance_timeseries.loc[:, disturbance].values,
                line=go.scatter.Line(shape='hv')
            ))
            figure.update_layout(
                title=f'DER: {der_name} / Disturbance: {disturbance}',
                xaxis=go.layout.XAxis(tickformat='%H:%M'),
                showlegend=False
            )
            # figure.show()
            fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'disturbance_{der_name}_{disturbance}'))

    for commodity_type in ['active_power', 'reactive_power', 'thermal_power']:

        if commodity_type in price_data.price_timeseries.columns.get_level_values('commodity_type'):
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=price_data.price_timeseries.index,
                y=price_data.price_timeseries.loc[:, (commodity_type, 'source', 'source')].values,
                line=go.scatter.Line(shape='hv')
            ))
            figure.update_layout(
                title=f'Price: {commodity_type}',
                xaxis=go.layout.XAxis(tickformat='%H:%M')
            )
            # figure.show()
            fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'price_{commodity_type}'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


class OptimizationProblemCVXPY(object):
    """Optimization problem object for use with CVXPY."""

    constraints: list
    objective: cp.Expression
    has_der_objective: bool = False
    has_electric_grid_objective: bool = False
    has_thermal_grid_objective: bool = False
    cvxpy_problem: cp.Problem

    def __init__(self):

        self.constraints = []
        self.objective = cp.Constant(value=0.0)

    def solve(
            self,
            keep_problem=False,
            **kwargs
    ):

        # Instantiate CVXPY problem object.
        if hasattr(self, 'cvxpy_problem') and keep_problem:
            pass
        else:
            self.cvxpy_problem = cp.Problem(cp.Minimize(self.objective), self.constraints)

        # Solve optimization problem.
        self.cvxpy_problem.solve(
            solver=(
                fledge.config.config['optimization']['solver_name'].upper()
                if fledge.config.config['optimization']['solver_name'] is not None
                else None
            ),
            verbose=fledge.config.config['optimization']['show_solver_output'],
            **kwargs,
            **fledge.config.solver_parameters
        )

        # Assert that solver exited with an optimal solution. If not, raise an error.
        if not (self.cvxpy_problem.status == cp.OPTIMAL):
            raise cp.SolverError(f"Solver termination status: {self.cvxpy_problem.status}")


if __name__ == '__main__':
    main()
