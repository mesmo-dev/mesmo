"""Example script for DRO problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge


# To do 1. Deterministic MG S1 problem
# Need: Data set for reserves



def main():

    # Settings.
    scenario_name = 'singapore_6node'

    # Get results path.
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV definition files.
    fledge.data_interface.recreate_database()

    # Obtain price data object.
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Define reserve price # unit: SGD/MWh
    reserve_price = np.array([12.12, 13.11, 19.2, 19.6, 18.48, 24.33, 30, 30, 14.59, 24.33, 26.29, 13.33, 21.39, 21.16, 17.21,
                     16.30, 13.02, 11.03, 8, 8, 8, 7.45, 2.09, 1.2, 0.5])
    # convert to unit: SGD/10kWh
    reserve_price = 1e-2 * reserve_price

    stochastic_scenario_set = ['no_reserve', 'up_reserve_act', 'down_reserve_act']
    # Define energy price # unit: SGD/10kWh?
    energy_price = price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values

    time_size = 25
    # time_size = len(linear_electric_grid_model_scenario_o.electric_grid_model.timesteps)

    # Obtain DER & grid model objects.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Getting linear electric grid model using "global approximation" method.
    linear_electric_grid_model = fledge.electric_grid_models.LinearElectricGridModelGlobal(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define model variables.
    # Define flexible DER state space variables.
    optimization_problem.state_vector = {}
    optimization_problem.control_vector = {}
    optimization_problem.output_vector = {}
    optimization_problem.der_active_power_vector = {}
    optimization_problem.der_reactive_power_vector = {}
    # optimization_problem.der_thermal_power_vector = {}


    for stochastic_scenario_name in stochastic_scenario_set:
        for der_name in der_model_set.flexible_der_names:
            optimization_problem.state_vector[der_name, stochastic_scenario_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].states)
                ))
            )
            optimization_problem.control_vector[der_name, stochastic_scenario_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].controls)
                ))
            )
            optimization_problem.output_vector[der_name, stochastic_scenario_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].outputs)
                ))
            )

    for stochastic_scenario_name in stochastic_scenario_set:
        # Define DER power vector variables. remove thermal DERs
        optimization_problem.der_active_power_vector[stochastic_scenario_name] = (
            cp.Variable((len(der_model_set.timesteps), len(der_model_set.electric_ders)))
        )
        optimization_problem.der_reactive_power_vector[stochastic_scenario_name] = (
            cp.Variable((len(der_model_set.timesteps), len(der_model_set.electric_ders)))
        )

    # Define bids quantity variable.
    optimization_problem.balance_energy = cp.Variable(shape=(1, time_size))
    optimization_problem.up_reserve = cp.Variable(shape=(1, time_size))
    optimization_problem.down_reserve = cp.Variable(shape=(1, time_size))

    # Define node voltage variable.
    optimization_problem.node_voltage_magnitude_vector = dict.fromkeys(stochastic_scenario_set)

    for stochastic_scenario_name in stochastic_scenario_set:
        optimization_problem.node_voltage_magnitude_vector[stochastic_scenario_name] = (
            cp.Variable((len(linear_electric_grid_model.electric_grid_model.timesteps), len(linear_electric_grid_model.electric_grid_model.nodes)))
        )

    #cp.sum(optimization_problem.der_active_power_vector[stochastic_scenario_name], axis=1),

    # Define voltage constraints for all scenarios
    for stochastic_scenario_name in stochastic_scenario_set:
        optimization_problem.constraints.append(
            optimization_problem.node_voltage_magnitude_vector[stochastic_scenario_name]
            ==
            (
                    cp.transpose(
                        linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                        @ cp.transpose(cp.multiply(
                            optimization_problem.der_active_power_vector[stochastic_scenario_name],
                            np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                        ) - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
                        + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                        @ cp.transpose(cp.multiply(
                            optimization_problem.der_reactive_power_vector[stochastic_scenario_name],
                            np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                        ) - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])) # what is this for? I forget..
                    )
                    + np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
            )
            / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
        )

    # from multimethod import multimethod
    # @multimethod
    # def stupid_fun(
    #         arg1: str,
    #         arg2: float,
    # ):
    #     return arg1 + arg2
    #
    # @multimethod
    # def stupid_fun(
    #         arg1: float,
    #         arg2: float,
    #         alg=None
    # ):
    #     return arg1 + arg2
    #
    # out = stupid_fun(1, 10, alg=[0.9])

    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)

    for stochastic_scenario_name in stochastic_scenario_set:
        # Define up/lower bounds for voltage magnitude
        optimization_problem.voltage_magnitude_vector_minimum_constraint = (
                optimization_problem.node_voltage_magnitude_vector[stochastic_scenario_name]
                - np.array([node_voltage_magnitude_vector_minimum.ravel()])
                / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
                >=
                0.0
        )

        optimization_problem.constraints.append(optimization_problem.voltage_magnitude_vector_minimum_constraint)

        optimization_problem.voltage_magnitude_vector_maximum_constraint = (
                optimization_problem.node_voltage_magnitude_vector[stochastic_scenario_name]
                - np.array([node_voltage_magnitude_vector_maximum.ravel()])
                / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
                <=
                0.0
        )
        optimization_problem.constraints.append(optimization_problem.voltage_magnitude_vector_maximum_constraint)

    # # Define DER constraints...

    for der_name, der_model in der_model_set.der_models.items():
        # Fixed DERs
        if issubclass(type(der_model), fledge.der_models.FixedDERModel):
            if der_model.is_electric_grid_connected:
                for stochastic_scenario_name in stochastic_scenario_set:
                    if hasattr(optimization_problem, 'der_active_power_vector'):
                        optimization_problem.constraints.append(
                            optimization_problem.der_active_power_vector[stochastic_scenario_name][:, der_model.electric_grid_der_index]
                            ==
                            np.transpose([der_model.active_power_nominal_timeseries.values])
                            / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)
                        )
                    if hasattr(optimization_problem, 'der_reactive_power_vector'):
                        optimization_problem.constraints.append(
                            optimization_problem.der_reactive_power_vector[stochastic_scenario_name][:, der_model.electric_grid_der_index]
                            ==
                            np.transpose([der_model.reactive_power_nominal_timeseries.values])
                            / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)
                        )
        # Flexi DERs
        elif issubclass(type(der_model), fledge.der_models.FlexibleDERModel):
            for stochastic_scenario_name in stochastic_scenario_set:
                # Initial state.
                # - For states which represent storage state of charge, initial state of charge is final state of charge.
                if any(der_model.states.isin(der_model.storage_states)):
                    optimization_problem.constraints.append(
                        optimization_problem.state_vector[der_name, stochastic_scenario_name][0, der_model.states.isin(der_model.storage_states)]
                        ==
                        optimization_problem.state_vector[der_name, stochastic_scenario_name][-1, der_model.states.isin(der_model.storage_states)]
                    )
                # - For other states, set initial state according to the initial state vector.
                if any(~der_model.states.isin(der_model.storage_states)):
                    optimization_problem.constraints.append(
                        optimization_problem.state_vector[der_name, stochastic_scenario_name][0, ~der_model.states.isin(der_model.storage_states)]
                        ==
                        der_model.state_vector_initial.loc[~der_model.states.isin(der_model.storage_states)].values
                    )

                # State equation.
                optimization_problem.constraints.append(
                    optimization_problem.state_vector[der_name, stochastic_scenario_name][1:, :]
                    ==
                    cp.transpose(
                        der_model.state_matrix.values
                        @ cp.transpose(optimization_problem.state_vector[der_name, stochastic_scenario_name][:-1, :])
                        + der_model.control_matrix.values
                        @ cp.transpose(optimization_problem.control_vector[der_name, stochastic_scenario_name][:-1, :])
                        + der_model.disturbance_matrix.values
                        @ np.transpose(der_model.disturbance_timeseries.iloc[:-1, :].values)
                    )
                )

                # Output limits.
                outputs_minimum_infinite = (
                    (der_model.output_minimum_timeseries == -np.inf).all()
                )
                optimization_problem.constraints.append(
                    optimization_problem.output_vector[der_name, stochastic_scenario_name][:, ~outputs_minimum_infinite]
                    >=
                    der_model.output_minimum_timeseries.loc[:, ~outputs_minimum_infinite].values
                )
                outputs_maximum_infinite = (
                    (der_model.output_maximum_timeseries == np.inf).all()
                )
                optimization_problem.constraints.append(
                    optimization_problem.output_vector[der_name, stochastic_scenario_name][:, ~outputs_maximum_infinite]
                    <=
                    der_model.output_maximum_timeseries.loc[:, ~outputs_maximum_infinite].values
                )

                # Define connection constraints.
                if der_model.is_electric_grid_connected:
                    if hasattr(optimization_problem, 'der_active_power_vector'):
                        optimization_problem.constraints.append(
                            optimization_problem.der_active_power_vector[stochastic_scenario_name][:, der_model.electric_grid_der_index]
                            ==
                            cp.transpose(
                                der_model.mapping_active_power_by_output.values
                                @ cp.transpose(optimization_problem.output_vector[der_name, stochastic_scenario_name])
                            )
                            / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)
                        )
                    if hasattr(optimization_problem, 'der_reactive_power_vector'):
                        optimization_problem.constraints.append(
                            optimization_problem.der_reactive_power_vector[stochastic_scenario_name][:, der_model.electric_grid_der_index]
                            ==
                            cp.transpose(
                                der_model.mapping_reactive_power_by_output.values
                                @ cp.transpose(optimization_problem.output_vector[der_name, stochastic_scenario_name])
                            )
                            / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)
                        )


    # # Define power balance constraints # how to get the power injections from load nodes
    # To do: remove the source node in the vector
    for stochastic_scenario_name in stochastic_scenario_set:
        if stochastic_scenario_name == 'no_reserve':
            optimization_problem.constraints.append(
                optimization_problem.balance_energy == cp.sum(optimization_problem.der_active_power_vector[stochastic_scenario_name], axis=1)
            )
        elif stochastic_scenario_name == 'up_reserve_act':
            optimization_problem.constraints.append(
                optimization_problem.balance_energy + optimization_problem.up_reserve == cp.sum(
                    optimization_problem.der_active_power_vector[stochastic_scenario_name], axis=1)
            )
        else:
            optimization_problem.constraints.append(
                optimization_problem.balance_energy - optimization_problem.down_reserve == cp.sum(
                    optimization_problem.der_active_power_vector[stochastic_scenario_name], axis=1)
            )


    # for timestep in self.timesteps:
    #     optimization_problem.constraints.append(
    #         optimization_problem.up_reserve[0, timestep]>=0
    #     )
    #     optimization_problem.constraints.append(
    #         optimization_problem.down_reserve[0, timestep]>=0
    #     )
    #
    #
    # Define model objective.
    #   der_model_set.define_optimization_objective(optimization_problem, price_data)
    #   linear_electric_grid_model.define_optimization_objective(optimization_problem, price_data)
    # Define custom objective.
    #optimization_problem.objective += 1e2 * cp.sum(optimization_problem.electric_power_peak)
    optimization_problem.objective += (
            (
                    price_data.price_timeseries.loc[:, ('thermal_power', 'source', 'source')].values.T
                    * timestep_interval_hours  # In Wh.
                    @ cp.sum(-1.0 * (
                    cp.multiply(
                        optimization_problem.der_thermal_power_vector,
                        np.array([self.thermal_grid_model.der_thermal_power_vector_reference])
                    )
                    / self.thermal_grid_model.cooling_plant_efficiency
            ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
            )
            + ((
                       price_data.price_sensitivity_coefficient
                       * timestep_interval_hours  # In Wh.
                       * cp.sum((
                                        cp.multiply(
                                            optimization_problem.der_thermal_power_vector,
                                            np.array([self.thermal_grid_model.der_thermal_power_vector_reference])
                                        )
                                        / self.thermal_grid_model.cooling_plant_efficiency
                                ) ** 2)
               ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)


    # Solve optimization problem.
    optimization_problem.solve()

    # Get model results.
    results = fledge.problems.Results()
    results.update(der_model_set.get_optimization_results(optimization_problem))
    results.update(linear_electric_grid_model.get_optimization_results(optimization_problem))
    # Get custom results.
    electric_power_peak = optimization_problem.electric_power_peak.value
    print(f"electric_power_peak = {electric_power_peak}")

    # Store results to CSV.
    results.save(results_path)

    # Plot some results.
    for der_model in der_model_set.flexible_der_models.values():

        for output in der_model.outputs:

            figure = go.Figure()
            figure.add_scatter(
                x=der_model.output_maximum_timeseries.index,
                y=der_model.output_maximum_timeseries.loc[:, output].values,
                name='Maximum',
                line=go.scatter.Line(shape='hv')
            )
            figure.add_scatter(
                x=der_model.output_minimum_timeseries.index,
                y=der_model.output_minimum_timeseries.loc[:, output].values,
                name='Minimum',
                line=go.scatter.Line(shape='hv')
            )
            figure.add_scatter(
                x=results.output_vector.index,
                y=results.output_vector.loc[:, (der_model.der_name, output)].values,
                name='Optimal',
                line=go.scatter.Line(shape='hv')
            )
            figure.update_layout(
                title=f'DER: {der_model.der_name} / Output: {output}',
                xaxis=go.layout.XAxis(tickformat='%H:%M'),
                legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
            )
            # figure.show()
            fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'der_{der_model.der_name}_out_{output}'))

        for disturbance in der_model.disturbances:

            figure = go.Figure()
            figure.add_scatter(
                x=der_model.disturbance_timeseries.index,
                y=der_model.disturbance_timeseries.loc[:, disturbance].values,
                line=go.scatter.Line(shape='hv')
            )
            figure.update_layout(
                title=f'DER: {der_model.der_name} / Disturbance: {disturbance}',
                xaxis=go.layout.XAxis(tickformat='%H:%M'),
                showlegend=False
            )
            # figure.show()
            fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'der_{der_model.der_name}_dis_{disturbance}'))

    for commodity_type in ['active_power', 'reactive_power']:

        if commodity_type in price_data.price_timeseries.columns.get_level_values('commodity_type'):
            figure = go.Figure()
            figure.add_scatter(
                x=price_data.price_timeseries.index,
                y=price_data.price_timeseries.loc[:, (commodity_type, 'source', 'source')].values,
                line=go.scatter.Line(shape='hv')
            )
            figure.update_layout(
                title=f'Price: {commodity_type}',
                xaxis=go.layout.XAxis(tickformat='%H:%M')
            )
            # figure.show()
            fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'price_{commodity_type}'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
