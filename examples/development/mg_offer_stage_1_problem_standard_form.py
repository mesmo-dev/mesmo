"""Example script for DRO problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge


def main():
    # Settings.
    scenario_name = 'singapore_6node'
    stochastic_scenarios = ['no_reserve', 'up_reserve', 'down_reserve']
    variable_name_vector = ['energy', 'up_reserve', 'down_reserve', 'der_active_power', 'der_reactive_power',
                            'reactive_power_source', 'der_voltage_magnitude_no_reserve', 'der_control_vector',
                            'der_output_vector', 'der_state_vector']

    # Get results path.
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV definition files.
    fledge.data_interface.recreate_database()

    # Obtain price data object.
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain DER & grid model objects.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Getting linear electric grid model using "global approximation" method.
    linear_electric_grid_model = fledge.electric_grid_models.LinearElectricGridModelGlobal(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define bids quantity variable.
    optimization_problem.no_reserve = (
        cp.Variable((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))
    )
    optimization_problem.up_reserve = (
        cp.Variable((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))
    )
    optimization_problem.down_reserve = (
        cp.Variable((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))
    )

    optimization_problem.s1_index_locator = {}
    # vector definition
    optimization_problem.variable_vector_s1 = (
        cp.Variable((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))
    )
    # location specifier
    for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
        # location specifier
        optimization_problem.s1_index_locator['energy', str(time_index)] = [time_index, time_index]

    # vector definition

    temp_size = optimization_problem.variable_vector_s1.size
    optimization_problem.variable_vector_s1 = cp.vstack(
        (optimization_problem.variable_vector_s1,
         cp.Variable((len(linear_electric_grid_model.electric_grid_model.timesteps), 1)))
    )
    # location specifier
    for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
        optimization_problem.s1_index_locator['up_reserve', str(time_index)] = [temp_size + time_index,
                                                                                temp_size + time_index]

    # vector definition
    temp_size = optimization_problem.variable_vector_s1.size
    optimization_problem.variable_vector_s1 = cp.vstack(
        (optimization_problem.variable_vector_s1,
         cp.Variable((len(linear_electric_grid_model.electric_grid_model.timesteps), 1)))
    )
    # location specifier
    for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
        optimization_problem.s1_index_locator['down_reserve', str(time_index)] = [temp_size + time_index,
                                                                                  temp_size + time_index]

    # Define model variables.
    # Define flexible DER state space variables.
    optimization_problem.state_vector = {}
    optimization_problem.control_vector = {}
    optimization_problem.output_vector = {}
    optimization_problem.der_active_power_vector = {}
    optimization_problem.der_reactive_power_vector = {}

    for stochastic_scenario in stochastic_scenarios:
        for der_name in der_model_set.flexible_der_names:
            optimization_problem.state_vector[der_name, stochastic_scenario] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].states)
                ))
            )

            optimization_problem.control_vector[der_name, stochastic_scenario] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].controls)
                ))
            )
            optimization_problem.output_vector[der_name, stochastic_scenario] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].outputs)
                ))
            )

            # state vector in s1
            for time_index in range(len(der_model_set.flexible_der_models[der_name].timesteps)):
                temp_size = optimization_problem.variable_vector_s1.size
                optimization_problem.variable_vector_s1 = cp.vstack((
                    optimization_problem.variable_vector_s1, cp.Variable((
                        len(der_model_set.flexible_der_models[der_name].states), 1))
                ))

                # location specifier
                optimization_problem.s1_index_locator[der_name, 'der_state_vector', stochastic_scenario,
                                                      str(time_index)] = [temp_size,
                                                                          optimization_problem.variable_vector_s1.size - 1]
            # control vector in s1
            for time_index in range(len(der_model_set.flexible_der_models[der_name].timesteps)):
                temp_size = optimization_problem.variable_vector_s1.size
                optimization_problem.variable_vector_s1 = cp.vstack((
                    optimization_problem.variable_vector_s1, cp.Variable((
                        len(der_model_set.flexible_der_models[der_name].controls), 1))
                ))

                # location specifier
                optimization_problem.s1_index_locator[der_name, 'der_control_vector', stochastic_scenario,
                                                      str(time_index)] = [temp_size,
                                                                          optimization_problem.variable_vector_s1.size - 1]

            # output vector in s1
            for time_index in range(len(der_model_set.flexible_der_models[der_name].timesteps)):
                temp_size = optimization_problem.variable_vector_s1.size
                optimization_problem.variable_vector_s1 = cp.vstack((
                    optimization_problem.variable_vector_s1, cp.Variable((
                        len(der_model_set.flexible_der_models[der_name].outputs), 1))
                ))

                # location specifier
                optimization_problem.s1_index_locator[der_name, 'der_output_vector', stochastic_scenario,
                                                      str(time_index)] = [temp_size,
                                                                          optimization_problem.variable_vector_s1.size - 1]

    for stochastic_scenario in stochastic_scenarios:
        # Define DER power vector variables.
        optimization_problem.der_active_power_vector[stochastic_scenario] = (
            cp.Variable((len(der_model_set.timesteps), len(der_model_set.electric_ders)))
        )
        optimization_problem.der_reactive_power_vector[stochastic_scenario] = (
            cp.Variable((len(der_model_set.timesteps), len(der_model_set.electric_ders)))
        )

        for time_index in range(len(der_model_set.timesteps)):
            temp_size = optimization_problem.variable_vector_s1.size
            optimization_problem.variable_vector_s1 = cp.vstack((
                optimization_problem.variable_vector_s1, cp.Variable((len(der_model_set.electric_ders), 1))
            ))

            # location specifier
            optimization_problem.s1_index_locator['der_active_power_vector', stochastic_scenario,
                                                  str(time_index)] = [temp_size,
                                                                      optimization_problem.variable_vector_s1.size - 1]

        for time_index in range(len(der_model_set.timesteps)):
            temp_size = optimization_problem.variable_vector_s1.size
            optimization_problem.variable_vector_s1 = cp.vstack((
                optimization_problem.variable_vector_s1, cp.Variable((len(der_model_set.electric_ders), 1))
            ))

            # location specifier
            optimization_problem.s1_index_locator['der_reactive_power_vector', stochastic_scenario,
                                                  str(time_index)] = [temp_size,
                                                                      optimization_problem.variable_vector_s1.size - 1]
    # Define node voltage variable.
    optimization_problem.node_voltage_magnitude_vector = dict.fromkeys(stochastic_scenarios)

    for stochastic_scenario in stochastic_scenarios:
        optimization_problem.node_voltage_magnitude_vector[stochastic_scenario] = (
            cp.Variable((
                len(linear_electric_grid_model.electric_grid_model.timesteps),
                len(linear_electric_grid_model.electric_grid_model.nodes)
            ))
        )

        for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
            temp_size = optimization_problem.variable_vector_s1.size
            optimization_problem.variable_vector_s1 = cp.vstack((
                optimization_problem.variable_vector_s1,
                cp.Variable((len(linear_electric_grid_model.electric_grid_model.nodes), 1))
            ))

            # location specifier
            optimization_problem.s1_index_locator['node_voltage_magnitude_vector', stochastic_scenario,
                                                  str(time_index)] = [temp_size,
                                                                      optimization_problem.variable_vector_s1.size - 1]

    # Define voltage constraints for all scenarios
    for stochastic_scenario in stochastic_scenarios:
        optimization_problem.constraints.append(
            optimization_problem.node_voltage_magnitude_vector[stochastic_scenario]
            ==
            (
                    cp.transpose(
                        linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                        @ cp.transpose(cp.multiply(
                            optimization_problem.der_active_power_vector[stochastic_scenario],
                            np.array(
                                [np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                        ) - np.array(
                            [np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
                        + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                        @ cp.transpose(cp.multiply(
                            optimization_problem.der_reactive_power_vector[stochastic_scenario],
                            np.array(
                                [np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                        ) - np.array(
                            [np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
                    )
                    + np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
            )
            / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
        )

    node_voltage_magnitude_vector_minimum = (
            0.5 * np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
    )
    node_voltage_magnitude_vector_maximum = (
            1.5 * np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
    )

    # Define upper/lower bounds for voltage magnitude.
    for stochastic_scenario in stochastic_scenarios:
        optimization_problem.voltage_magnitude_vector_minimum_constraint = (
                optimization_problem.node_voltage_magnitude_vector[stochastic_scenario]
                - np.array([node_voltage_magnitude_vector_minimum.ravel()])
                / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
                >=
                0.0
        )
        optimization_problem.constraints.append(optimization_problem.voltage_magnitude_vector_minimum_constraint)
        optimization_problem.voltage_magnitude_vector_maximum_constraint = (
                optimization_problem.node_voltage_magnitude_vector[stochastic_scenario]
                - np.array([node_voltage_magnitude_vector_maximum.ravel()])
                / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
                <=
                0.0
        )
        optimization_problem.constraints.append(optimization_problem.voltage_magnitude_vector_maximum_constraint)

    # Define DER constraints.
    for stochastic_scenario in stochastic_scenarios:
        for der_name, der_model in der_model_set.der_models.items():

            # Fixed DERs.
            if issubclass(type(der_model), fledge.der_models.FixedDERModel):
                if der_model.is_electric_grid_connected:
                    optimization_problem.constraints.append(
                        optimization_problem.der_active_power_vector[stochastic_scenario][
                        :, der_model.electric_grid_der_index
                        ]
                        ==
                        np.transpose([der_model.active_power_nominal_timeseries.values])
                        / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)
                    )
                    optimization_problem.constraints.append(
                        optimization_problem.der_reactive_power_vector[stochastic_scenario][
                        :, der_model.electric_grid_der_index
                        ]
                        ==
                        np.transpose([der_model.reactive_power_nominal_timeseries.values])
                        / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)
                    )

            # Flexible DERs.
            elif issubclass(type(der_model), fledge.der_models.FlexibleDERModel):

                # Manipulate building model to avoid over-consumption for up-reserves.
                if issubclass(type(der_model), fledge.der_models.FlexibleBuildingModel):
                    der_model.output_maximum_timeseries.loc[
                    :, der_model.output_maximum_timeseries.columns.str.contains('_heat_')
                    ] = 0.0

                # Initial state.
                # - For states which represent storage state of charge, initial state of charge is final state of charge.
                if any(der_model.states.isin(der_model.storage_states)):
                    optimization_problem.constraints.append(
                        optimization_problem.state_vector[der_name, stochastic_scenario][
                            0, der_model.states.isin(der_model.storage_states)
                        ]
                        ==
                        optimization_problem.state_vector[der_name, stochastic_scenario][
                            -1, der_model.states.isin(der_model.storage_states)
                        ]
                    )
                # - For other states, set initial state according to the initial state vector.
                if any(~der_model.states.isin(der_model.storage_states)):
                    optimization_problem.constraints.append(
                        optimization_problem.state_vector[der_name, stochastic_scenario][
                            0, ~der_model.states.isin(der_model.storage_states)
                        ]
                        ==
                        der_model.state_vector_initial.loc[~der_model.states.isin(der_model.storage_states)].values
                    )

                # State equation.
                optimization_problem.constraints.append(
                    optimization_problem.state_vector[der_name, stochastic_scenario][1:, :]
                    ==
                    cp.transpose(
                        der_model.state_matrix.values
                        @ cp.transpose(optimization_problem.state_vector[der_name, stochastic_scenario][:-1, :])
                        + der_model.control_matrix.values
                        @ cp.transpose(optimization_problem.control_vector[der_name, stochastic_scenario][:-1, :])
                        + der_model.disturbance_matrix.values
                        @ np.transpose(der_model.disturbance_timeseries.iloc[:-1, :].values)
                    )
                )
                optimization_problem.constraints.append(
                    optimization_problem.control_vector[der_name, stochastic_scenario][-1, :]
                    ==
                    optimization_problem.control_vector[der_name, stochastic_scenario][-2, :]
                )

                # Output equation.
                optimization_problem.constraints.append(
                    optimization_problem.output_vector[der_name, stochastic_scenario]
                    ==
                    cp.transpose(
                        der_model.state_output_matrix.values
                        @ cp.transpose(optimization_problem.state_vector[der_name, stochastic_scenario])
                        + der_model.control_output_matrix.values
                        @ cp.transpose(optimization_problem.control_vector[der_name, stochastic_scenario])
                        + der_model.disturbance_output_matrix.values
                        @ np.transpose(der_model.disturbance_timeseries.values)
                    )
                )

                # Output limits.
                outputs_minimum_infinite = (
                    (der_model.output_minimum_timeseries == -np.inf).all()
                )
                optimization_problem.constraints.append(
                    optimization_problem.output_vector[der_name, stochastic_scenario][:, ~outputs_minimum_infinite]
                    >=
                    der_model.output_minimum_timeseries.loc[:, ~outputs_minimum_infinite].values
                )
                outputs_maximum_infinite = (
                    (der_model.output_maximum_timeseries == np.inf).all()
                )
                optimization_problem.constraints.append(
                    optimization_problem.output_vector[der_name, stochastic_scenario][:, ~outputs_maximum_infinite]
                    <=
                    der_model.output_maximum_timeseries.loc[:, ~outputs_maximum_infinite].values
                )

                # Define connection constraints.
                if der_model.is_electric_grid_connected:
                    optimization_problem.constraints.append(
                        optimization_problem.der_active_power_vector[stochastic_scenario][
                        :, der_model.electric_grid_der_index
                        ]
                        ==
                        cp.transpose(
                            der_model.mapping_active_power_by_output.values
                            @ cp.transpose(optimization_problem.output_vector[der_name, stochastic_scenario])
                        )
                        / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)
                    )
                    optimization_problem.constraints.append(
                        optimization_problem.der_reactive_power_vector[stochastic_scenario][
                        :, der_model.electric_grid_der_index
                        ]
                        ==
                        cp.transpose(
                            der_model.mapping_reactive_power_by_output.values
                            @ cp.transpose(optimization_problem.output_vector[der_name, stochastic_scenario])
                        )
                        / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)
                    )
    # Define Substation constr. TO DO

    # Define power balance constraints.
    for stochastic_scenario in stochastic_scenarios:
        if stochastic_scenario == 'no_reserve':
            optimization_problem.constraints.append(
                optimization_problem.no_reserve
                ==
                -1.0 * cp.sum(cp.multiply(
                    optimization_problem.der_active_power_vector[stochastic_scenario],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ), axis=1, keepdims=True)
            )
        elif stochastic_scenario == 'up_reserve':
            optimization_problem.constraints.append(
                optimization_problem.no_reserve
                + optimization_problem.up_reserve
                ==
                -1.0 * cp.sum(cp.multiply(
                    optimization_problem.der_active_power_vector[stochastic_scenario],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ), axis=1, keepdims=True)
            )
        else:
            optimization_problem.constraints.append(
                optimization_problem.no_reserve
                - optimization_problem.down_reserve
                ==
                -1.0 * cp.sum(cp.multiply(
                    optimization_problem.der_active_power_vector[stochastic_scenario],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ), axis=1, keepdims=True)
            )

    optimization_problem.constraints.append(
        optimization_problem.up_reserve >= 0
    )
    optimization_problem.constraints.append(
        optimization_problem.down_reserve >= 0
    )

    # Obtain timestep interval in hours, for conversion of power to energy.
    timestep_interval_hours = (
            (price_data.price_timeseries.index[1] - price_data.price_timeseries.index[0])
            / pd.Timedelta('1h')
    )

    # Define objective.
    optimization_problem.objective += (
            (
                    price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
                    * timestep_interval_hours  # In Wh.
                    @ optimization_problem.no_reserve
            )
            + (
                    -0.1
                    * price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
                    * timestep_interval_hours  # In Wh.
                    @ optimization_problem.up_reserve
            )
            + (
                    -1.1
                    * price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
                    * timestep_interval_hours  # In Wh.
                    @ optimization_problem.down_reserve
            )
    )
    # initialise A_1 definition
    A_1_column_number = optimization_problem.variable_vector_s1.size
    A_1 = np.zeros((len(linear_electric_grid_model.electric_grid_model.timesteps), A_1_column_number))
    b_1 = np.zeros((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))
    # standard form constr definition
    # To do - b_1 vector definition

    for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
        # constr 12 b)
        A_1[time_index, optimization_problem.s1_index_locator['energy', str(time_index)][0]] = 1
        A_1[time_index,
        optimization_problem.s1_index_locator['der_active_power_vector', 'no_reserve', str(time_index)][0]:
        optimization_problem.s1_index_locator['der_active_power_vector', 'no_reserve', str(time_index)][
            1] + 1] = np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])

    # constr 12 c)
    A_1_temp = np.zeros((len(linear_electric_grid_model.electric_grid_model.timesteps), A_1_column_number))
    for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
        A_1_temp[time_index, optimization_problem.s1_index_locator['energy', str(time_index)][0]] = 1
        A_1_temp[time_index, optimization_problem.s1_index_locator['up_reserve', str(time_index)][0]] = 1
        A_1_temp[time_index,
        optimization_problem.s1_index_locator['der_active_power_vector', 'up_reserve', str(time_index)][0]:
        optimization_problem.s1_index_locator['der_active_power_vector', 'up_reserve', str(time_index)][
            1] + 1] = np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])

    b_1 = np.vstack((b_1, np.zeros((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))))
    A_1 = np.vstack((A_1, A_1_temp))

    # constr 12 d)
    A_1_temp = np.zeros((len(linear_electric_grid_model.electric_grid_model.timesteps), A_1_column_number))
    for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
        A_1_temp[time_index, optimization_problem.s1_index_locator['energy', str(time_index)][0]] = 1
        A_1_temp[time_index, optimization_problem.s1_index_locator['down_reserve', str(time_index)][0]] = -1
        A_1_temp[time_index,
        optimization_problem.s1_index_locator['der_active_power_vector', 'down_reserve', str(time_index)][0]:
        optimization_problem.s1_index_locator['der_active_power_vector', 'down_reserve', str(time_index)][
            1] + 1] = np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])

    b_1 = np.vstack((b_1, np.zeros((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))))
    A_1 = np.vstack((A_1, A_1_temp))

    # constr 12 g)
    # V^ref * Vm - M^vp * p^ref * p^der - M^vq * q^ref * q^der = -(M^vp*p^* + M^vq*q^*)

    for stochastic_scenario in stochastic_scenarios:
        for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
            A_1_temp = np.zeros((len(linear_electric_grid_model.electric_grid_model.nodes), A_1_column_number))
            # voltage magnitude coefficient
            A_1_temp[:,
            optimization_problem.s1_index_locator[
                'node_voltage_magnitude_vector', stochastic_scenario, str(time_index)][0]:
            optimization_problem.s1_index_locator[
                'node_voltage_magnitude_vector', stochastic_scenario, str(time_index)][
                1] + 1] = np.diagflat(
                np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)]))
            # active power contribution:
            A_1_temp[:,
            optimization_problem.s1_index_locator['der_active_power_vector', stochastic_scenario, str(time_index)][0]:
            optimization_problem.s1_index_locator['der_active_power_vector', stochastic_scenario, str(time_index)][
                1] + 1] = linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active @ np.diagflat(
                np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))
            # reactive power contribution:
            A_1_temp[:,
            optimization_problem.s1_index_locator['der_reactive_power_vector', stochastic_scenario, str(time_index)][0]:
            optimization_problem.s1_index_locator['der_reactive_power_vector', stochastic_scenario, str(time_index)][
                1] + 1] = linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive @ np.diagflat(
                np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))

            active_reference_point_temp = np.array(
                [np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            reactive_reference_point_temp = np.array(
                [np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            b_1_temp = -(linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                         @ active_reference_point_temp.T +
                         linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive @ reactive_reference_point_temp.T)
            b_1 = np.vstack((b_1, b_1_temp))
            A_1 = np.vstack((A_1, A_1_temp))

    # constr 12 i) - 12k) To do
    # Define DER constraints.

    for der_name, der_model in der_model_set.der_models.items():
        # Fixed DERs.
        if issubclass(type(der_model), fledge.der_models.FixedDERModel):
            if der_model.is_electric_grid_connected:
                for stochastic_scenario in stochastic_scenarios:
                    for time_index in range(len(der_model_set.timesteps)):
                        # for active power (fixed ders)
                        A_1_temp = np.zeros((1, A_1_column_number))

                        A_1_temp[:, optimization_problem.s1_index_locator[
                                        'der_active_power_vector', stochastic_scenario, str(time_index)][0
                                    ] + der_model.electric_grid_der_index[0]] = 1

                        b_1_temp = [der_model.active_power_nominal_timeseries.values[time_index] / (
                            der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)]

                        b_1 = np.vstack((b_1, b_1_temp))
                        A_1 = np.vstack((A_1, A_1_temp))

                        # for reactive power (fixed ders)
                        A_1_temp = np.zeros((1, A_1_column_number))

                        A_1_temp[:, optimization_problem.s1_index_locator[
                                        'der_reactive_power_vector', stochastic_scenario, str(time_index)][
                                        0] + der_model.electric_grid_der_index[0]] = 1

                        b_1_temp = [der_model.reactive_power_nominal_timeseries.values[time_index] / (
                            der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)]

                        b_1 = np.vstack((b_1, b_1_temp))
                        A_1 = np.vstack((A_1, A_1_temp))

    # Flexible DERs.
    for der_name, der_model in der_model_set.der_models.items():
        if issubclass(type(der_model), fledge.der_models.FlexibleDERModel):

            # # Manipulate building model to avoid over-consumption for up-reserves.
            # if issubclass(type(der_model), fledge.der_models.FlexibleBuildingModel):
            #     der_model.output_maximum_timeseries.loc[
            #     :, der_model.output_maximum_timeseries.columns.str.contains('_heat_')
            #     ] = 0.0
            # Initial state.
            # - For states which represent storage state of charge, initial state of charge is final state of charge.
            for stochastic_scenario in stochastic_scenarios:
                for time_index in range(len(der_model_set.timesteps)):
                    print(time_index)

                    # Initial state
                    if time_index == 0:
                        if any(der_model.states.isin(der_model.storage_states)):
                            A_1_temp = np.zeros((1, A_1_column_number))
                            A_1_temp[:,
                            optimization_problem.s1_index_locator[der_name, 'der_state_vector', stochastic_scenario,
                                                                  str(time_index)][0] + der_model.states.isin(
                                der_model.storage_states)] = 1
                            A_1_temp[:,
                            optimization_problem.s1_index_locator[der_name, 'der_state_vector', stochastic_scenario,
                                                                  str(len(der_model_set.timesteps)-1)][0] + der_model.states.isin(
                                der_model.storage_states)] = -1

                            b_1 = np.vstack((b_1, [0]))
                            A_1 = np.vstack((A_1, A_1_temp))


                    #     optimization_problem.constraints.append(
                    #         optimization_problem.state_vector[der_name, stochastic_scenario][
                    #             0, der_model.states.isin(der_model.storage_states)
                    #         ]
                    #         ==
                    #         optimization_problem.state_vector[der_name, stochastic_scenario][
                    #             -1, der_model.states.isin(der_model.storage_states)
                    #         ]
                    #     )
                    # # - For other states, set initial state according to the initial state vector.
                    # if any(~der_model.states.isin(der_model.storage_states)):
                    #     optimization_problem.constraints.append(
                    #         optimization_problem.state_vector[der_name, stochastic_scenario][
                    #             0, ~der_model.states.isin(der_model.storage_states)
                    #         ]
                    #         ==
                    #         der_model.state_vector_initial.loc[~der_model.states.isin(der_model.storage_states)].values
                    #     )
                    #
                    # # State equation.
                    # optimization_problem.constraints.append(
                    #     optimization_problem.state_vector[der_name, stochastic_scenario][1:, :]
                    #     ==
                    #     cp.transpose(
                    #         der_model.state_matrix.values
                    #         @ cp.transpose(optimization_problem.state_vector[der_name, stochastic_scenario][:-1, :])
                    #         + der_model.control_matrix.values
                    #         @ cp.transpose(optimization_problem.control_vector[der_name, stochastic_scenario][:-1, :])
                    #         + der_model.disturbance_matrix.values
                    #         @ np.transpose(der_model.disturbance_timeseries.iloc[:-1, :].values)
                    #     )
                    # )
                    # optimization_problem.constraints.append(
                    #     optimization_problem.control_vector[der_name, stochastic_scenario][-1, :]
                    #     ==
                    #     optimization_problem.control_vector[der_name, stochastic_scenario][-2, :]
                    # )
                    #
                    # # Output equation.
                    # optimization_problem.constraints.append(
                    #     optimization_problem.output_vector[der_name, stochastic_scenario]
                    #     ==
                    #     cp.transpose(
                    #         der_model.state_output_matrix.values
                    #         @ cp.transpose(optimization_problem.state_vector[der_name, stochastic_scenario])
                    #         + der_model.control_output_matrix.values
                    #         @ cp.transpose(optimization_problem.control_vector[der_name, stochastic_scenario])
                    #         + der_model.disturbance_output_matrix.values
                    #         @ np.transpose(der_model.disturbance_timeseries.values)
                    #     )
                    # )
                    #
                    # # Output limits.
                    # outputs_minimum_infinite = (
                    #     (der_model.output_minimum_timeseries == -np.inf).all()
                    # )
                    # optimization_problem.constraints.append(
                    #     optimization_problem.output_vector[der_name, stochastic_scenario][:, ~outputs_minimum_infinite]
                    #     >=
                    #     der_model.output_minimum_timeseries.loc[:, ~outputs_minimum_infinite].values
                    # )
                    # outputs_maximum_infinite = (
                    #     (der_model.output_maximum_timeseries == np.inf).all()
                    # )
                    # optimization_problem.constraints.append(
                    #     optimization_problem.output_vector[der_name, stochastic_scenario][:, ~outputs_maximum_infinite]
                    #     <=
                    #     der_model.output_maximum_timeseries.loc[:, ~outputs_maximum_infinite].values
                    # )
                    #
                    # # Define connection constraints.
                    # if der_model.is_electric_grid_connected:
                    #     optimization_problem.constraints.append(
                    #         optimization_problem.der_active_power_vector[stochastic_scenario][
                    #         :, der_model.electric_grid_der_index
                    #         ]
                    #         ==
                    #         cp.transpose(
                    #             der_model.mapping_active_power_by_output.values
                    #             @ cp.transpose(optimization_problem.output_vector[der_name, stochastic_scenario])
                    #         )
                    #         / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)
                    #     )
                    #     optimization_problem.constraints.append(
                    #         optimization_problem.der_reactive_power_vector[stochastic_scenario][
                    #         :, der_model.electric_grid_der_index
                    #         ]
                    #         ==
                    #         cp.transpose(
                    #             der_model.mapping_reactive_power_by_output.values
                    #             @ cp.transpose(optimization_problem.output_vector[der_name, stochastic_scenario])
                    #         )
                    #         / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)
                    #     )

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain reserve results.
    no_reserve = pd.Series(optimization_problem.no_reserve.value.ravel(), index=der_model_set.timesteps)
    up_reserve = pd.Series(optimization_problem.up_reserve.value.ravel(), index=der_model_set.timesteps)
    down_reserve = pd.Series(optimization_problem.down_reserve.value.ravel(), index=der_model_set.timesteps)

    # Instantiate DER results variables.
    state_vector = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.states)
        for stochastic_scenario in stochastic_scenarios
    }
    control_vector = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.controls)
        for stochastic_scenario in stochastic_scenarios
    }
    output_vector = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
        for stochastic_scenario in stochastic_scenarios
    }
    der_active_power_vector = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.electric_ders)
        for stochastic_scenario in stochastic_scenarios
    }
    der_active_power_vector_per_unit = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.electric_ders)
        for stochastic_scenario in stochastic_scenarios
    }
    der_reactive_power_vector = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.electric_ders)
        for stochastic_scenario in stochastic_scenarios
    }
    der_reactive_power_vector_per_unit = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.electric_ders)
        for stochastic_scenario in stochastic_scenarios
    }

    # Obtain DER results.
    for stochastic_scenario in stochastic_scenarios:
        for der_name in der_model_set.flexible_der_names:
            state_vector[stochastic_scenario].loc[:, (der_name, slice(None))] = (
                optimization_problem.state_vector[der_name, stochastic_scenario].value
            )
            control_vector[stochastic_scenario].loc[:, (der_name, slice(None))] = (
                optimization_problem.control_vector[der_name, stochastic_scenario].value
            )
            output_vector[stochastic_scenario].loc[:, (der_name, slice(None))] = (
                optimization_problem.output_vector[der_name, stochastic_scenario].value
            )
        for der_name in der_model_set.der_names:
            if der_model_set.der_models[der_name].is_electric_grid_connected:
                der_active_power_vector_per_unit[stochastic_scenario].loc[:, (slice(None), der_name)] = (
                    optimization_problem.der_active_power_vector[stochastic_scenario][
                    :, fledge.utils.get_index(der_model_set.electric_ders, der_name=der_name)
                    ].value
                )
                der_active_power_vector[stochastic_scenario].loc[:, (slice(None), der_name)] = (
                        der_active_power_vector_per_unit[stochastic_scenario].loc[:, (slice(None), der_name)].values
                        * der_model_set.der_models[der_name].active_power_nominal
                )
                der_reactive_power_vector_per_unit[stochastic_scenario].loc[:, (slice(None), der_name)] = (
                    optimization_problem.der_reactive_power_vector[stochastic_scenario][
                    :, fledge.utils.get_index(der_model_set.electric_ders, der_name=der_name)
                    ].value
                )
                der_reactive_power_vector[stochastic_scenario].loc[:, (slice(None), der_name)] = (
                        der_reactive_power_vector_per_unit[stochastic_scenario].loc[:, (slice(None), der_name)].values
                        * der_model_set.der_models[der_name].reactive_power_nominal
                )

    # Plot some results.
    figure = go.Figure()
    figure.add_scatter(
        x=no_reserve.index,
        y=no_reserve.values,
        name='no_reserve',
        line=go.scatter.Line(shape='hv', width=5, dash='dot')
    )
    figure.add_scatter(
        x=up_reserve.index,
        y=up_reserve.values,
        name='up_reserve',
        line=go.scatter.Line(shape='hv', width=4, dash='dot')
    )
    figure.add_scatter(
        x=down_reserve.index,
        y=down_reserve.values,
        name='down_reserve',
        line=go.scatter.Line(shape='hv', width=3, dash='dot')
    )
    figure.add_scatter(
        x=up_reserve.index,
        y=(no_reserve + up_reserve).values,
        name='no_reserve + up_reserve',
        line=go.scatter.Line(shape='hv', width=2, dash='dot')
    )
    figure.add_scatter(
        x=up_reserve.index,
        y=(no_reserve - down_reserve).values,
        name='no_reserve - down_reserve',
        line=go.scatter.Line(shape='hv', width=1, dash='dot')
    )
    figure.update_layout(
        title=f'Power balance',
        xaxis=go.layout.XAxis(tickformat='%H:%M'),
        legend=go.layout.Legend(x=0.01, xanchor='auto', y=0.99, yanchor='auto')
    )
    # figure.show()
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'0_power_balance'))

    for der_name, der_model in der_model_set.flexible_der_models.items():

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
            for number, stochastic_scenario in enumerate(stochastic_scenarios):
                figure.add_scatter(
                    x=output_vector[stochastic_scenario].index,
                    y=output_vector[stochastic_scenario].loc[:, (der_name, output)].values,
                    name=f'Optimal: {stochastic_scenario}',
                    line=go.scatter.Line(shape='hv', width=number + 3, dash='dot')
                )
            figure.update_layout(
                title=f'DER: ({der_model.der_type}, {der_name}) / Output: {output}',
                xaxis=go.layout.XAxis(tickformat='%H:%M'),
                legend=go.layout.Legend(x=0.01, xanchor='auto', y=0.99, yanchor='auto')
            )
            # figure.show()
            fledge.utils.write_figure_plotly(figure, os.path.join(
                results_path, f'der_{der_model.der_type}_{der_name}_output_{output}'
            ))

        # for control in der_model.controls:
        #     figure = go.Figure()
        #     for number, stochastic_scenario in enumerate(stochastic_scenarios):
        #         figure.add_scatter(
        #             x=output_vector[stochastic_scenario].index,
        #             y=output_vector[stochastic_scenario].loc[:, (der_name, control)].values,
        #             name=f'Optimal: {stochastic_scenario}',
        #             line=go.scatter.Line(shape='hv', width=number+3, dash='dot')
        #         )
        #     figure.update_layout(
        #         title=f'DER: ({der_model.der_type}, {der_name}) / Control: {control}',
        #         xaxis=go.layout.XAxis(tickformat='%H:%M'),
        #         legend=go.layout.Legend(x=0.01, xanchor='auto', y=0.99, yanchor='auto')
        #     )
        #     # figure.show()
        #     fledge.utils.write_figure_plotly(figure, os.path.join(
        #         results_path, f'der_{der_model.der_type}_{der_name}_control_{control}'
        #     ))

        # for disturbance in der_model.disturbances:
        #     figure = go.Figure()
        #     figure.add_scatter(
        #         x=der_model.disturbance_timeseries.index,
        #         y=der_model.disturbance_timeseries.loc[:, disturbance].values,
        #         line=go.scatter.Line(shape='hv')
        #     )
        #     figure.update_layout(
        #         title=f'DER: ({der_model.der_type}, {der_name}) / Disturbance: {disturbance}',
        #         xaxis=go.layout.XAxis(tickformat='%H:%M'),
        #         showlegend=False
        #     )
        #     # figure.show()
        #     fledge.utils.write_figure_plotly(figure, os.path.join(
        #         results_path, f'der_{der_model.der_type}_{der_name}_disturbance_{disturbance}'
        #     ))

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
