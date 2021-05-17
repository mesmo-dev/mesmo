"""Example script for DRO problem."""

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge



def main():
    # Settings.
    scenario_name = 'singapore_6node'
    stochastic_scenarios = ['no_reserve', 'up_reserve', 'down_reserve']

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


    # Obtain standard form.
    standard_form = fledge.utils.StandardForm()

    # Define variables.
    standard_form.define_variable('energy', timestep=linear_electric_grid_model.electric_grid_model.timesteps)
    standard_form.define_variable('up_reserve', timestep=linear_electric_grid_model.electric_grid_model.timesteps)
    standard_form.define_variable('down_reserve', timestep=linear_electric_grid_model.electric_grid_model.timesteps)

    print()

    for stochastic_scenario in stochastic_scenarios:
        for der_name, der_model in der_model_set.flexible_der_models.items():
            standard_form.define_variable(
                'state_vector', timestep=der_model.timesteps, der_name=[der_model.der_name], state=der_model.states,
                scenario=[stochastic_scenario]
            )
            standard_form.define_variable(
                'control_vector', timestep=der_model.timesteps, der_name=[der_model.der_name], control=der_model.controls,
                scenario=[stochastic_scenario])
            standard_form.define_variable(
                'output_vector', timestep=der_model.timesteps, der_name=[der_model.der_name], output=der_model.outputs,
                scenario=[stochastic_scenario]
            )


    for stochastic_scenario in stochastic_scenarios:
        standard_form.define_variable(
            'der_active_power_vector', timestep=der_model_set.timesteps, der_model=der_model_set.ders,
            scenario=[stochastic_scenario]
        )

        standard_form.define_variable(
            'der_reactive_power_vector', timestep=der_model_set.timesteps, der_model=der_model_set.ders,
            scenario=[stochastic_scenario]
        )

    for stochastic_scenario in stochastic_scenarios:
        standard_form.define_variable(
            'nodal_voltage_magnitude', timestep=der_model_set.timesteps,
            grid_node=linear_electric_grid_model.electric_grid_model.nodes.to_numpy(),
            scenario=[stochastic_scenario]
        )

    # power balance equation.
    for timestep_grid, timestep_der in zip(linear_electric_grid_model.electric_grid_model.timesteps,
                                           der_model_set.timesteps):
        standard_form.define_constraint(
            ('variable', 1.0,
             dict(name='energy', timestep=timestep_grid)),
            '==',
            ('variable', -np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]),
             dict(name='der_active_power_vector', timestep=timestep_der,
                  der_model=der_model_set.ders, scenario='no_reserve')),
        )

    # power balance equation. up reserve
    for timestep_grid, timestep_der in zip(linear_electric_grid_model.electric_grid_model.timesteps,
                                           der_model_set.timesteps):
        standard_form.define_constraint(
            ('variable', 1.0,
             dict(name='energy', timestep=timestep_grid)),
            ('variable', 1.0,
             dict(name='up_reserve', timestep=timestep_grid)),
            '==',
            ('variable',
             -np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]),
             dict(name='der_active_power_vector', timestep=timestep_der,
                  der_model=der_model_set.ders, scenario='up_reserve')),
        )

    # power balance equation. down reserve
    for timestep_grid, timestep_der in zip(linear_electric_grid_model.electric_grid_model.timesteps,
                                           der_model_set.timesteps):
        standard_form.define_constraint(
            ('variable', 1.0,
             dict(name='energy', timestep=timestep_grid)),
            ('variable', -1.0,
             dict(name='down_reserve', timestep=timestep_grid)),
            '==',
            ('variable',
             -np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]),
             dict(name='der_active_power_vector', timestep=timestep_der,
                  der_model=der_model_set.ders, scenario='down_reserve')),
        )

    print()

    # constr 12 g)
    # V^ref * Vm - M^vp * p^ref * p^der - M^vq * q^ref * q^der = -(M^vp*p^* + M^vq*q^*)
    active_reference_point_temp = np.array(
        [np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
    reactive_reference_point_temp = np.array(
        [np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])


    for stochastic_scenario in stochastic_scenarios:
        for timestep_grid, timestep_der in zip(linear_electric_grid_model.electric_grid_model.timesteps,
                                               der_model.timesteps):
            standard_form.define_constraint(
                ('variable', np.diagflat(
                    np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])),
                    dict(name='nodal_voltage_magnitude', timestep=timestep_grid,
                         grid_node=linear_electric_grid_model.electric_grid_model.nodes.to_numpy(),
                         scenario=[stochastic_scenario])),
                ('variable',
                 -linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active @ np.diagflat(
                     np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])),
                 dict(name='der_active_power_vector', timestep=timestep_der,
                      der_model=der_model_set.ders, scenario=[stochastic_scenario])),
                ('variable',
                 -linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive @ np.diagflat(
                     np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])),
                 dict(name='der_reactive_power_vector', timestep=timestep_der,
                      der_model=der_model_set.ders, scenario=[stochastic_scenario])),
                '==',
                ('constant', -(linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                               @ active_reference_point_temp.T +
                               linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                               @ reactive_reference_point_temp.T))
            )

    print()
    for der_name, der_model in der_model_set.der_models.items():
        # Fixed DERs.
        if issubclass(type(der_model), fledge.der_models.FixedDERModel):
            if der_model.is_electric_grid_connected:
                for stochastic_scenario in stochastic_scenarios:
                    for timestep_der in der_model_set.timesteps:
                        standard_form.define_constraint(
                            ('variable', 1.0, dict(name='der_active_power_vector', timestep=timestep_der,
                                                   der_model=(der_model.der_type, der_model.der_name),
                                                   scenario=[stochastic_scenario])),
                            '==',
                            ('constant', (der_model.active_power_nominal_timeseries[timestep_der] / (
                            der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)).tolist())
                        )

                        standard_form.define_constraint(
                            ('variable', 1.0, dict(name='der_reactive_power_vector', timestep=timestep_der,
                                                   der_model=(der_model.der_type, der_model.der_name),
                                                   scenario=[stochastic_scenario])),
                            '==',
                            ('constant', (der_model.reactive_power_nominal_timeseries[timestep_der] / (
                            der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)).tolist())
                        )

    # Flexible DERs.
    for der_name, der_model in der_model_set.der_models.items():
        if issubclass(type(der_model), fledge.der_models.FlexibleDERModel):

            # Manipulate building model to avoid over-consumption for up-reserves.
            if issubclass(type(der_model), fledge.der_models.FlexibleBuildingModel):
                der_model.output_maximum_timeseries.loc[
                :, der_model.output_maximum_timeseries.columns.str.contains('_heat_')
                ] = 0.0

            # Initial state.
            # - For states which represent storage state of charge, initial state of charge is final state of charge.
            for stochastic_scenario in stochastic_scenarios:
                #for timestep_der in der_model_set.timesteps:

                # # Initial state.
                # # - TODO: For states which represent storage state of charge, initial state of charge is final state of charge.
                if any(~der_model.states.isin(der_model.storage_states)):
                    standard_form.define_constraint(
                        ('constant', der_model.state_vector_initial.values[~der_model.states.isin(der_model.storage_states)]),
                        '==',
                        ('variable', 1.0, dict(
                            name='state_vector', der_name = [der_model.der_name], timestep=der_model.timesteps[0],
                            state=der_model.states[~der_model.states.isin(der_model.storage_states)],
                            scenario=[stochastic_scenario],
                        ))
                    )

                if any(der_model.states.isin(der_model.storage_states)):
                    standard_form.define_constraint(
                        ('variable', 1.0, dict(
                            name='state_vector', der_name=[der_model.der_name], timestep=der_model.timesteps[0],
                            state=der_model.states[der_model.states.isin(der_model.storage_states)],
                            scenario=[stochastic_scenario]
                        )),
                        '==',
                        ('variable', 1.0, dict(
                            name='state_vector', der_name=[der_model.der_name], timestep=der_model.timesteps[-1],
                            state=der_model.states[der_model.states.isin(der_model.storage_states)],
                            scenario=[stochastic_scenario]
                        ))
                    )


    # # State equation.
    # for timestep, timestep_previous in zip(der_model.timesteps[1:], der_model.timesteps[:-1]):
    #     standard_form.define_constraint(
    #         ('variable', 1.0, dict(name='state_vector', timestep=timestep)),
    #         '==',
    #         ('variable', der_model.state_matrix.values, dict(name='state_vector', timestep=timestep_previous)),
    #         ('variable', der_model.control_matrix.values, dict(name='control_vector', timestep=timestep)),
    #         ('constant', der_model.disturbance_matrix.values @ der_model.disturbance_timeseries.loc[timestep, :].values)
    #     )
    #
    # # Output equation.
    # for timestep in der_model.timesteps:
    #     standard_form.define_constraint(
    #         ('variable', 1.0, dict(name='output_vector', timestep=timestep)),
    #         '==',
    #         ('variable', der_model.state_output_matrix.values, dict(name='state_vector', timestep=timestep)),
    #         ('variable', der_model.control_output_matrix.values, dict(name='control_vector', timestep=timestep)),
    #         ('constant',
    #          der_model.disturbance_output_matrix.values @ der_model.disturbance_timeseries.loc[timestep, :].values)
    #     )
    #
    # # Output limits.
    # for timestep in der_model.timesteps:
    #     standard_form.define_constraint(
    #         ('variable', 1.0, dict(name='output_vector', timestep=timestep)),
    #         '>=',
    #         ('constant', der_model.output_minimum_timeseries.loc[timestep, :].values),
    #     )
    #     standard_form.define_constraint(
    #         ('variable', 1.0, dict(name='output_vector', timestep=timestep)),
    #         '<=',
    #         ('constant', der_model.output_maximum_timeseries.loc[timestep, :].values),
    #     )



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

    for stochastic_scenario in stochastic_scenarios:
        for der_name in der_model_set.flexible_der_names:

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
        # # Define DER power vector variables.
        # optimization_problem.der_active_power_vector[stochastic_scenario] = (
        #     cp.Variable((len(der_model_set.timesteps), len(der_model_set.electric_ders)))
        # )
        # optimization_problem.der_reactive_power_vector[stochastic_scenario] = (
        #     cp.Variable((len(der_model_set.timesteps), len(der_model_set.electric_ders)))
        # )

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

    for stochastic_scenario in stochastic_scenarios:

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



    # Obtain timestep interval in hours, for conversion of power to energy.
    timestep_interval_hours = (
            (price_data.price_timeseries.index[1] - price_data.price_timeseries.index[0])
            / pd.Timedelta('1h')
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

    # constr 12 i) - 12k)
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
                            storage_index_temp = np.argwhere(der_model.states.isin(der_model.storage_states) == True)

                            A_1_temp[:,
                                    optimization_problem.s1_index_locator[
                                    der_name, 'der_state_vector', stochastic_scenario,
                                    str(time_index)
                                    ][0]
                                    + storage_index_temp
                            ] = 1
                            A_1_temp[:,
                                    optimization_problem.s1_index_locator[
                                    der_name, 'der_state_vector', stochastic_scenario,
                                    str(len(der_model_set.timesteps)-1)
                                    ][0]
                                    + storage_index_temp
                            ] = -1

                            b_1 = np.vstack((b_1, [0]))
                            A_1 = np.vstack((A_1, A_1_temp))
                        elif any(~der_model.states.isin(der_model.storage_states)):
                            not_a_storage_index_vector_temp = np.argwhere(
                                ~der_model.states.isin(der_model.storage_states) == True
                            )

                            A_1_temp = np.zeros((not_a_storage_index_vector_temp.size, A_1_column_number))

                            A_1_temp[:,
                            optimization_problem.s1_index_locator[
                                der_name, 'der_state_vector', stochastic_scenario,
                                str(time_index)
                            ][0]
                            + not_a_storage_index_vector_temp
                            ] = 1

                            # here it should be a col vector
                            b_1_temp = der_model.state_vector_initial.loc[
                                ~der_model.states.isin(der_model.storage_states)
                            ].values

                            b_1 = np.vstack((b_1, b_1_temp))
                            A_1 = np.vstack((A_1, A_1_temp))

                    # State equation.
                    if time_index != 0:
                        A_1_temp = np.zeros((der_model.state_matrix.shape[0], A_1_column_number))

                        A_1_temp[:,
                        optimization_problem.s1_index_locator[
                            der_name, 'der_state_vector', stochastic_scenario,
                            str(time_index)
                        ][0]:
                        optimization_problem.s1_index_locator[
                            der_name, 'der_state_vector', stochastic_scenario,
                            str(time_index)
                        ][1] + 1
                        ] = np.eye(der_model.state_matrix.shape[0])

                        A_1_temp[:,
                        optimization_problem.s1_index_locator[
                            der_name, 'der_state_vector', stochastic_scenario,
                            str(time_index-1)
                        ][0]:
                        optimization_problem.s1_index_locator[
                            der_name, 'der_state_vector', stochastic_scenario,
                            str(time_index-1)
                        ][1]+1
                        ] = - der_model.state_matrix.values

                        A_1_temp[:,
                        optimization_problem.s1_index_locator[
                            der_name, 'der_control_vector', stochastic_scenario,
                            str(time_index)
                        ][0]:
                        optimization_problem.s1_index_locator[
                            der_name, 'der_control_vector', stochastic_scenario,
                            str(time_index)
                        ][1]+1
                        ] = - der_model.control_matrix.values

                        b_1_temp = der_model.disturbance_matrix.values @ np.transpose(
                            der_model.disturbance_timeseries.iloc[time_index, :].values
                        )

                        b_1 = np.vstack((b_1, b_1_temp))
                        A_1 = np.vstack((A_1, A_1_temp))

                    # Output equation.
                    A_1_temp = np.zeros((der_model.state_output_matrix.shape[0], A_1_column_number))

                    A_1_temp[:,
                    optimization_problem.s1_index_locator[
                        der_name, 'der_output_vector', stochastic_scenario,
                        str(time_index)
                    ][0]:
                    optimization_problem.s1_index_locator[
                        der_name, 'der_output_vector', stochastic_scenario,
                        str(time_index)
                    ][1] + 1
                    ] = np.eye(der_model.state_output_matrix.shape[0])

                    A_1_temp[:,
                    optimization_problem.s1_index_locator[
                        der_name, 'der_state_vector', stochastic_scenario,
                        str(time_index)
                    ][0]:
                    optimization_problem.s1_index_locator[
                        der_name, 'der_state_vector', stochastic_scenario,
                        str(time_index)
                    ][1] + 1
                    ] = - der_model.state_output_matrix.values

                    A_1_temp[:,
                    optimization_problem.s1_index_locator[
                        der_name, 'der_control_vector', stochastic_scenario,
                        str(time_index)
                    ][0]:
                    optimization_problem.s1_index_locator[
                        der_name, 'der_control_vector', stochastic_scenario,
                        str(time_index)
                    ][1] + 1
                    ] = - der_model.control_output_matrix.values

                    b_1_temp = np.transpose(der_model.disturbance_output_matrix.values @ np.transpose(
                        der_model.disturbance_timeseries.iloc[time_index, :].values
                    )).reshape(der_model.state_output_matrix.shape[0],1)

                    b_1 = np.vstack((b_1, b_1_temp))
                    A_1 = np.vstack((A_1, A_1_temp))

                    # Define connection constraints.
                    if der_model.is_electric_grid_connected:
                        # for active power (flexible DERs)
                        A_1_temp = np.zeros((1, A_1_column_number))

                        A_1_temp[:, optimization_problem.s1_index_locator[
                                        'der_active_power_vector', stochastic_scenario, str(time_index)][0
                                    ] + der_model.electric_grid_der_index[0]] = 1

                        A_1_temp[:,
                        optimization_problem.s1_index_locator[
                            der_name, 'der_output_vector', stochastic_scenario,
                            str(time_index)
                        ][0]:
                        optimization_problem.s1_index_locator[
                            der_name, 'der_output_vector', stochastic_scenario,
                            str(time_index)
                        ][1] + 1
                        ] = - der_model.mapping_active_power_by_output.values\
                            / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)

                        b_1 = np.vstack((b_1, [0]))
                        A_1 = np.vstack((A_1, A_1_temp))

                        # for reactive power (flexible DERs)
                        A_1_temp = np.zeros((1, A_1_column_number))

                        A_1_temp[:, optimization_problem.s1_index_locator[
                                        'der_reactive_power_vector', stochastic_scenario, str(time_index)][0
                                    ] + der_model.electric_grid_der_index[0]] = 1

                        A_1_temp[:,
                        optimization_problem.s1_index_locator[
                            der_name, 'der_output_vector', stochastic_scenario,
                            str(time_index)
                        ][0]:
                        optimization_problem.s1_index_locator[
                            der_name, 'der_output_vector', stochastic_scenario,
                            str(time_index)
                        ][1] + 1
                        ] = - der_model.mapping_reactive_power_by_output.values \
                            / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)

                        b_1 = np.vstack((b_1, [0]))
                        A_1 = np.vstack((A_1, A_1_temp))

    # equality constraints
    A_1 = np.vstack((-A_1, A_1))
    b_1 = np.vstack((-b_1, b_1))

    # inequality constraints
    # Flexible DERs. # Output limits.
    for der_name, der_model in der_model_set.der_models.items():
        if issubclass(type(der_model), fledge.der_models.FlexibleDERModel):
            for stochastic_scenario in stochastic_scenarios:
                for time_index in range(len(der_model_set.timesteps)):
                    # upper bound
                    A_1_temp = np.zeros((der_model.state_output_matrix.shape[0], A_1_column_number))

                    A_1_temp[:,
                    optimization_problem.s1_index_locator[
                        der_name, 'der_output_vector', stochastic_scenario,
                        str(time_index)
                    ][0]:
                    optimization_problem.s1_index_locator[
                        der_name, 'der_output_vector', stochastic_scenario,
                        str(time_index)
                    ][1] + 1
                    ] = np.eye(der_model.state_output_matrix.shape[0])

                    b_1_temp = der_model.output_maximum_timeseries.loc[
                               der_model.timesteps[time_index], :
                               ].values.reshape(der_model.state_output_matrix.shape[0], 1)

                    b_1 = np.vstack((b_1, b_1_temp))
                    A_1 = np.vstack((A_1, A_1_temp))

                    A_1_temp = np.zeros((der_model.state_output_matrix.shape[0], A_1_column_number))

                    A_1_temp[:,
                    optimization_problem.s1_index_locator[
                        der_name, 'der_output_vector', stochastic_scenario,
                        str(time_index)
                    ][0]:
                    optimization_problem.s1_index_locator[
                        der_name, 'der_output_vector', stochastic_scenario,
                        str(time_index)
                    ][1] + 1
                    ] = -np.eye(der_model.state_output_matrix.shape[0])

                    b_1_temp = -der_model.output_minimum_timeseries.loc[
                               der_model.timesteps[time_index], :
                               ].values.reshape(der_model.state_output_matrix.shape[0], 1)

                    b_1 = np.vstack((b_1, b_1_temp))
                    A_1 = np.vstack((A_1, A_1_temp))

    node_voltage_magnitude_vector_minimum = (
            0.5 * np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
    )
    node_voltage_magnitude_vector_maximum = (
            1.5 * np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
    )

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

            b_1_temp = np.transpose(np.array([node_voltage_magnitude_vector_maximum.ravel()])/np.array(
                [np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)]
            ))

            b_1 = np.vstack((b_1, b_1_temp))
            A_1 = np.vstack((A_1, A_1_temp))

            A_1_temp = np.zeros((len(linear_electric_grid_model.electric_grid_model.nodes), A_1_column_number))
            # voltage magnitude coefficient
            A_1_temp[:,
            optimization_problem.s1_index_locator[
                'node_voltage_magnitude_vector', stochastic_scenario, str(time_index)][0]:
            optimization_problem.s1_index_locator[
                'node_voltage_magnitude_vector', stochastic_scenario, str(time_index)][
                1] + 1] = - np.diagflat(
                np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)]))

            b_1_temp = np.transpose(-np.array([node_voltage_magnitude_vector_minimum.ravel()])/np.array(
                [np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)]
            ))

            b_1 = np.vstack((b_1, b_1_temp))
            A_1 = np.vstack((A_1, A_1_temp))

    # constr 12 m)
    A_1_temp = np.zeros((len(linear_electric_grid_model.electric_grid_model.timesteps), A_1_column_number))
    b_1_temp = np.zeros((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))
    for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
        A_1_temp[time_index, optimization_problem.s1_index_locator['up_reserve', str(time_index)][0]] = -1

    b_1 = np.vstack((b_1, b_1_temp))
    A_1 = np.vstack((A_1, A_1_temp))

    A_1_temp = np.zeros((len(linear_electric_grid_model.electric_grid_model.timesteps), A_1_column_number))
    b_1_temp = np.zeros((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))
    for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
        A_1_temp[time_index, optimization_problem.s1_index_locator['down_reserve', str(time_index)][0]] = -1

    b_1 = np.vstack((b_1, b_1_temp))
    A_1 = np.vstack((A_1, A_1_temp))

    # Define objective vector.
    f_1 = np.zeros((1, A_1_column_number))
    # energy
    price_temp = price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T \
        * timestep_interval_hours
    for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
        f_1[optimization_problem.s1_index_locator['energy', str(time_index)][0]] = price_temp[time_index]

    # up reserve
    price_temp = -0.1 * price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T \
        * timestep_interval_hours
    for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
        f_1[optimization_problem.s1_index_locator['up_reserve', str(time_index)][0]] = price_temp[time_index]

    # down reserve
    price_temp = - 1.1 * price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T \
        * timestep_interval_hours
    for time_index in range(len(linear_electric_grid_model.electric_grid_model.timesteps)):
        f_1[optimization_problem.s1_index_locator['down_reserve', str(time_index)][0]] = price_temp[time_index]

    optimization_problem.objective += (f_1 @ optimization_problem.variable_vector_s1)

    optimization_problem.constraints.append(
            A_1 @ optimization_problem.variable_vector_s1 <= b_1)

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
