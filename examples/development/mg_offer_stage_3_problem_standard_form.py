"""Example script for DRO problem."""

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge

from mg_offer_stage_1_problem_standard_form import stage_1_problem_standard_form
from mg_offer_stage_2_problem_standard_form import stage_2_problem_standard_form


def stage_3_problem_standard_form(scenario_name):
    print('stage 3 problem modelling...')
    # Settings.
    #scenario_name = 'singapore_6node_custom'
    stochastic_scenarios = ['no_reserve', 'up_reserve', 'down_reserve']
    stochastic_scenarios_stage_3 = ['up_reserve_activated', 'down_reserve_activated']

    # Get results path.
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV definition files.
    #fledge.data_interface.recreate_database()

    # Obtain price data object.
    # price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain DER & grid model objects.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Getting linear electric grid model using "global approximation" method.
    linear_electric_grid_model = fledge.electric_grid_models.LinearElectricGridModelGlobal(scenario_name)

    # Instantiate optimization problem.
    # optimization_problem = fledge.utils.OptimizationProblem()

    # Instantiate standard form.
    standard_form = fledge.utils.StandardForm()

    # Define variables for stage 1.
    standard_form.define_variable('energy_s1', timestep=linear_electric_grid_model.electric_grid_model.timesteps)
    standard_form.define_variable('up_reserve_s1', timestep=linear_electric_grid_model.electric_grid_model.timesteps)
    standard_form.define_variable('down_reserve_s1', timestep=linear_electric_grid_model.electric_grid_model.timesteps)

    for stochastic_scenario in stochastic_scenarios:
        for der_name, der_model in der_model_set.flexible_der_models.items():
            standard_form.define_variable(
                'state_vector_s1', timestep=der_model.timesteps, der_name=[der_model.der_name], state=der_model.states,
                scenario=[stochastic_scenario]
            )
            standard_form.define_variable(
                'control_vector_s1', timestep=der_model.timesteps, der_name=[der_model.der_name],
                control=der_model.controls,
                scenario=[stochastic_scenario])
            standard_form.define_variable(
                'output_vector_s1', timestep=der_model.timesteps, der_name=[der_model.der_name],
                output=der_model.outputs,
                scenario=[stochastic_scenario]
            )

        standard_form.define_variable(
            'der_active_power_vector_s1', timestep=der_model_set.timesteps, der_model=der_model_set.ders,
            scenario=[stochastic_scenario]
        )

        standard_form.define_variable(
            'der_reactive_power_vector_s1', timestep=der_model_set.timesteps, der_model=der_model_set.ders,
            scenario=[stochastic_scenario]
        )

        standard_form.define_variable(
            'nodal_voltage_magnitude_s1', timestep=der_model_set.timesteps,
            node=linear_electric_grid_model.electric_grid_model.nodes,
            scenario=[stochastic_scenario]
        )

    # Define variables stage 2.
    standard_form.define_variable('energy_deviation_s2',
                                  timestep=linear_electric_grid_model.electric_grid_model.timesteps)
    standard_form.define_variable('uncertainty_energy_price_deviation_s2',
                                  timestep=linear_electric_grid_model.electric_grid_model.timesteps)
    standard_form.define_variable('uncertainty_up_reserve_price_deviation_s2',
                                  timestep=linear_electric_grid_model.electric_grid_model.timesteps)
    standard_form.define_variable('uncertainty_down_reserve_price_deviation_s2',
                                  timestep=linear_electric_grid_model.electric_grid_model.timesteps)

    for der_name, der_model in der_model_set.flexible_der_models.items():
        standard_form.define_variable(
            'state_vector_s2', timestep=der_model.timesteps, der_name=[der_model.der_name], state=der_model.states
        )
        standard_form.define_variable(
            'control_vector_s2', timestep=der_model.timesteps, der_name=[der_model.der_name],
            control=der_model.controls
        )
        standard_form.define_variable(
            'output_vector_s2', timestep=der_model.timesteps, der_name=[der_model.der_name],
            output=der_model.outputs
        )
        if not der_model.disturbances.empty:
            standard_form.define_variable(
                'uncertainty_disturbances_vector_s2', timestep=der_model.timesteps, der_name=[der_model.der_name],
                disturbance=der_model.disturbances,
            )

    standard_form.define_variable(
        'der_active_power_vector_s2', timestep=der_model_set.timesteps, der_model=der_model_set.ders,
    )

    standard_form.define_variable(
        'der_reactive_power_vector_s2', timestep=der_model_set.timesteps, der_model=der_model_set.ders,
    )

    standard_form.define_variable(
        'nodal_voltage_magnitude_s2', timestep=der_model_set.timesteps,
        node=linear_electric_grid_model.electric_grid_model.nodes,
    )

    # Define variables for stage 3.
    for stochastic_scenario in stochastic_scenarios_stage_3:
        standard_form.define_variable(
            'energy_deviation_s3', timestep=linear_electric_grid_model.electric_grid_model.timesteps,
            scenario=[stochastic_scenario]
        )

        for der_name, der_model in der_model_set.flexible_der_models.items():
            standard_form.define_variable(
                'state_vector_s3', timestep=der_model.timesteps, der_name=[der_model.der_name], state=der_model.states,
                scenario=[stochastic_scenario]
            )
            standard_form.define_variable(
                'control_vector_s3', timestep=der_model.timesteps, der_name=[der_model.der_name],
                control=der_model.controls,
                scenario=[stochastic_scenario])
            standard_form.define_variable(
                'output_vector_s3', timestep=der_model.timesteps, der_name=[der_model.der_name],
                output=der_model.outputs,
                scenario=[stochastic_scenario]
            )

        standard_form.define_variable(
            'der_active_power_vector_s3', timestep=der_model_set.timesteps, der=der_model_set.ders,
            scenario=[stochastic_scenario]
        )

        standard_form.define_variable(
            'der_reactive_power_vector_s3', timestep=der_model_set.timesteps, der=der_model_set.ders,
            scenario=[stochastic_scenario]
        )

        standard_form.define_variable(
            'nodal_voltage_magnitude_s3', timestep=der_model_set.timesteps,
            node=linear_electric_grid_model.electric_grid_model.nodes,
            scenario=[stochastic_scenario]
        )

    # Define power balance constraints.
    for timestep in der_model_set.timesteps:
        standard_form.define_constraint(
            ('variable', 1.0, dict(name='energy_s1', timestep=timestep)),
            ('variable', 1.0, dict(name='energy_deviation_s2', timestep=timestep)),
            ('variable', 1.0, dict(name='energy_deviation_s3', timestep=timestep, scenario=['up_reserve_activated'])),
            ('variable', 1.0, dict(name='up_reserve_s1', timestep=timestep)),
            '==',
            (
                'variable',
                -1.0 * np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]),
                dict(
                    name='der_active_power_vector_s3', timestep=timestep,
                    der=der_model_set.ders, scenario=['up_reserve_activated']
                )
            ),
        )
        standard_form.define_constraint(
            ('variable', 1.0, dict(name='energy_s1', timestep=timestep)),
            ('variable', 1.0, dict(name='energy_deviation_s2', timestep=timestep)),
            ('variable', 1.0, dict(name='energy_deviation_s3', timestep=timestep, scenario=['down_reserve_activated'])),
            ('variable', -1.0, dict(name='down_reserve_s1', timestep=timestep)),
            '==',
            (
                'variable',
                -1.0 * np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]),
                dict(
                    name='der_active_power_vector_s3', timestep=timestep,
                    der=der_model_set.ders, scenario=['down_reserve_activated']
                )
            ),
        )

    # Define power flow constraints.
    for stochastic_scenario in stochastic_scenarios_stage_3:
        for timestep in der_model_set.timesteps:
            standard_form.define_constraint(
                (
                    'variable',
                    np.diagflat(np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])),
                    dict(
                        name='nodal_voltage_magnitude_s3', timestep=timestep,
                        node=linear_electric_grid_model.electric_grid_model.nodes,
                        scenario=[stochastic_scenario]
                    )
                ),
                (
                    'variable',
                    -linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                    @ np.diagflat(np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])),
                    dict(
                        name='der_active_power_vector_s3', timestep=timestep,
                        der=der_model_set.ders, scenario=[stochastic_scenario]
                    )
                ),
                (
                    'variable',
                    -linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                    @ np.diagflat(np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])),
                    dict(
                        name='der_reactive_power_vector_s3', timestep=timestep,
                        der=der_model_set.ders, scenario=[stochastic_scenario]
                    )
                ),
                '==',
                (
                    'constant',
                    (
                        -1.0 * (
                            linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                            @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                            @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                        ) + np.transpose([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
                    )[:, 0]
                )
            )

    # Voltage upper / lower bound.
    node_voltage_magnitude_vector_minimum = (
        0.5 * np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
    )
    node_voltage_magnitude_vector_maximum = (
        1.5 * np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
    )
    for stochastic_scenario in stochastic_scenarios_stage_3:
        for timestep in linear_electric_grid_model.electric_grid_model.timesteps:
            standard_form.define_constraint(
                (
                    'variable',
                    np.diagflat(np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])),
                    dict(
                        name='nodal_voltage_magnitude_s3', timestep=timestep,
                        node=linear_electric_grid_model.electric_grid_model.nodes, scenario=[stochastic_scenario]
                    )
                ),
                '>=',
                ('constant', np.transpose(np.array([node_voltage_magnitude_vector_minimum.ravel()]))[:, 0])
            )
            standard_form.define_constraint(
                (
                    'variable',
                    np.diagflat(np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])),
                    dict(
                        name='nodal_voltage_magnitude_s3', timestep=timestep,
                        node=linear_electric_grid_model.electric_grid_model.nodes, scenario=[stochastic_scenario]
                    )
                ),
                '<=',
                ('constant', np.transpose(np.array([node_voltage_magnitude_vector_maximum.ravel()]))[:, 0])
            )

    for der_name, der_model in der_model_set.der_models.items():
        # Fixed DERs.
        if issubclass(type(der_model), fledge.der_models.FixedDERModel):
            if der_model.is_electric_grid_connected:
                for stochastic_scenario in stochastic_scenarios_stage_3:
                    for timestep in der_model_set.timesteps:
                        standard_form.define_constraint(
                            (
                                'variable',
                                1.0,
                                dict(
                                    name='der_active_power_vector_s3', timestep=timestep,
                                    der=(der_model.der_type, der_model.der_name), scenario=[stochastic_scenario]
                                )
                            ),
                            '==',
                            (
                                'constant',
                                (
                                    der_model.active_power_nominal_timeseries[timestep]
                                    / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)
                                ).tolist()
                            )
                        )

                        standard_form.define_constraint(
                            (
                                'variable',
                                1.0,
                                dict(
                                    name='der_reactive_power_vector_s3', timestep=timestep,
                                    der=(der_model.der_type, der_model.der_name), scenario=[stochastic_scenario]
                                )
                            ),
                            '==',
                            (
                                'constant',
                                (
                                    der_model.reactive_power_nominal_timeseries[timestep]
                                    / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)
                                ).tolist()
                            )
                        )

        # Flexible DERs.
        if issubclass(type(der_model), fledge.der_models.FlexibleDERModel):

            # Manipulate building model to avoid over-consumption for up-reserves.
            if issubclass(type(der_model), fledge.der_models.FlexibleBuildingModel):
                der_model.output_maximum_timeseries.loc[
                    :, der_model.output_maximum_timeseries.columns.str.contains('_heat_')
                ] = 0.0

            # Initial state.
            # - For states which represent storage state of charge, initial state of charge is final state of charge.
            for stochastic_scenario in stochastic_scenarios_stage_3:
                # Initial state.
                if any(~der_model.states.isin(der_model.storage_states)):
                    standard_form.define_constraint(
                        (
                            'constant',
                            der_model.state_vector_initial.values[~der_model.states.isin(der_model.storage_states)]
                        ),
                        '==',
                        (
                            'variable',
                            1.0,
                            dict(
                                name='state_vector_s3', der_name=[der_model.der_name], timestep=der_model.timesteps[0],
                                state=der_model.states[~der_model.states.isin(der_model.storage_states)],
                                scenario=[stochastic_scenario],
                            )
                        )
                    )
                if any(der_model.states.isin(der_model.storage_states)):
                    standard_form.define_constraint(
                        (
                            'variable',
                            1.0,
                            dict(
                                name='state_vector_s3', der_name=[der_model.der_name], timestep=der_model.timesteps[0],
                                state=der_model.states[der_model.states.isin(der_model.storage_states)],
                                scenario=[stochastic_scenario]
                            )
                        ),
                        '==',
                        (
                            'variable',
                            1.0,
                            dict(
                                name='state_vector_s3', der_name=[der_model.der_name], timestep=der_model.timesteps[-1],
                                state=der_model.states[der_model.states.isin(der_model.storage_states)],
                                scenario=[stochastic_scenario]
                            )
                        )
                    )

                # State equation.
                for timestep, timestep_previous in zip(der_model.timesteps[1:], der_model.timesteps[:-1]):
                    if not der_model.disturbances.empty:
                        standard_form.define_constraint(
                            (
                                'variable',
                                1.0,
                                dict(
                                    name='state_vector_s3', der_name=[der_model.der_name], timestep=timestep,
                                    state=der_model.states, scenario=[stochastic_scenario]
                                )
                            ),
                            '==',
                            (
                                'variable',
                                der_model.state_matrix.values,
                                dict(
                                    name='state_vector_s3', der_name=[der_model.der_name], timestep=timestep_previous,
                                    state=der_model.states, scenario=[stochastic_scenario]
                                )
                            ),
                            (
                                'variable',
                                der_model.control_matrix.values,
                                dict(
                                    name='control_vector_s3', der_name=[der_model.der_name], timestep=timestep_previous,
                                    control=der_model.controls, scenario=[stochastic_scenario]
                                )
                            ),
                            (
                                'constant',
                                der_model.disturbance_matrix.values
                                @ der_model.disturbance_timeseries.loc[timestep_previous, :].values
                            ),
                            (
                                'variable', der_model.disturbance_matrix.values, dict(
                                 name='uncertainty_disturbances_vector_s2', timestep=timestep_previous,
                                 der_name=[der_model.der_name], disturbance=der_model.disturbances
                                )
                            )
                        )
                    else:
                        standard_form.define_constraint(
                            (
                                'variable',
                                1.0,
                                dict(
                                    name='state_vector_s3', der_name=[der_model.der_name], timestep=timestep,
                                    state=der_model.states, scenario=[stochastic_scenario]
                                )
                            ),
                            '==',
                            (
                                'variable',
                                der_model.state_matrix.values,
                                dict(
                                    name='state_vector_s3', der_name=[der_model.der_name], timestep=timestep_previous,
                                    state=der_model.states, scenario=[stochastic_scenario]
                                )
                            ),
                            (
                                'variable',
                                der_model.control_matrix.values,
                                dict(
                                    name='control_vector_s3', der_name=[der_model.der_name], timestep=timestep_previous,
                                    control=der_model.controls, scenario=[stochastic_scenario]
                                )
                            ),
                            (
                                'constant',
                                der_model.disturbance_matrix.values
                                @ der_model.disturbance_timeseries.loc[timestep_previous, :].values
                            )
                        )

                # Output equation.
                for timestep in der_model.timesteps:
                    if not der_model.disturbances.empty:
                        standard_form.define_constraint(
                            (
                                'variable',
                                1.0,
                                dict(
                                    name='output_vector_s3', der_name=[der_model.der_name],
                                    timestep=timestep, output=der_model.outputs, scenario=[stochastic_scenario]
                                )
                            ),
                            '==',
                            (
                                'variable',
                                der_model.state_output_matrix.values,
                                dict(
                                    name='state_vector_s3', der_name=[der_model.der_name], timestep=timestep,
                                    state=der_model.states, scenario=[stochastic_scenario]
                                )
                            ),
                            (
                                'variable',
                                der_model.control_output_matrix.values,
                                dict(
                                    name='control_vector_s3', der_name=[der_model.der_name], timestep=timestep,
                                    control=der_model.controls, scenario=[stochastic_scenario]
                                )
                            ),
                            (
                                'constant',
                                der_model.disturbance_output_matrix.values
                                @ der_model.disturbance_timeseries.loc[timestep, :].values
                            ),
                            (
                                'variable', der_model.disturbance_output_matrix.values, dict(
                                    name='uncertainty_disturbances_vector_s2', timestep=timestep,
                                    der_name=[der_model.der_name], disturbance=der_model.disturbances
                                )
                            )
                        )
                    else:
                        standard_form.define_constraint(
                            (
                                'variable',
                                1.0,
                                dict(
                                    name='output_vector_s3', der_name=[der_model.der_name],
                                    timestep=timestep, output=der_model.outputs, scenario=[stochastic_scenario]
                                )
                            ),
                            '==',
                            (
                                'variable',
                                der_model.state_output_matrix.values,
                                dict(
                                    name='state_vector_s3', der_name=[der_model.der_name], timestep=timestep,
                                    state=der_model.states, scenario=[stochastic_scenario]
                                )
                            ),
                            (
                                'variable',
                                der_model.control_output_matrix.values,
                                dict(
                                    name='control_vector_s3', der_name=[der_model.der_name], timestep=timestep,
                                    control=der_model.controls, scenario=[stochastic_scenario]
                                )
                            ),
                            (
                                'constant',
                                der_model.disturbance_output_matrix.values
                                @ der_model.disturbance_timeseries.loc[timestep, :].values
                            )
                        )

                # Define connection constraints.
                if der_model.is_electric_grid_connected:
                    for timestep in der_model.timesteps:

                        # Active power.
                        standard_form.define_constraint(
                            (
                                (
                                    'variable',
                                    der_model.mapping_active_power_by_output.values
                                    / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0),
                                    dict(
                                        name='output_vector_s3', der_name=[der_model.der_name], timestep=timestep,
                                        output=der_model.outputs, scenario=[stochastic_scenario]
                                    )
                                ) if timestep != der_model.timesteps[-1] else (
                                    'constant',
                                    0.0
                                )
                            ),
                            '==',
                            (
                                'variable',
                                1.0,
                                dict(
                                    name='der_active_power_vector_s3', timestep=timestep,
                                    der=(der_model.der_type, der_model.der_name), scenario=[stochastic_scenario]
                                )
                            )
                        )

                        # Reactive power.
                        standard_form.define_constraint(
                            (
                                (
                                    'variable',
                                    der_model.mapping_reactive_power_by_output.values
                                    / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0),
                                    dict(
                                        name='output_vector_s3', der_name=[der_model.der_name], timestep=timestep,
                                        output=der_model.outputs, scenario=[stochastic_scenario]
                                    )
                                ) if timestep != der_model.timesteps[-1] else (
                                    'constant',
                                    0.0
                                )
                            ),
                            '==',
                            (
                                'variable',
                                1.0,
                                dict(
                                    name='der_reactive_power_vector_s3', timestep=timestep,
                                    der=(der_model.der_type, der_model.der_name), scenario=[stochastic_scenario]
                                )
                            )
                        )

                # Output limits.
                for timestep in der_model.timesteps:
                    standard_form.define_constraint(
                        (
                            'variable',
                            1.0,
                            dict(
                                name='output_vector_s3', der_name=[der_model.der_name], timestep=timestep,
                                output=der_model.outputs, scenario=[stochastic_scenario]
                            )
                        ),
                        '>=',
                        ('constant', der_model.output_minimum_timeseries.loc[timestep, :].values)
                    )
                    standard_form.define_constraint(
                        (
                            'variable',
                            1.0,
                            dict(
                                name='output_vector_s3', der_name=[der_model.der_name], timestep=timestep,
                                output=der_model.outputs, scenario=[stochastic_scenario]
                            )
                        ),
                        '<=',
                        ('constant', der_model.output_maximum_timeseries.loc[timestep, :].values)
                    )

    # Obtain standard form matrix / vector representation.
    a_matrix = standard_form.get_a_matrix()
    b_vector = standard_form.get_b_vector()

    # Obtain DRO constraint matrices A3, B3, C3 and D3

    # A3 matrix
    voltage_s1_index = fledge.utils.get_index(
        standard_form.variables, name='nodal_voltage_magnitude_s1',
        timestep=der_model_set.timesteps, node=linear_electric_grid_model.electric_grid_model.nodes,
        scenario=['down_reserve']
    )

    s1_last_index = max(voltage_s1_index)

    A3_matrix = a_matrix[:, 0:s1_last_index+1]

    # C3 matrix
    delta_indices = np.array([])

    temp_indices = fledge.utils.get_index(
        standard_form.variables, name='uncertainty_energy_price_deviation_s2',
        timestep=linear_electric_grid_model.electric_grid_model.timesteps
    )

    delta_indices = np.hstack((delta_indices, temp_indices))

    temp_indices = fledge.utils.get_index(
        standard_form.variables, name='uncertainty_up_reserve_price_deviation_s2',
        timestep=linear_electric_grid_model.electric_grid_model.timesteps
    )

    delta_indices = np.hstack((delta_indices, temp_indices))

    temp_indices = fledge.utils.get_index(
        standard_form.variables, name='uncertainty_down_reserve_price_deviation_s2',
        timestep=linear_electric_grid_model.electric_grid_model.timesteps
    )

    delta_indices = np.hstack((delta_indices, temp_indices))

    for der_name, der_model in der_model_set.flexible_der_models.items():
        if not der_model.disturbances.empty:
            temp_indices = fledge.utils.get_index(
                standard_form.variables, name='uncertainty_disturbances_vector_s2',
                timestep=der_model.timesteps, der_name=[der_model.der_name],
                disturbance=der_model.disturbances,
            )

            delta_indices = np.hstack((delta_indices, temp_indices))

    C3_matrix = a_matrix[:, delta_indices]

    # B3 matrix
    s1_indices = np.linspace(0, s1_last_index, s1_last_index+1)

    voltage_s2_index = fledge.utils.get_index(
        standard_form.variables, name='nodal_voltage_magnitude_s2', timestep=der_model_set.timesteps,
        node=linear_electric_grid_model.electric_grid_model.nodes,
    )

    s2_last_index = max(voltage_s2_index)

    s2_indices = np.setdiff1d(
        np.linspace(0, s2_last_index, s2_last_index+1), np.hstack((s1_indices, delta_indices))
    )

    B3_matrix = a_matrix[:, s2_indices]

    # D3 matrix
    s3_indices = np.setdiff1d(
        np.linspace(0, a_matrix.shape[1]-1, a_matrix.shape[1]), np.hstack((s1_indices, s2_indices, delta_indices))
    )

    D3_matrix = a_matrix[:, s3_indices]

    # objective matrices
    # m_Q3_s2
    der_cost_factor = 0.01
    penalty_factor = 0.1
    up_reserve_activated_probability = 0.3
    down_reserve_activated_probability = 0.25
    m_Q3_s2 = np.zeros((s2_indices.shape[0], 1))

    der_active_power_vector_s2_indices = fledge.utils.get_index(
        standard_form.variables, name='der_active_power_vector_s2', timestep=der_model_set.timesteps,
        der_model=der_model_set.ders,
    )

    m_Q3_s2[np.where(pd.Index(s2_indices).isin(der_active_power_vector_s2_indices)), 0] = \
        (up_reserve_activated_probability+down_reserve_activated_probability)*der_cost_factor

    # m_Q3_s3
    m_Q3_s3 = np.zeros((s3_indices.shape[0], 1))

    energy_deviation_up_reserve_s3_indices = fledge.utils.get_index(
        standard_form.variables, name='energy_deviation_s3', timestep=linear_electric_grid_model.electric_grid_model.timesteps,
            scenario=['up_reserve_activated']
    )

    energy_deviation_down_reserve_s3_indices = fledge.utils.get_index(
        standard_form.variables, name='energy_deviation_s3', timestep=linear_electric_grid_model.electric_grid_model.timesteps,
            scenario=['down_reserve_activated']
    )

    der_active_power_vector_up_reserve_s3_indices = fledge.utils.get_index(
        standard_form.variables, name='der_active_power_vector_s3', timestep=der_model_set.timesteps,
        der=der_model_set.ders, scenario=['up_reserve_activated']
    )
    der_active_power_vector_down_reserve_s3_indices = fledge.utils.get_index(
        standard_form.variables, name='der_active_power_vector_s3', timestep=der_model_set.timesteps,
        der=der_model_set.ders, scenario=['down_reserve_activated']
    )

    m_Q3_s3[np.where(pd.Index(s3_indices).isin(energy_deviation_up_reserve_s3_indices)), 0] = \
        -penalty_factor*up_reserve_activated_probability

    m_Q3_s3[np.where(pd.Index(s3_indices).isin(energy_deviation_down_reserve_s3_indices)), 0] = \
        -penalty_factor*down_reserve_activated_probability

    m_Q3_s3[np.where(pd.Index(s3_indices).isin(der_active_power_vector_up_reserve_s3_indices)), 0] = \
        -der_cost_factor*up_reserve_activated_probability

    m_Q3_s3[np.where(pd.Index(s3_indices).isin(der_active_power_vector_down_reserve_s3_indices)), 0] = \
        -der_cost_factor*down_reserve_activated_probability

    return standard_form, b_vector, A3_matrix, B3_matrix, C3_matrix, D3_matrix, m_Q3_s2, m_Q3_s3, delta_indices, \
           s1_indices, s2_indices, s3_indices


def main():
    scenario_name = 'singapore_6node_custom'
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Get results path.
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    standard_form_stage_1, a_matrix, b_vector, f_vector, stochastic_scenarios, der_model_set \
        = stage_1_problem_standard_form(scenario_name)
    # Instantiate optimization problem.
    optimization_problem_stage_1 = fledge.utils.OptimizationProblem()

    # Define optimization problem.
    optimization_problem_stage_1.x_vector = cp.Variable((len(standard_form_stage_1.variables), 1))
    optimization_problem_stage_1.constraints.append(
        a_matrix.toarray() @ optimization_problem_stage_1.x_vector <= b_vector
    )
    optimization_problem_stage_1.objective += (
        (
                f_vector.T
                @ optimization_problem_stage_1.x_vector
        )
    )
    # Define optimization objective

    # Solve optimization problem.
    optimization_problem_stage_1.solve()

    # Obtain results.
    results = standard_form_stage_1.get_results(optimization_problem_stage_1.x_vector)

    energy_offer_stage_1 = pd.Series(results['energy'].values.ravel(), index=der_model_set.timesteps)
    up_reserve_offer_stage_1 = pd.Series(results['up_reserve'].values.ravel(), index=der_model_set.timesteps)
    down_reserve_offer_stage_1 = pd.Series(results['down_reserve'].values.ravel(), index=der_model_set.timesteps)

    standard_form_stage_2, b2_vector, A2_matrix, B2_matrix, C2_matrix, M_Q2_delta, m_Q2_s2, s2_indices_stage2, \
    delta_indices_stage2, s1_indices = stage_2_problem_standard_form(scenario_name)

    # Define stage 2 problem
    optimization_problem_stage_2 = fledge.utils.OptimizationProblem()
    # Define optimization problem.
    optimization_problem_stage_2.s_1 = cp.Variable((len(s1_indices), 1))
    optimization_problem_stage_2.s_2 = cp.Variable((len(s2_indices_stage2), 1))
    optimization_problem_stage_2.delta = cp.Variable((len(delta_indices_stage2), 1))

    optimization_problem_stage_2.constraints.append(
        A2_matrix.toarray() @ optimization_problem_stage_2.s_1 + B2_matrix.toarray() @ optimization_problem_stage_2.s_2
        + C2_matrix @ optimization_problem_stage_2.delta <= b2_vector
    )

    optimization_problem_stage_2.constraints.append(
        optimization_problem_stage_2.delta == 0
    )

    index_energy_stage_2 = fledge.utils.get_index(
        standard_form_stage_2.variables, name='energy_s1',
        timestep=der_model_set.timesteps,
    )

    optimization_problem_stage_2.constraints.append(
        optimization_problem_stage_2.s_1[np.where(pd.Index(s1_indices).isin(index_energy_stage_2))[0], 0]
        == np.transpose(energy_offer_stage_1.to_numpy())
    )

    optimization_problem_stage_2.constraints.append(
        optimization_problem_stage_2.s_1[max(index_energy_stage_2)+1:-1, 0]
        == 0
    )

    optimization_problem_stage_2.objective += (
        #     (
        #         optimization_problem_stage_2.s_1.T @ M_Q2_delta @ optimization_problem_stage_2.delta
        #    ) +
        (
                m_Q2_s2.T @ optimization_problem_stage_2.s_2
        )
    )

    # Solve optimization problem.
    optimization_problem_stage_2.solve()

    # Obtain results.
    index_energy_deviation_stage_2 = fledge.utils.get_index(
        standard_form_stage_2.variables, name='energy_deviation_s2',
        timestep=der_model_set.timesteps,
    )

    result_energy_deviation_stage_2 = \
        optimization_problem_stage_2.s_2.value[
            np.where(pd.Index(s2_indices_stage2).isin(index_energy_deviation_stage_2))[0], 0]

    standard_form_stage_3, b3_vector, A3_matrix, B3_matrix, C3_matrix, D3_matrix, m_Q3_s2, m_Q3_s3, \
        delta_indices_stage3, s1_indices_stage3, s2_indices_stage3, s3_indices_stage3 = stage_3_problem_standard_form(scenario_name)

    # Define stage 3 problem
    optimization_problem_stage_3 = fledge.utils.OptimizationProblem()
    # Define optimization problem.
    optimization_problem_stage_3.s_1 = cp.Variable((len(s1_indices_stage3), 1))
    optimization_problem_stage_3.s_2 = cp.Variable((len(s2_indices_stage3), 1))
    optimization_problem_stage_3.s_3 = cp.Variable((len(s3_indices_stage3), 1))
    optimization_problem_stage_3.delta = cp.Variable((len(delta_indices_stage3), 1))

    optimization_problem_stage_3.constraints.append(
        A3_matrix.toarray() @ optimization_problem_stage_3.s_1 + B3_matrix.toarray() @ optimization_problem_stage_3.s_2
        + C3_matrix @ optimization_problem_stage_3.delta + D3_matrix.toarray() @ optimization_problem_stage_3.s_3
        <= b3_vector
    )

    optimization_problem_stage_3.constraints.append(
        optimization_problem_stage_3.delta == 0
    )

    index_energy_stage_2 = fledge.utils.get_index(
        standard_form_stage_2.variables, name='energy_s1',
        timestep=der_model_set.timesteps,
    )

    index_up_reserve_offer = fledge.utils.get_index(
        standard_form_stage_2.variables, name='up_reserve_s1',
        timestep=der_model_set.timesteps,
    )

    index_down_reserve_offer = fledge.utils.get_index(
        standard_form_stage_2.variables, name='down_reserve_s1',
        timestep=der_model_set.timesteps,
    )

    optimization_problem_stage_3.constraints.append(
        optimization_problem_stage_3.s_1[np.where(pd.Index(s1_indices).isin(index_energy_stage_2))[0], 0]
        == np.transpose(energy_offer_stage_1.to_numpy())
    )

    optimization_problem_stage_3.constraints.append(
        optimization_problem_stage_3.s_1[np.where(pd.Index(s1_indices).isin(index_up_reserve_offer))[0], 0]
        == np.transpose(up_reserve_offer_stage_1.to_numpy())
    )

    optimization_problem_stage_3.constraints.append(
        optimization_problem_stage_3.s_1[np.where(pd.Index(s1_indices).isin(index_down_reserve_offer))[0], 0]
        == np.transpose(down_reserve_offer_stage_1.to_numpy())
    )

    optimization_problem_stage_3.constraints.append(
        optimization_problem_stage_3.s_1[max(index_down_reserve_offer)+1:-1, 0]
        == 0
    )

    index_energy_deviation_stage_2 = fledge.utils.get_index(
        standard_form_stage_3.variables, name='energy_deviation_s2',
        timestep=der_model_set.timesteps,
    )

    optimization_problem_stage_3.constraints.append(
        optimization_problem_stage_3.s_2[np.where(pd.Index(s2_indices_stage3).isin(index_energy_deviation_stage_2))[0], 0]
        == result_energy_deviation_stage_2
    )

    optimization_problem_stage_3.constraints.append(
        optimization_problem_stage_3.s_2[max(np.where(pd.Index(s2_indices_stage3).isin(index_energy_deviation_stage_2))[0])+1:-1, 0]
        == 0
    )

    # For DER nodal injections, p^j_der, the value is set as zero as they don't affect the optimization results (const)
    optimization_problem_stage_3.objective += (
        #     (
        #         optimization_problem_stage_2.s_1.T @ M_Q2_delta @ optimization_problem_stage_2.delta
        #    ) +
        (
                m_Q3_s2.T @ optimization_problem_stage_3.s_2
        ) +
        (
                m_Q3_s3.T @ optimization_problem_stage_3.s_3
        )
    )

    # Solve optimization problem.
    optimization_problem_stage_3.solve()

    # Obtain results.
    index_energy_deviation_stage_3_up_reserve_activated = fledge.utils.get_index(
        standard_form_stage_3.variables, name='energy_deviation_s3',
        timestep=der_model_set.timesteps, scenario='up_reserve_activated'
    )

    result_energy_deviation_stage_3_up_reserve_activated = \
        optimization_problem_stage_3.s_3.value[
            np.where(pd.Index(s3_indices_stage3).isin(index_energy_deviation_stage_3_up_reserve_activated))[0], 0]




if __name__ == '__main__':
    main()
