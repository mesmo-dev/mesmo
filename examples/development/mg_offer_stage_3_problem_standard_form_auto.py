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
    stochastic_scenarios_stage_3 = ['up_reserve_activated', 'down_reserve_activated']

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

    # Define optimization problem.
    optimization_problem.x_vector = cp.Variable((len(standard_form.variables), 1))
    optimization_problem.constraints.append(
        a_matrix.toarray() @ optimization_problem.x_vector <= b_vector
    )

    # Obtain timestep interval in hours, for conversion of power to energy.
    timestep_interval_hours = (der_model_set.timesteps[1] - der_model_set.timesteps[0]) / pd.Timedelta('1h')

    # Obtain energy price timeseries.
    price_timeseries_energy = (
        price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T * timestep_interval_hours
    )

    # Define objective.
    # Active power cost / revenue.
    # - Cost for load / demand, revenue for generation / supply.
    x_index_energy = (
        fledge.utils.get_index(
            standard_form.variables, name='energy', timestep=linear_electric_grid_model.electric_grid_model.timesteps
        )
    )
    x_index_up_reserve = (
        fledge.utils.get_index(
            standard_form.variables, name='up_reserve', timestep=linear_electric_grid_model.electric_grid_model.timesteps
        )
    )
    x_index_down_reserve = (
        fledge.utils.get_index(
            standard_form.variables, name='down_reserve', timestep=linear_electric_grid_model.electric_grid_model.timesteps
        )
    )

    optimization_problem.objective += (
        (
            np.array([price_timeseries_energy])
            @ optimization_problem.x_vector[x_index_energy, :]
        )
        + (
            -0.1 * np.array([price_timeseries_energy])
            @ optimization_problem.x_vector[x_index_up_reserve, :]
        )
        + (
            -1.1 * np.array([price_timeseries_energy])
            @ optimization_problem.x_vector[x_index_down_reserve, :]
        )
        # + (
        #     1e-2 * cp.sum(
        #         optimization_problem.x_vector[x_index_energy, :] ** 2
        #         + optimization_problem.x_vector[x_index_up_reserve, :] ** 2
        #         + optimization_problem.x_vector[x_index_down_reserve, :] ** 2
        #     )
        # )
    )

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = standard_form.get_results(optimization_problem.x_vector)

    # Obtain reserve results.
    no_reserve = pd.Series(results['energy'].values.ravel(), index=der_model_set.timesteps)
    up_reserve = pd.Series(results['up_reserve'].values.ravel(), index=der_model_set.timesteps)
    down_reserve = pd.Series(results['down_reserve'].values.ravel(), index=der_model_set.timesteps)

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
                results['state_vector'].loc[:, (stochastic_scenario, der_name, slice(None))].values
            )
            control_vector[stochastic_scenario].loc[:, (der_name, slice(None))] = (
                results['control_vector'].loc[:, (stochastic_scenario, der_name, slice(None))].values
            )
            output_vector[stochastic_scenario].loc[:, (der_name, slice(None))] = (
                results['output_vector'].loc[:, (stochastic_scenario, der_name, slice(None))].values
            )
        for der_name, der_model in der_model_set.der_models.items():
            if der_model.is_electric_grid_connected:
                der_active_power_vector_per_unit[stochastic_scenario].loc[:, (der_model.der_type, der_name)] = (
                    results['der_active_power_vector'].loc[:, [(stochastic_scenario, (der_model.der_type, der_name))]].values
                )
                der_active_power_vector[stochastic_scenario].loc[:, (slice(None), der_name)] = (
                    der_active_power_vector_per_unit[stochastic_scenario].loc[:, (slice(None), der_name)].values
                    * der_model.active_power_nominal
                )
                der_reactive_power_vector_per_unit[stochastic_scenario].loc[:, (slice(None), der_name)] = (
                    results['der_reactive_power_vector'].loc[:, [(stochastic_scenario, (der_model.der_type, der_name))]].values
                )
                der_reactive_power_vector[stochastic_scenario].loc[:, (slice(None), der_name)] = (
                    der_reactive_power_vector_per_unit[stochastic_scenario].loc[:, (slice(None), der_name)].values
                    * der_model.reactive_power_nominal
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
