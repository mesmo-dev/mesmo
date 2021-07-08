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

    # Define grid limits.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference

    # Obtain standard form.
    fledge.utils.log_time('standard-form interface')
    standard_form = fledge.utils.StandardForm()

    # Define problem for electric grid model
    fledge.utils.log_time('standard-form problem')

    # Define variables.
    # Define DER power vector variables.
    # TODO: I added MultiIndex to utils.py, line 267 --> or should this be done with is subclass?
    standard_form.define_variable(
        'der_active_power_vector',
        timestep=scenario_data.timesteps,
        der_index=linear_electric_grid_model.electric_grid_model.ders
    )
    standard_form.define_variable(
        'der_reactive_power_vector',
        timestep=scenario_data.timesteps,
        der_index=linear_electric_grid_model.electric_grid_model.ders
    )
    # Define node voltage variable.
    standard_form.define_variable(
        'node_voltage_magnitude_vector',
        timestep=scenario_data.timesteps,
        node=linear_electric_grid_model.electric_grid_model.nodes
    )
    # Define branch power magnitude variables.
    standard_form.define_variable(
        'branch_power_magnitude_vector_1',
        timestep=scenario_data.timesteps,
        branch=linear_electric_grid_model.electric_grid_model.branches
    )
    standard_form.define_variable(
        'branch_power_magnitude_vector_2',
        timestep=scenario_data.timesteps,
        branch=linear_electric_grid_model.electric_grid_model.branches
    )
    # Define loss variables.
    # TODO: is this correct with only one dimension?
    standard_form.define_variable(
        'loss_active',
        timestep=scenario_data.timesteps
    )
    standard_form.define_variable(
        'loss_reactive',
        timestep=scenario_data.timesteps
    )

    # Define constraints.
    # Define voltage variable terms.
    voltage_active_term = np.multiply(
        (linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active.T / (
            np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)]))).T,
        np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))

    voltage_reactive_term = np.multiply(
        (linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive.T / (
            np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)]))).T,
        np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))

    # Define voltage constant term.
    voltage_constant = ((
            - linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
            @ np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]).T
            - linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
            @ np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]).T
            ).T + np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
            ) / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])

    # Define voltage equation.
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='node_voltage_magnitude_vector', timestep=scenario_data.timesteps, node=linear_electric_grid_model.electric_grid_model.nodes)),
        '==',
        ('variable', voltage_active_term, dict(name='der_active_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)),
        ('variable', voltage_reactive_term, dict(name='der_reactive_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)),
        ('constant', voltage_constant.repeat(len(scenario_data.timesteps))),
        broadcast='timestep'
    )

    # Define branch flow (direction 1) variable terms.
    branch_power_1_active_variable = np.multiply(
        (linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active.T / (
            np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference]))).T,
        np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))
    
    branch_power_1_reactive_variable = np.multiply(
        (linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive.T / (
            np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference]))).T,
        np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))

    # Define branch flow (direction 1) constant terms.
    branch_power_1_constant = ((
        - linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
        @ np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]).T
        - linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
        @ np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]).T
        ).T + np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1.ravel())])
        ) / np.array([np.abs(linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference)])

    # Define branch flow (direction 1) equation.
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='branch_power_magnitude_vector_1', timestep=scenario_data.timesteps, branch=linear_electric_grid_model.electric_grid_model.branches)),
        '==',
        ('variable', branch_power_1_active_variable, dict(name='der_active_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)),
        ('variable', branch_power_1_reactive_variable, dict(name='der_reactive_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)),
        ('constant', branch_power_1_constant.repeat(len(scenario_data.timesteps))),
        broadcast='timestep'
    )

    # Define branch flow (direction 2) variable terms.
    branch_power_2_active_variable = np.multiply(
        (linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active.T / (
            np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference]))).T,
        np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))

    branch_power_2_reactive_variable = np.multiply(
        (linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive.T / (
            np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference]))).T,
        np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))

    # Define branch flow (direction 2) constant terms.
    branch_power_2_constant = ((
        - linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
        @ np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]).T
        - linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
        @ np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]).T
        ).T + np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2.ravel())])
        ) / np.array([np.abs(linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference)])

    # Define branch flow (direction 2) equation.
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='branch_power_magnitude_vector_2', timestep=scenario_data.timesteps, branch=linear_electric_grid_model.electric_grid_model.branches)),
        '==',
        ('variable', branch_power_2_active_variable, dict(name='der_active_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)),
        ('variable', branch_power_2_reactive_variable, dict(name='der_reactive_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)),
        ('constant', branch_power_2_constant.repeat(len(scenario_data.timesteps))),
        broadcast='timestep'
    )
    
    # Define active loss variable terms.
    loss_active_active_variable = np.multiply(
        linear_electric_grid_model.sensitivity_loss_active_by_der_power_active.toarray(),
        np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))

    loss_active_reactive_variable = np.multiply(
        linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive.toarray(),
        np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))

    # Define active loss constant term.
    loss_active_constant = ((
        - linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
        @ np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]).T
        - linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
        @ np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]).T
        ).T + np.real(linear_electric_grid_model.power_flow_solution.loss)
    )
    
    # Define active loss equation.
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='loss_active', timestep=scenario_data.timesteps)),
        '==',
        ('variable', loss_active_active_variable, dict(name='der_active_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)),
        ('variable', loss_active_reactive_variable, dict(name='der_reactive_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)),
        ('constant', loss_active_constant.repeat(len(scenario_data.timesteps))),
        broadcast='timestep'
    )
    
    # Define reactive loss variable terms.
    loss_reactive_active_variable = np.multiply(
        linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active.toarray(),
        np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))

    loss_reactive_reactive_variable = np.multiply(
        linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive.toarray(),
        np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]))
    
    # Define active loss constant term.
    loss_reactive_constant = ((
        - linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
        @ np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]).T
        - linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
        @ np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]).T
        ).T + np.imag(linear_electric_grid_model.power_flow_solution.loss)
    )

    # Define reactive loss equation.
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='loss_reactive', timestep=scenario_data.timesteps)),
        '==',
        ('variable', loss_reactive_active_variable, dict(name='der_active_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)),
        ('variable', loss_reactive_reactive_variable, dict(name='der_reactive_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)),
        ('constant', loss_reactive_constant.repeat(len(scenario_data.timesteps))),
        broadcast='timestep'
    )

    # Define voltage limits.
    # Add dedicated keys to enable retrieving dual variables.
    voltage_limit_minimum = (
        np.array([node_voltage_magnitude_vector_minimum.ravel()])
        / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
    ).ravel()
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='node_voltage_magnitude_vector', timestep=scenario_data.timesteps, node=linear_electric_grid_model.electric_grid_model.nodes)),
        '>=',
        ('constant', voltage_limit_minimum.repeat(len(scenario_data.timesteps))),
        keys=dict(name='voltage_magnitude_vector_minimum_constraint', timestep=scenario_data.timesteps, node=linear_electric_grid_model.electric_grid_model.nodes),
        broadcast='timestep'
    )
    voltage_limit_maximum = (
            np.array([node_voltage_magnitude_vector_maximum.ravel()])
            / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
    ).ravel()
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='node_voltage_magnitude_vector', timestep=scenario_data.timesteps, node=linear_electric_grid_model.electric_grid_model.nodes)),
        '<=',
        ('constant', voltage_limit_maximum.repeat(len(scenario_data.timesteps))),
        keys=dict(name='voltage_magnitude_vector_maximum_constraint', timestep=scenario_data.timesteps, node=linear_electric_grid_model.electric_grid_model.nodes),
        broadcast='timestep'
    )

    # Define branch flow limits.
    # Add dedicated keys to enable retrieving dual variables.
    branch_power_minimum = (
        - np.array([branch_power_magnitude_vector_maximum.ravel()])
        / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
    )
    branch_power_maximum = (
        np.array([branch_power_magnitude_vector_maximum.ravel()])
        / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
    )
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='branch_power_magnitude_vector_1', timestep=scenario_data.timesteps, branch=linear_electric_grid_model.electric_grid_model.branches)),
        '>=',
        ('constant', branch_power_minimum.repeat(len(scenario_data.timesteps))),
        keys=dict(name='branch_power_magnitude_vector_1_minimum_constraint', timestep=scenario_data.timesteps, branch=linear_electric_grid_model.electric_grid_model.branches),
        broadcast='timestep'
    )
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='branch_power_magnitude_vector_1', timestep=scenario_data.timesteps, branch=linear_electric_grid_model.electric_grid_model.branches)),
        '<=',
        ('constant', branch_power_maximum.repeat(len(scenario_data.timesteps))),
        keys=dict(name='branch_power_magnitude_vector_1_maximum_constraint', timestep=scenario_data.timesteps, branch=linear_electric_grid_model.electric_grid_model.branches),
        broadcast='timestep'
    )
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='branch_power_magnitude_vector_2', timestep=scenario_data.timesteps, branch=linear_electric_grid_model.electric_grid_model.branches)),
        '>=',
        ('constant', branch_power_minimum.repeat(len(scenario_data.timesteps))),
        keys=dict(name='branch_power_magnitude_vector_2_minimum_constraint', timestep=scenario_data.timesteps, branch=linear_electric_grid_model.electric_grid_model.branches),
        broadcast='timestep'
    )
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='branch_power_magnitude_vector_2', timestep=scenario_data.timesteps, branch=linear_electric_grid_model.electric_grid_model.branches)),
        '<=',
        ('constant', branch_power_maximum.repeat(len(scenario_data.timesteps))),
        keys=dict(name='branch_power_magnitude_vector_2_maximum_constraint', timestep=scenario_data.timesteps, branch=linear_electric_grid_model.electric_grid_model.branches),
        broadcast='timestep'
    )

    # Obtain timestep interval in hours, for conversion of power to energy.
    timestep_interval_hours = (scenario_data.timesteps[1] - scenario_data.timesteps[0]) / pd.Timedelta('1h')

    # Define objective.
    # Active power cost / revenue.
    # - Cost for load / demand, revenue for generation / supply.
    standard_form.define_objective_low_level(
        variables=[(
            price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.reshape(1, len(scenario_data.timesteps))
            * -1.0 * timestep_interval_hours  # In Wh.
            @ sp.block_diag([np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])] * len(scenario_data.timesteps)),
            dict(name='der_active_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)
        ), (
            price_data.price_timeseries.loc[:, ('reactive_power', 'source', 'source')].values.reshape(1, len(scenario_data.timesteps))
            * -1.0 * timestep_interval_hours  # In Wh.
            @ sp.block_diag([np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])] * len(scenario_data.timesteps)),
            dict(name='der_reactive_power_vector', timestep=scenario_data.timesteps, der_index=linear_electric_grid_model.electric_grid_model.ders)
        # ), (
        #     price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values
        #     * timestep_interval_hours,  # In Wh.
        #     ('variable', 1.0, dict(name='loss_active', timestep=scenario_data.timesteps)),
        ), ],
        # variables_quadractic=[(
        #     price_data.price_sensitivity_coefficient
        #     * timestep_interval_hours,  # In Wh.
        #     der_model.mapping_active_power_by_output.values,
        #     dict(name='output_vector', timestep=der_model.timesteps)
        # ),],
        constant=0.0
    )

    # Aggregate DER models.
    der_model_set.storage_states = (
        pd.MultiIndex.from_tuples([
            (der_name, state)
            for der_name in der_model_set.flexible_der_names
            for state in der_model_set.flexible_der_models[der_name].storage_states
        ], names=['der_name', 'state'])
        if len(der_model_set.flexible_der_names) > 0 else pd.Index([])
    )
    der_model_set.state_matrix = (
        sp.block_diag([
            der_model_set.flexible_der_models[der_name].state_matrix.values
            for der_name in der_model_set.flexible_der_names
        ])
    )
    der_model_set.control_matrix = (
        sp.block_diag([
            der_model_set.flexible_der_models[der_name].control_matrix.values
            for der_name in der_model_set.flexible_der_names
        ])
    )
    der_model_set.disturbance_matrix = (
        sp.block_diag([
            der_model_set.flexible_der_models[der_name].disturbance_matrix.values
            for der_name in der_model_set.flexible_der_names
        ])
    )
    der_model_set.state_output_matrix = (
        sp.block_diag([
            der_model_set.flexible_der_models[der_name].state_output_matrix.values
            for der_name in der_model_set.flexible_der_names
        ])
    )
    der_model_set.control_output_matrix = (
        sp.block_diag([
            der_model_set.flexible_der_models[der_name].control_output_matrix.values
            for der_name in der_model_set.flexible_der_names
        ])
    )
    der_model_set.disturbance_output_matrix = (
        sp.block_diag([
            der_model_set.flexible_der_models[der_name].disturbance_output_matrix.values
            for der_name in der_model_set.flexible_der_names
        ])
    )
    der_model_set.mapping_active_power_by_output = (
        sp.block_diag([
            der_model_set.flexible_der_models[der_name].mapping_active_power_by_output.values
            / (
                der_model_set.flexible_der_models[der_name].active_power_nominal
                if der_model_set.flexible_der_models[der_name].active_power_nominal != 0.0
                else 1.0
            )
            if der_model_set.flexible_der_models[der_name].is_electric_grid_connected
            else 0.0 * der_model_set.flexible_der_models[der_name].mapping_active_power_by_output.values
            for der_name in der_model_set.flexible_der_names
        ])
    )
    der_model_set.mapping_reactive_power_by_output = (
        sp.block_diag([
            der_model_set.flexible_der_models[der_name].mapping_reactive_power_by_output.values
            / (
                der_model_set.flexible_der_models[der_name].reactive_power_nominal
                if der_model_set.flexible_der_models[der_name].reactive_power_nominal != 0.0
                else 1.0
            )
            if der_model_set.flexible_der_models[der_name].is_electric_grid_connected
            else 0.0 * der_model_set.flexible_der_models[der_name].mapping_reactive_power_by_output.values
            for der_name in der_model_set.flexible_der_names
        ])
    )
    der_model_set.state_vector_initial = (
        np.concatenate([
            der_model_set.flexible_der_models[der_name].state_vector_initial.values
            for der_name in der_model_set.flexible_der_names
        ])
    )
    der_model_set.disturbance_timeseries = (
        pd.concat([
            der_model_set.flexible_der_models[der_name].disturbance_timeseries
            for der_name in der_model_set.flexible_der_names
        ], axis='columns')
    )
    der_model_set.output_minimum_timeseries = (
        pd.concat([
            der_model_set.flexible_der_models[der_name].output_minimum_timeseries
            for der_name in der_model_set.flexible_der_names
        ], axis='columns')
    )
    der_model_set.output_maximum_timeseries = (
        pd.concat([
            der_model_set.flexible_der_models[der_name].output_maximum_timeseries
            for der_name in der_model_set.flexible_der_names
        ], axis='columns')
    )
    der_model_set.electric_grid_der_index = (
        np.concatenate([
            der_model_set.flexible_der_models[der_name].electric_grid_der_index
            for der_name in der_model_set.flexible_der_names
        ])
    )

    # Define variables.
    standard_form.define_variable('state_vector', timestep=der_model_set.timesteps, state=der_model_set.states)
    standard_form.define_variable('control_vector', timestep=der_model_set.timesteps, control=der_model_set.controls)
    standard_form.define_variable('output_vector', timestep=der_model_set.timesteps, output=der_model_set.outputs)

    # Define constraints.

    # Initial state.
    # - For states which represent storage state of charge, initial state of charge is final state of charge.
    if any(~der_model_set.states.isin(der_model_set.storage_states)):
        standard_form.define_constraint(
            ('constant', der_model_set.state_vector_initial[~der_model_set.states.isin(der_model_set.storage_states)]),
            '==',
            ('variable', 1.0, dict(
                name='state_vector', timestep=der_model_set.timesteps[0],
                state=der_model_set.states[~der_model_set.states.isin(der_model_set.storage_states)]
            ))
        )
    # - For other states, set initial state according to the initial state vector.
    if any(der_model_set.states.isin(der_model_set.storage_states)):
        standard_form.define_constraint(
            ('variable', 1.0, dict(
                name='state_vector', timestep=der_model_set.timesteps[0],
                state=der_model_set.states[der_model_set.states.isin(der_model_set.storage_states)]
            )),
            '==',
            ('variable', 1.0, dict(
                name='state_vector', timestep=der_model_set.timesteps[-1],
                state=der_model_set.states[der_model_set.states.isin(der_model_set.storage_states)]
            ))
        )

    # State equation.
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='state_vector', timestep=der_model_set.timesteps[1:])),
        '==',
        ('variable', der_model_set.state_matrix, dict(name='state_vector', timestep=der_model_set.timesteps[:-1])),
        ('variable', der_model_set.control_matrix, dict(name='control_vector', timestep=der_model_set.timesteps[:-1])),
        ('constant', (der_model_set.disturbance_matrix @ der_model_set.disturbance_timeseries.iloc[:-1, :].T.values).T.ravel()),
        broadcast='timestep'
    )

    # Output equation.
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='output_vector', timestep=der_model_set.timesteps)),
        '==',
        ('variable', der_model_set.state_output_matrix, dict(name='state_vector', timestep=der_model_set.timesteps)),
        ('variable', der_model_set.control_output_matrix, dict(name='control_vector', timestep=der_model_set.timesteps)),
        ('constant', (der_model_set.disturbance_output_matrix @ der_model_set.disturbance_timeseries.T.values).T.ravel()),
        broadcast='timestep'
    )

    # Output limits.
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='output_vector', timestep=der_model_set.timesteps)),
        '>=',
        ('constant', der_model_set.output_minimum_timeseries.values.ravel()),
        broadcast='timestep'
    )
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='output_vector', timestep=der_model_set.timesteps)),
        '<=',
        ('constant', der_model_set.output_maximum_timeseries.values.ravel()),
        broadcast='timestep'
    )

    # Define connection constraints.
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='der_active_power_vector', timestep=scenario_data.timesteps, der_index=electric_grid_model.ders[der_model_set.electric_grid_der_index])),
        '==',
        ('variable', der_model_set.mapping_active_power_by_output, dict(name='output_vector', timestep=der_model_set.timesteps)),
        ('constant', np.zeros(len(der_model_set.timesteps) * len(der_model_set.electric_grid_der_index))),
        broadcast='timestep'
    )
    standard_form.define_constraint(
        ('variable', 1.0, dict(name='der_reactive_power_vector', timestep=scenario_data.timesteps, der_index=electric_grid_model.ders[der_model_set.electric_grid_der_index])),
        '==',
        ('variable', der_model_set.mapping_reactive_power_by_output, dict(name='output_vector', timestep=der_model_set.timesteps)),
        ('constant', np.zeros(len(der_model_set.timesteps) * len(der_model_set.electric_grid_der_index))),
        broadcast='timestep'
    )

    # # Define problem for DER models.
    # for der_name, der_model in der_model_set.flexible_der_models.items():
    #
    #     # Define variables.
    #     standard_form.define_variable('state_vector', timestep=der_model.timesteps, state=der_model.states, der_name=[der_name])
    #     standard_form.define_variable('control_vector', timestep=der_model.timesteps, control=der_model.controls, der_name=[der_name])
    #     standard_form.define_variable('output_vector', timestep=der_model.timesteps, output=der_model.outputs, der_name=[der_name])
    #
    #     # Define constraints.
    #
    #     # Initial state.
    #     # - For states which represent storage state of charge, initial state of charge is final state of charge.
    #     if any(~der_model.states.isin(der_model.storage_states)):
    #         standard_form.define_constraint(
    #             ('constant', der_model.state_vector_initial.values[~der_model.states.isin(der_model.storage_states)]),
    #             '==',
    #             ('variable', 1.0, dict(
    #                 name='state_vector', timestep=der_model.timesteps[0],
    #                 state=der_model.states[~der_model.states.isin(der_model.storage_states)], der_name=der_name
    #             ))
    #         )
    #     # - For other states, set initial state according to the initial state vector.
    #     if any(der_model.states.isin(der_model.storage_states)):
    #         standard_form.define_constraint(
    #             ('variable', 1.0, dict(
    #                 name='state_vector', timestep=der_model.timesteps[0],
    #                 state=der_model.states[der_model.states.isin(der_model.storage_states)], der_name=der_name
    #             )),
    #             '==',
    #             ('variable', 1.0, dict(
    #                 name='state_vector', timestep=der_model.timesteps[-1],
    #                 state=der_model.states[der_model.states.isin(der_model.storage_states)], der_name=der_name
    #             ))
    #         )
    #
    #     # State equation.
    #     standard_form.define_constraint(
    #         ('variable', 1.0, dict(name='state_vector', timestep=der_model.timesteps[1:], der_name=der_name)),
    #         '==',
    #         ('variable', der_model.state_matrix.values, dict(name='state_vector', timestep=der_model.timesteps[:-1], der_name=der_name)),
    #         ('variable', der_model.control_matrix.values, dict(name='control_vector', timestep=der_model.timesteps[:-1], der_name=der_name)),
    #         ('constant', (der_model.disturbance_matrix.values @ der_model.disturbance_timeseries.iloc[:-1, :].T.values).T.ravel()),
    #         keys=dict(name='state_equation', timestep=der_model.timesteps[1:], state=der_model.states, der_name=der_name),
    #         broadcast='timestep'
    #     )
    #
    #     # Output equation.
    #     standard_form.define_constraint(
    #         ('variable', 1.0, dict(name='output_vector', timestep=der_model.timesteps, der_name=der_name)),
    #         '==',
    #         ('variable', der_model.state_output_matrix.values, dict(name='state_vector', timestep=der_model.timesteps, der_name=der_name)),
    #         ('variable', der_model.control_output_matrix.values, dict(name='control_vector', timestep=der_model.timesteps, der_name=der_name)),
    #         ('constant', (der_model.disturbance_output_matrix.values @ der_model.disturbance_timeseries.T.values).T.ravel()),
    #         broadcast='timestep'
    #     )
    #
    #     # Output limits.
    #     standard_form.define_constraint(
    #         ('variable', 1.0, dict(name='output_vector', timestep=der_model.timesteps, der_name=der_name)),
    #         '>=',
    #         ('constant', der_model.output_minimum_timeseries.values.ravel()),
    #         keys=dict(name='output_minimum', timestep=der_model.timesteps, output=der_model.outputs, der_name=der_name),
    #         broadcast='timestep'
    #     )
    #     standard_form.define_constraint(
    #         ('variable', 1.0, dict(name='output_vector', timestep=der_model.timesteps, der_name=der_name)),
    #         '<=',
    #         ('constant', der_model.output_maximum_timeseries.values.ravel()),
    #         keys=dict(name='output_maximum', timestep=der_model.timesteps, output=der_model.outputs, der_name=der_name),
    #         broadcast='timestep'
    #     )
    #
    #     # Define connection constraints.
    #     # TODO: Define grid connection constraints!
    #     if der_model.is_electric_grid_connected:
    #         connection_constraint_active = (
    #             (der_model.mapping_active_power_by_output.values / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0))
    #         )
    #         standard_form.define_constraint(
    #             ('variable', 1.0, dict(name='der_active_power_vector', timestep=scenario_data.timesteps, der_index=electric_grid_model.ders[der_model.electric_grid_der_index])),
    #             ('variable', (-1) * connection_constraint_active, dict(name='output_vector', timestep=der_model.timesteps, der_name=der_name)),
    #             '==',
    #             ('constant', np.zeros(len(scenario_data.timesteps))),
    #             broadcast='timestep'
    #         )
    #         connection_constraint_reactive = (
    #             (der_model.mapping_reactive_power_by_output.values / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0))
    #         )
    #         standard_form.define_constraint(
    #             ('variable', 1.0, dict(name='der_reactive_power_vector', timestep=scenario_data.timesteps, der_index=electric_grid_model.ders[der_model.electric_grid_der_index])),
    #             ('variable', (-1) * connection_constraint_reactive, dict(name='output_vector', timestep=der_model.timesteps, der_name=der_name)),
    #             '==',
    #             ('constant', np.zeros(len(scenario_data.timesteps))),
    #             broadcast='timestep'
    #         )
    #
    #     # # Obtain timestep interval in hours, for conversion of power to energy.
    #     # timestep_interval_hours = (der_model.timesteps[1] - der_model.timesteps[0]) / pd.Timedelta('1h')
    #     #
    #     # # Define objective.
    #     # # Active power cost / revenue.
    #     # # - Cost for load / demand, revenue for generation / supply.
    #     # standard_form.define_objective_low_level(
    #     #     variables=[(
    #     #         price_data.price_timeseries.loc[der_model.timesteps, ('active_power', slice(None), der_model.der_name)].T.values
    #     #         * -1.0 * timestep_interval_hours  # In Wh.
    #     #         @ sp.block_diag([der_model.mapping_active_power_by_output.values] * len(der_model.timesteps)),
    #     #         dict(name='output_vector', timestep=der_model.timesteps, der_name=der_name)
    #     #     ),],
    #     #     # variables_quadractic=[(
    #     #     #     price_data.price_sensitivity_coefficient
    #     #     #     * timestep_interval_hours,  # In Wh.
    #     #     #     der_model.mapping_active_power_by_output.values,
    #     #     #     dict(name='output_vector', timestep=der_model.timesteps)
    #     #     # ),],
    #     #     constant=0.0
    #     # )

    fledge.utils.log_time('standard-form problem')

    # Solve optimization problem.
    fledge.utils.log_time('standard-form solve')
    standard_form.solve()
    fledge.utils.log_time('standard-form solve')

    # Obtain results.
    results_1 = standard_form.get_results()
    duals_1 = standard_form.get_duals()
    fledge.utils.log_time('standard-form interface')

    # Instantiate optimization problem.
    fledge.utils.log_time('cvxpy interface')
    fledge.utils.log_time('cvxpy problem')
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define electric grid variables.
    optimization_problem.der_active_power_vector = (
        cp.Variable(
            (len(linear_electric_grid_model.electric_grid_model.timesteps), 
             len(linear_electric_grid_model.electric_grid_model.ders)))
    )
    optimization_problem.der_reactive_power_vector = (
        cp.Variable(
            (len(linear_electric_grid_model.electric_grid_model.timesteps), 
             len(linear_electric_grid_model.electric_grid_model.ders)))
    )
    optimization_problem.node_voltage_magnitude_vector = (
        cp.Variable(
            (len(linear_electric_grid_model.electric_grid_model.timesteps), 
             len(linear_electric_grid_model.electric_grid_model.nodes)))
    )
    optimization_problem.branch_power_magnitude_vector_1 = (
        cp.Variable(
            (len(linear_electric_grid_model.electric_grid_model.timesteps), 
             len(linear_electric_grid_model.electric_grid_model.branches)))
    )
    optimization_problem.branch_power_magnitude_vector_2 = (
        cp.Variable(
            (len(linear_electric_grid_model.electric_grid_model.timesteps), 
             len(linear_electric_grid_model.electric_grid_model.branches)))
    )

    # Define loss variables.
    optimization_problem.loss_active = cp.Variable((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))
    optimization_problem.loss_reactive = cp.Variable((len(linear_electric_grid_model.electric_grid_model.timesteps), 1))

    # Define DER variables.
    optimization_problem.state_vector = {
        der_model.der_name: cp.Variable((len(der_model.timesteps), len(der_model.states)))
        for der_name, der_model in der_model_set.flexible_der_models.items()
    }
    optimization_problem.control_vector = {
        der_model.der_name: cp.Variable((len(der_model.timesteps), len(der_model.controls)))
        for der_name, der_model in der_model_set.flexible_der_models.items()
    }
    optimization_problem.output_vector = {
        der_model.der_name: cp.Variable((len(der_model.timesteps), len(der_model.outputs)))
        for der_name, der_model in der_model_set.flexible_der_models.items()
    }
    
    # Define grid problem.
    timestep_index = slice(None)
    # Define voltage equation.
    optimization_problem.constraints.append(
        optimization_problem.node_voltage_magnitude_vector[timestep_index, :]
        ==
        (
            cp.transpose(
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                @ cp.transpose(cp.multiply(
                    optimization_problem.der_active_power_vector[timestep_index, :],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                @ cp.transpose(cp.multiply(
                    optimization_problem.der_reactive_power_vector[timestep_index, :],
                    np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
            )
            + np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
        )
        / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
    )
    # Define branch flow equation.
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_1[timestep_index, :]
        ==
        (
            cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                @ cp.transpose(cp.multiply(
                    optimization_problem.der_active_power_vector[timestep_index, :],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
                + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                @ cp.transpose(cp.multiply(
                    optimization_problem.der_reactive_power_vector[timestep_index, :],
                    np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
            )
            + np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1.ravel())])
        )
        / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
    )
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_2[timestep_index, :]
        ==
        (
            cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                @ cp.transpose(cp.multiply(
                    optimization_problem.der_active_power_vector[timestep_index, :],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
                + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                @ cp.transpose(cp.multiply(
                    optimization_problem.der_reactive_power_vector[timestep_index, :],
                    np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                ) - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
            )
            + np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2.ravel())])
        )
        / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
    )

    # Define loss equation.
    optimization_problem.constraints.append(
        optimization_problem.loss_active[timestep_index, :]
        ==
        cp.transpose(
            linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
            @ cp.transpose(cp.multiply(
                optimization_problem.der_active_power_vector[timestep_index, :],
                np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
            ) - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
            + linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
            @ cp.transpose(cp.multiply(
                optimization_problem.der_reactive_power_vector[timestep_index, :],
                np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
            ) - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
        )
        + np.real(linear_electric_grid_model.power_flow_solution.loss)
    )
    optimization_problem.constraints.append(
        optimization_problem.loss_reactive[timestep_index, :]
        ==
        cp.transpose(
            linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
            @ cp.transpose(cp.multiply(
                optimization_problem.der_active_power_vector[timestep_index, :],
                np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
            ) - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
            + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
            @ cp.transpose(cp.multiply(
                optimization_problem.der_reactive_power_vector[timestep_index, :],
                np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
            ) - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())]))
        )
        + np.imag(linear_electric_grid_model.power_flow_solution.loss)
    )

    # Define voltage limits.
    optimization_problem.voltage_magnitude_vector_minimum_constraint = (
            optimization_problem.node_voltage_magnitude_vector
            - np.array([node_voltage_magnitude_vector_minimum.ravel()])
            / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
            >=
            0.0
    )
    optimization_problem.voltage_magnitude_vector_maximum_constraint = (
            optimization_problem.node_voltage_magnitude_vector
            - np.array([node_voltage_magnitude_vector_maximum.ravel()])
            / np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])
            <=
            0.0
    )
    optimization_problem.constraints.append(optimization_problem.voltage_magnitude_vector_maximum_constraint)

    # Define branch flow limits.
    optimization_problem.branch_power_magnitude_vector_1_minimum_constraint = (
            optimization_problem.branch_power_magnitude_vector_1
            + np.array([branch_power_magnitude_vector_maximum.ravel()])
            / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
            >=
            0.0
    )
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_1_minimum_constraint
    )
    optimization_problem.branch_power_magnitude_vector_1_maximum_constraint = (
            optimization_problem.branch_power_magnitude_vector_1
            - np.array([branch_power_magnitude_vector_maximum.ravel()])
            / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
            <=
            0.0
    )
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_1_maximum_constraint
    )
    optimization_problem.branch_power_magnitude_vector_2_minimum_constraint = (
            optimization_problem.branch_power_magnitude_vector_2
            + np.array([branch_power_magnitude_vector_maximum.ravel()])
            / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
            >=
            0.0
    )
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_2_minimum_constraint
    )
    optimization_problem.branch_power_magnitude_vector_2_maximum_constraint = (
            optimization_problem.branch_power_magnitude_vector_2
            - np.array([branch_power_magnitude_vector_maximum.ravel()])
            / np.array([linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference])
            <=
            0.0
    )
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_2_maximum_constraint
    )

    # Define DER problem.
    for der_name, der_model in der_model_set.flexible_der_models.items():

        # Define constraints.
        # Initial state.
        # - For states which represent storage state of charge, initial state of charge is final state of charge.
        if any(der_model.states.isin(der_model.storage_states)):
            optimization_problem.constraints.append(
                optimization_problem.state_vector[der_model.der_name][0, der_model.states.isin(der_model.storage_states)]
                ==
                optimization_problem.state_vector[der_model.der_name][-1, der_model.states.isin(der_model.storage_states)]
            )
        # - For other states, set initial state according to the initial state vector.
        if any(~der_model.states.isin(der_model.storage_states)):
            optimization_problem.constraints.append(
                optimization_problem.state_vector[der_model.der_name][0, ~der_model.states.isin(der_model.storage_states)]
                ==
                der_model.state_vector_initial.loc[~der_model.states.isin(der_model.storage_states)].values
            )

        # State equation.
        optimization_problem.constraints.append(
            optimization_problem.state_vector[der_model.der_name][1:, :]
            ==
            cp.transpose(
                der_model.state_matrix.values
                @ cp.transpose(optimization_problem.state_vector[der_model.der_name][:-1, :])
                + der_model.control_matrix.values
                @ cp.transpose(optimization_problem.control_vector[der_model.der_name][:-1, :])
                + der_model.disturbance_matrix.values
                @ np.transpose(der_model.disturbance_timeseries.iloc[:-1, :].values)
            )
        )

        # Output equation.
        optimization_problem.constraints.append(
            optimization_problem.output_vector[der_model.der_name]
            ==
            cp.transpose(
                der_model.state_output_matrix.values
                @ cp.transpose(optimization_problem.state_vector[der_model.der_name])
                + der_model.control_output_matrix.values
                @ cp.transpose(optimization_problem.control_vector[der_model.der_name])
                + der_model.disturbance_output_matrix.values
                @ np.transpose(der_model.disturbance_timeseries.values)
            )
        )

        # Output limits.
        outputs_minimum_infinite = (
            (der_model.output_minimum_timeseries == -np.inf).all()
        )
        optimization_problem.constraints.append(
            optimization_problem.output_vector[der_model.der_name][:, ~outputs_minimum_infinite]
            >=
            der_model.output_minimum_timeseries.loc[:, ~outputs_minimum_infinite].values
        )
        outputs_maximum_infinite = (
            (der_model.output_maximum_timeseries == np.inf).all()
        )
        optimization_problem.constraints.append(
            optimization_problem.output_vector[der_model.der_name][:, ~outputs_maximum_infinite]
            <=
            der_model.output_maximum_timeseries.loc[:, ~outputs_maximum_infinite].values
        )

        # Define connection constraints.
        optimization_problem.constraints.append(
            optimization_problem.der_active_power_vector[:, der_model.electric_grid_der_index]
            ==
            cp.transpose(
                der_model.mapping_active_power_by_output.values
                @ cp.transpose(optimization_problem.output_vector[der_model.der_name])
            )
            / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)
        )
        optimization_problem.constraints.append(
            optimization_problem.der_reactive_power_vector[:, der_model.electric_grid_der_index]
            ==
            cp.transpose(
                der_model.mapping_reactive_power_by_output.values
                @ cp.transpose(optimization_problem.output_vector[der_model.der_name])
            )
            / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)
        )

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (der_model.timesteps[1] - der_model.timesteps[0]) / pd.Timedelta('1h')

        # Define objective.
        # Active power cost / revenue.
        # - Cost for load / demand, revenue for generation / supply.
        # optimization_problem.objective += (
        #     (
        #         price_data.price_timeseries.loc[:, ('active_power', slice(None), der_model.der_name)].values.T
        #         * -1.0 * timestep_interval_hours  # In Wh.
        #         @ cp.transpose(
        #             der_model.mapping_active_power_by_output.values
        #             @ cp.transpose(optimization_problem.output_vector[der_model.der_name])
        #         )
        #     )
        #     # + ((
        #     #     price_data.price_sensitivity_coefficient
        #     #     * timestep_interval_hours  # In Wh.
        #     #     * cp.sum((
        #     #         der_model.mapping_active_power_by_output.values
        #     #         @ cp.transpose(optimization_problem.output_vector[der_model.der_name])
        #     #     ) ** 2)
        #     # ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
        # )
    
    # Define objective. 
    optimization_problem.objective += (
        (
            np.array([
                price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values[timestep_index]
            ])
            * timestep_interval_hours  # In Wh.
            @ cp.sum(-1.0 * (
                cp.multiply(
                    optimization_problem.der_active_power_vector[timestep_index, :],
                    np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                )
            ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
        )
        # + ((
        #     price_data.price_sensitivity_coefficient
        #     * timestep_interval_hours  # In Wh.
        #     * cp.sum((
        #         cp.multiply(
        #             optimization_problem.der_active_power_vector[timestep_index, :],
        #             np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
        #         )
        #     ) ** 2)
        # ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
    )

    # Reactive power cost / revenue.
    # - Cost for load / demand, revenue for generation / supply.
    optimization_problem.objective += (
        (
            np.array([
                price_data.price_timeseries.loc[:, ('reactive_power', 'source', 'source')].values[timestep_index]
            ])
            * timestep_interval_hours  # In Wh.
            @ cp.sum(-1.0 * (
                cp.multiply(
                    optimization_problem.der_reactive_power_vector[timestep_index, :],
                    np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
                )
            ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
        )
        # + ((
        #     price_data.price_sensitivity_coefficient
        #     * timestep_interval_hours  # In Wh.
        #     * cp.sum((
        #         cp.multiply(
        #             optimization_problem.der_reactive_power_vector[timestep_index, :],
        #             np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])
        #         )
        #     ) ** 2)  # Sum along DERs, i.e. sum for each timestep.
        # ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
    )

    # Define active loss cost.
    # optimization_problem.objective += (
    #     (
    #         np.array([
    #             price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values[timestep_index]
    #         ])
    #         * timestep_interval_hours  # In Wh.
    #         @ (
    #             optimization_problem.loss_active[timestep_index, :]
    #         )
    #     )
    #     + ((
    #         price_data.price_sensitivity_coefficient
    #         * timestep_interval_hours  # In Wh.
    #         * cp.sum((
    #             optimization_problem.loss_active[timestep_index, :]
    #         ) ** 2)
    #     ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
    # )
    fledge.utils.log_time('cvxpy problem')
    
    # Solve optimization problem.
    fledge.utils.log_time('cvxpy solve')
    optimization_problem.solve()
    fledge.utils.log_time('cvxpy solve')

    # Instantiate results variables.
    fledge.utils.log_time('cvxpy get results')
    state_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.states)
    control_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.controls)
    output_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)

    # Obtain results.
    for der_name in der_model_set.flexible_der_names:
        state_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem.state_vector[der_name].value
        )
        control_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem.control_vector[der_name].value
        )
        output_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem.output_vector[der_name].value
        )
    results_2 = (
        fledge.der_models.DERModelSetOperationResults(
            state_vector=state_vector,
            control_vector=control_vector,
            output_vector=output_vector
        )
    )
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
                y=results_1['output_vector'].loc[:, [(der_name, output)]].values.ravel(),
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


if __name__ == '__main__':
    main()
