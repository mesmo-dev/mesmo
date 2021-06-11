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

def stage_2_problem_standard_form():
    print('stage 2 problem modelling...')
    # Settings.
    scenario_name = 'singapore_6node_custom'
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
                'control_vector_s1', timestep=der_model.timesteps, der_name=[der_model.der_name], control=der_model.controls,
                scenario=[stochastic_scenario])
            standard_form.define_variable(
                'output_vector_s1', timestep=der_model.timesteps, der_name=[der_model.der_name], output=der_model.outputs,
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
            grid_node=linear_electric_grid_model.electric_grid_model.nodes,
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
        grid_node=linear_electric_grid_model.electric_grid_model.nodes,
    )

    # power balance equation.
    for timestep_grid, timestep_der in zip(linear_electric_grid_model.electric_grid_model.timesteps,
                                           der_model_set.timesteps):
        standard_form.define_constraint(
            (
                'variable', 1.0, dict(name='energy_s1', timestep=timestep_grid)
            ),
            (
                'variable', 1.0, dict(name='energy_deviation_s2', timestep=timestep_grid)
            ),
            '==',
            (
                'variable', -np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)]),
                 dict(name='der_active_power_vector_s2', timestep=timestep_der, der_model=der_model_set.ders)
            )
        )

    # constr 12 g)
    # V^ref * Vm - M^vp * p^ref * p^der - M^vq * q^ref * q^der = -(M^vp*p^* + M^vq*q^*) + v_power_flow
    active_reference_point_temp = np.array(
        [np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
    reactive_reference_point_temp = np.array(
        [np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])

    for timestep_grid, timestep_der in zip(linear_electric_grid_model.electric_grid_model.timesteps, der_model.timesteps):
        standard_form.define_constraint(
            (
                'variable', np.diagflat(
                np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])),
                dict(name='nodal_voltage_magnitude_s2', timestep=timestep_grid,
                     grid_node=linear_electric_grid_model.electric_grid_model.nodes)
            ),
            (
                'variable',
                 -linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active @ np.diagflat(
                 np.array([np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])),
                 dict(name='der_active_power_vector_s2', timestep=timestep_der, der_model=der_model_set.ders)
            ),
            (
                'variable',
                -linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive @ np.diagflat(
                np.array([np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference)])),
                dict(name='der_reactive_power_vector_s2', timestep=timestep_der, der_model=der_model_set.ders)
             ),
            '==',
            (
                'constant',
                (
                    -(linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                    @ active_reference_point_temp.T +
                    linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                    @ reactive_reference_point_temp.T) +
                    np.transpose(
                    np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
                    )
                )[:, 0]
            )
        )

    for der_name, der_model in der_model_set.der_models.items():
        # Fixed DERs.
        if issubclass(type(der_model), fledge.der_models.FixedDERModel):
            if der_model.is_electric_grid_connected:
                for timestep_der in der_model_set.timesteps:
                    standard_form.define_constraint(
                        (
                            'variable', 1.0, dict(
                                name='der_active_power_vector_s2', timestep=timestep_der,
                                der_model=(der_model.der_type, der_model.der_name)
                            )
                        ),
                        '==',
                        (
                            'constant',
                            (der_model.active_power_nominal_timeseries[timestep_der] / (
                             der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0)).tolist()
                        )
                    )

                    standard_form.define_constraint(
                        (
                            'variable', 1.0, dict(name='der_reactive_power_vector_s2', timestep=timestep_der,
                                                  der_model=(der_model.der_type, der_model.der_name))
                        ),
                        '==',
                        (
                            'constant',
                            (der_model.reactive_power_nominal_timeseries[timestep_der] / (
                             der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0)).tolist()
                        )
                    )

        # Flexible DERs.
        if issubclass(type(der_model), fledge.der_models.FlexibleDERModel):

            # Manipulate building model to avoid over-consumption for up-reserves.
            if issubclass(type(der_model), fledge.der_models.FlexibleBuildingModel):
                der_model.output_maximum_timeseries.loc[
                :, der_model.output_maximum_timeseries.columns.str.contains('_heat_')
                ] = 0.0

            #  Initial state.
            if any(~der_model.states.isin(der_model.storage_states)):
                standard_form.define_constraint(
                    ('constant', der_model.state_vector_initial.values[~der_model.states.isin(der_model.storage_states)]
                     ),
                    '==',
                    ('variable', 1.0, dict(
                        name='state_vector_s2', der_name=[der_model.der_name], timestep=der_model.timesteps[0],
                        state=der_model.states[~der_model.states.isin(der_model.storage_states)])
                     )
                )

            if any(der_model.states.isin(der_model.storage_states)):
                standard_form.define_constraint(
                    ('variable', 1.0, dict(
                        name='state_vector_s2', der_name=[der_model.der_name], timestep=der_model.timesteps[0],
                        state=der_model.states[der_model.states.isin(der_model.storage_states)]
                    )),
                    '==',
                    ('variable', 1.0, dict(
                        name='state_vector_s2', der_name=[der_model.der_name], timestep=der_model.timesteps[-1],
                        state=der_model.states[der_model.states.isin(der_model.storage_states)]
                    ))
                )

            # State equation.
            for timestep, timestep_previous in zip(der_model.timesteps[1:], der_model.timesteps[:-1]):
                if not der_model.disturbances.empty:
                    standard_form.define_constraint(
                        (
                            'variable', 1.0, dict(
                             name='state_vector_s2', der_name=[der_model.der_name], timestep=timestep,
                             state=der_model.states)
                        ),
                        '==',
                        (
                            'variable', der_model.state_matrix.values, dict(
                             name='state_vector_s2', der_name=[der_model.der_name], timestep=timestep_previous,
                             state=der_model.states
                            )
                        ),
                        (
                            'variable', der_model.control_matrix.values, dict(
                             name='control_vector_s2', der_name=[der_model.der_name], timestep=timestep_previous,
                             control=der_model.controls
                            )
                        ),
                        (
                            'constant',
                            der_model.disturbance_matrix.values @ der_model.disturbance_timeseries.loc[timestep, :].values
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
                            'variable', 1.0, dict(
                                 name='state_vector_s2', der_name=[der_model.der_name],
                                 timestep=timestep, state=der_model.states
                            )
                        ),
                        '==',
                        (
                            'variable', der_model.state_matrix.values, dict(
                                name='state_vector_s2', der_name=[der_model.der_name],
                                timestep=timestep_previous, state=der_model.states
                            )
                        ),
                        (
                            'variable', der_model.control_matrix.values, dict(
                                name='control_vector_s2', der_name=[der_model.der_name], timestep=timestep,
                                control=der_model.controls
                            )
                        ),
                        (
                            'constant', der_model.disturbance_matrix.values @ der_model.disturbance_timeseries.loc[timestep, :].values
                        )
                    )

            # Output equation.
            for timestep in der_model.timesteps:
                if not der_model.disturbances.empty:
                    standard_form.define_constraint(
                        (
                            'variable', 1.0, dict(
                                name='output_vector_s2', der_name=[der_model.der_name],
                                timestep=timestep, output=der_model.outputs
                            )
                        ),
                        '==',
                        (
                            'variable', der_model.state_output_matrix.values, dict(
                                name='state_vector_s2', der_name=[der_model.der_name], timestep=timestep,
                                state=der_model.states
                            )
                        ),
                        (
                            'variable', der_model.control_output_matrix.values, dict(
                                name='control_vector_s2', der_name=[der_model.der_name], timestep=timestep,
                                control=der_model.controls
                            )
                        ),
                        (
                            'constant',
                            der_model.disturbance_output_matrix.values @ der_model.disturbance_timeseries.loc[timestep, :].values
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
                            'variable', 1.0, dict(
                                name='output_vector_s2', der_name=[der_model.der_name], timestep=timestep,
                                output=der_model.outputs
                            )
                        ),
                        '==',
                        ('variable', der_model.state_output_matrix.values, dict(
                            name='state_vector_s2', der_name=[der_model.der_name], timestep=timestep,
                            state=der_model.states
                            )
                        ),
                        (
                            'variable', der_model.control_output_matrix.values, dict(
                                name='control_vector_s2', der_name=[der_model.der_name], timestep=timestep,
                                control=der_model.controls
                            )
                        ),
                        (
                            'constant',
                            der_model.disturbance_output_matrix.values @ der_model.disturbance_timeseries.loc[timestep, :].values
                        )
                    )

            # Define connection constraints.
            if der_model.is_electric_grid_connected:
                for timestep in der_model.timesteps:
                    # active power
                    standard_form.define_constraint(
                        (
                            'variable', der_model.mapping_active_power_by_output.values
                            / (der_model.active_power_nominal if der_model.active_power_nominal != 0.0 else 1.0),
                            dict(
                                name='output_vector_s2', der_name=[der_model.der_name], timestep=timestep,
                                output=der_model.outputs
                            )
                         ),
                        '==',
                        (
                            'variable', 1.0, dict(
                                name='der_active_power_vector_s2', timestep=timestep,
                                der_model=(der_model.der_type, der_model.der_name))
                        )
                    )

                    standard_form.define_constraint(
                        (
                            'variable', der_model.mapping_reactive_power_by_output.values
                            / (der_model.reactive_power_nominal if der_model.reactive_power_nominal != 0.0 else 1.0),
                            dict(
                                name='output_vector_s2', der_name=[der_model.der_name], timestep=timestep,
                                output=der_model.outputs
                            )
                         ),
                        '==',
                        (
                            'variable', 1.0, dict(
                                name='der_reactive_power_vector_s2', timestep=timestep,
                                der_model=(der_model.der_type, der_model.der_name))
                        )
                    )

            # Output limits.
            for timestep in der_model.timesteps:
                standard_form.define_constraint(
                    (
                        'variable', 1.0, dict(
                            name='output_vector_s2', der_name=[der_model.der_name], timestep=timestep,
                            output=der_model.outputs
                        )
                    ),
                    '>=',
                    (
                        'constant', der_model.output_minimum_timeseries.loc[timestep, :].values
                    )
                )

                standard_form.define_constraint(
                    (
                        'variable', 1.0, dict(
                            name='output_vector_s2', der_name=[der_model.der_name], timestep=timestep,
                            output=der_model.outputs
                        )
                    ),
                    '<=',
                    (
                        'constant', der_model.output_maximum_timeseries.loc[timestep, :].values
                    )
                )

    # voltage upper/lower bound
    node_voltage_magnitude_vector_minimum = (
            0.5 * np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
    )
    node_voltage_magnitude_vector_maximum = (
            1.5 * np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
    )


    for timestep_grid in linear_electric_grid_model.electric_grid_model.timesteps:
        standard_form.define_constraint(
            (
                'variable', np.diagflat(
                 np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])),
                 dict(
                     name='nodal_voltage_magnitude_s2', timestep=timestep_grid,
                     grid_node=linear_electric_grid_model.electric_grid_model.nodes
                 )
            ),
            '>=',
            (
                'constant', np.transpose(np.array([node_voltage_magnitude_vector_minimum.ravel()]))[:, 0]
            )
        )

        standard_form.define_constraint(
            (
                'variable', np.diagflat(
                 np.array([np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)])),
                 dict(
                     name='nodal_voltage_magnitude_s2', timestep=timestep_grid,
                     grid_node=linear_electric_grid_model.electric_grid_model.nodes
                 )
            ),
            '<=',
            (
                'constant', np.transpose(np.array([node_voltage_magnitude_vector_maximum.ravel()]))[:, 0]
            )
        )


    a_matrix = standard_form.get_a_matrix()
    b_vector = standard_form.get_b_vector()

    # Obtain DRO constr matrices: A2, B2, and C2 by slicing matrices
    # A2 matrix
    voltage_varibale_index = fledge.utils.get_index(
        standard_form.variables, name='nodal_voltage_magnitude_s1',
        timestep=der_model_set.timesteps, grid_node=linear_electric_grid_model.electric_grid_model.nodes,
        scenario=[stochastic_scenario]
    )

    s1_last_index = max(voltage_varibale_index)

    A2_matrix = a_matrix[:, 0:s1_last_index+1]

    # C2 matrix
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

    C2_matrix = a_matrix[:, delta_indices]

    # B2 matrix
    s1_indices = np.linspace(0, s1_last_index, s1_last_index+1)

    s2_indices = np.setdiff1d(
        np.linspace(0, a_matrix.shape[1]-1, a_matrix.shape[1]), np.hstack((s1_indices, delta_indices))
    )

    B2_matrix = a_matrix[:, s2_indices]


    # objective matrices
    # M_Q2_delta matrix
    M_Q2_delta = np.zeros((s1_indices.shape[0], delta_indices.shape[0]))

    energy_s1_indices = fledge.utils.get_index(
        standard_form.variables, name='energy_s1',
        timestep=linear_electric_grid_model.electric_grid_model.timesteps
    )

    up_reserve_s1_indices = fledge.utils.get_index(
        standard_form.variables, name='up_reserve_s1',
        timestep=linear_electric_grid_model.electric_grid_model.timesteps
    )

    down_reserve_s1_indices = fledge.utils.get_index(
        standard_form.variables, name='down_reserve_s1',
        timestep=linear_electric_grid_model.electric_grid_model.timesteps
    )

    energy_price_deviation_s2_indices = fledge.utils.get_index(
        standard_form.variables, name='uncertainty_energy_price_deviation_s2',
        timestep=linear_electric_grid_model.electric_grid_model.timesteps
    )

    up_reserve_price_deviation_s2_indices = fledge.utils.get_index(
        standard_form.variables, name='uncertainty_up_reserve_price_deviation_s2',
        timestep=linear_electric_grid_model.electric_grid_model.timesteps
    )

    down_reserve_price_deviation_s2_indices = fledge.utils.get_index(
        standard_form.variables, name='uncertainty_down_reserve_price_deviation_s2',
        timestep=linear_electric_grid_model.electric_grid_model.timesteps
    )

    M_Q2_delta[np.where(pd.Index(s1_indices).isin(up_reserve_s1_indices)),
               np.where(pd.Index(delta_indices).isin(up_reserve_price_deviation_s2_indices))] = 0.4

    M_Q2_delta[np.where(pd.Index(s1_indices).isin(down_reserve_s1_indices)),
               np.where(pd.Index(delta_indices).isin(down_reserve_price_deviation_s2_indices))] = 0.6

    M_Q2_delta[np.where(pd.Index(s1_indices).isin(energy_s1_indices)),
               np.where(pd.Index(delta_indices).isin(energy_price_deviation_s2_indices))] = -1

    # m_Q2_s2 vector
    m_Q2_s2 = np.zeros((s2_indices.shape[0], 1))
    penalty_factor = 0.1
    der_cost_factor = 0.01

    energy_deviation_s2_indices = fledge.utils.get_index(
        standard_form.variables, name='energy_deviation_s2',
        timestep=linear_electric_grid_model.electric_grid_model.timesteps
    )

    m_Q2_s2[np.where(pd.Index(s2_indices).isin(energy_deviation_s2_indices)), 0] = -penalty_factor

    der_active_power_vector_s2_indices = fledge.utils.get_index(
        standard_form.variables, name='der_active_power_vector_s2', timestep=der_model_set.timesteps,
        der_model=der_model_set.ders,
    )

    m_Q2_s2[np.where(pd.Index(s2_indices).isin(der_active_power_vector_s2_indices)), 0] = -der_cost_factor

    return standard_form, b_vector, A2_matrix, B2_matrix, C2_matrix, M_Q2_delta, m_Q2_s2, s2_indices, delta_indices, s1_indices


def main():
    scenario_name = 'singapore_6node_custom'
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Get results path.
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    standard_form_stage_1, a_matrix, b_vector, f_vector, stochastic_scenarios, der_model_set \
        = stage_1_problem_standard_form()
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


    standard_form_stage_2, b2_vector, A2_matrix, B2_matrix, C2_matrix, M_Q2_delta, m_Q2_s2, s2_indices_stage2, \
        delta_indices_stage2, s1_indices = stage_2_problem_standard_form()







if __name__ == '__main__':
    main()
