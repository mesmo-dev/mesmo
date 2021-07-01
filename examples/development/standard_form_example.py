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
    fledge.data_interface.recreate_database()

    # Obtain data.
    der_data = fledge.data_interface.DERData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name, der_data)

    # Obtain model.
    der_model_set = fledge.der_models.DERModelSet(der_data)

    # Obtain standard form.
    fledge.utils.log_time('standard-form interface')
    standard_form = fledge.utils.StandardForm()

    # Define problem.
    fledge.utils.log_time('standard-form problem')
    for der_name, der_model in der_model_set.flexible_der_models.items():

        # Define variables.
        standard_form.define_variable('state_vector', timestep=der_model.timesteps, state=der_model.states, der_name=[der_name])
        standard_form.define_variable('control_vector', timestep=der_model.timesteps, control=der_model.controls, der_name=[der_name])
        standard_form.define_variable('output_vector', timestep=der_model.timesteps, output=der_model.outputs, der_name=[der_name])

        # Define constraints.

        # Initial state.
        # - For states which represent storage state of charge, initial state of charge is final state of charge.
        if any(~der_model.states.isin(der_model.storage_states)):
            standard_form.define_constraint(
                ('constant', der_model.state_vector_initial.values[~der_model.states.isin(der_model.storage_states)]),
                '==',
                ('variable', 1.0, dict(
                    name='state_vector', timestep=der_model.timesteps[0],
                    state=der_model.states[~der_model.states.isin(der_model.storage_states)], der_name=der_name
                ))
            )
        # - For other states, set initial state according to the initial state vector.
        if any(der_model.states.isin(der_model.storage_states)):
            standard_form.define_constraint(
                ('variable', 1.0, dict(
                    name='state_vector', timestep=der_model.timesteps[0],
                    state=der_model.states[der_model.states.isin(der_model.storage_states)], der_name=der_name
                )),
                '==',
                ('variable', 1.0, dict(
                    name='state_vector', timestep=der_model.timesteps[-1],
                    state=der_model.states[der_model.states.isin(der_model.storage_states)], der_name=der_name
                ))
            )

        # State equation.
        standard_form.define_constraint(
            ('variable', 1.0, dict(name='state_vector', timestep=der_model.timesteps[1:], der_name=der_name)),
            '==',
            ('variable', der_model.state_matrix.values, dict(name='state_vector', timestep=der_model.timesteps[:-1], der_name=der_name)),
            ('variable', der_model.control_matrix.values, dict(name='control_vector', timestep=der_model.timesteps[:-1], der_name=der_name)),
            ('constant', (der_model.disturbance_matrix.values @ der_model.disturbance_timeseries.iloc[:-1, :].T.values).T.ravel()),
            keys=dict(name='state_equation', timestep=der_model.timesteps[1:], state=der_model.states, der_name=der_name),
            broadcast='timestep'
        )

        # Output equation.
        standard_form.define_constraint(
            ('variable', 1.0, dict(name='output_vector', timestep=der_model.timesteps, der_name=der_name)),
            '==',
            ('variable', der_model.state_output_matrix.values, dict(name='state_vector', timestep=der_model.timesteps, der_name=der_name)),
            ('variable', der_model.control_output_matrix.values, dict(name='control_vector', timestep=der_model.timesteps, der_name=der_name)),
            ('constant', (der_model.disturbance_output_matrix.values @ der_model.disturbance_timeseries.T.values).T.ravel()),
            broadcast='timestep'
        )

        # Output limits.
        standard_form.define_constraint(
            ('variable', 1.0, dict(name='output_vector', timestep=der_model.timesteps, der_name=der_name)),
            '>=',
            ('constant', der_model.output_minimum_timeseries.values.ravel()),
            keys=dict(name='output_minimum', timestep=der_model.timesteps, output=der_model.outputs, der_name=der_name),
            broadcast='timestep'
        )
        standard_form.define_constraint(
            ('variable', 1.0, dict(name='output_vector', timestep=der_model.timesteps, der_name=der_name)),
            '<=',
            ('constant', der_model.output_maximum_timeseries.values.ravel()),
            keys=dict(name='output_maximum', timestep=der_model.timesteps, output=der_model.outputs, der_name=der_name),
            broadcast='timestep'
        )

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (der_model.timesteps[1] - der_model.timesteps[0]) / pd.Timedelta('1h')

        # Define objective.
        # Active power cost / revenue.
        # - Cost for load / demand, revenue for generation / supply.
        standard_form.define_objective_low_level(
            variables=[(
                price_data.price_timeseries.loc[der_model.timesteps, ('active_power', slice(None), der_model.der_name)].T.values
                * -1.0 * timestep_interval_hours  # In Wh.
                @ sp.block_diag([der_model.mapping_active_power_by_output.values] * len(der_model.timesteps)),
                dict(name='output_vector', timestep=der_model.timesteps, der_name=der_name)
            ),],
            # variables_quadractic=[(
            #     price_data.price_sensitivity_coefficient
            #     * timestep_interval_hours,  # In Wh.
            #     der_model.mapping_active_power_by_output.values,
            #     dict(name='output_vector', timestep=der_model.timesteps)
            # ),],
            constant=0.0
        )

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
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define variables.
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

    # Define problem.
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

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (der_model.timesteps[1] - der_model.timesteps[0]) / pd.Timedelta('1h')

        # Define objective.
        # Active power cost / revenue.
        # - Cost for load / demand, revenue for generation / supply.
        optimization_problem.objective += (
            (
                price_data.price_timeseries.loc[:, ('active_power', slice(None), der_model.der_name)].values.T
                * -1.0 * timestep_interval_hours  # In Wh.
                @ cp.transpose(
                    der_model.mapping_active_power_by_output.values
                    @ cp.transpose(optimization_problem.output_vector[der_model.der_name])
                )
            )
            # + ((
            #     price_data.price_sensitivity_coefficient
            #     * timestep_interval_hours  # In Wh.
            #     * cp.sum((
            #         der_model.mapping_active_power_by_output.values
            #         @ cp.transpose(optimization_problem.output_vector[der_model.der_name])
            #     ) ** 2)
            # ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
        )

    # Solve optimization problem.
    optimization_problem.solve()

    # Instantiate results variables.
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
                y=results_1['output_vector'].loc[:, (der_name, output)].values.ravel(),
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
