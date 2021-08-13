"""Example script for DRO problem."""

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse
import fledge
from dro_data_interface import DRO_data


def stage_1_problem_standard_form(scenario_name, dro_data_set):

    print('stage 1 problem modelling...')

    # Settings.
    stochastic_scenarios = ['no_reserve', 'up_reserve', 'down_reserve']

    # Obtain price data object.
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain DER & grid model objects.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    linear_electric_grid_model_set = fledge.electric_grid_models.LinearElectricGridModelSet(scenario_name)

    # Manipulate building model to avoid over-consumption for up-reserves.
    for der_name, der_model in der_model_set.der_models.items():
        if isinstance(der_model, fledge.der_models.FlexibleBuildingModel):
            der_model.output_maximum_timeseries.loc[
                :, der_model.output_maximum_timeseries.columns.str.contains('_heat_')
            ] = 0.0

    # Instantiate optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define DER problem.
    der_model_set.define_optimization_variables(optimization_problem, scenarios=stochastic_scenarios)
    der_model_set.define_optimization_parameters(optimization_problem, price_data, scenarios=stochastic_scenarios)
    der_model_set.define_optimization_constraints(optimization_problem, scenarios=stochastic_scenarios)

    # Define electric grid problem.
    linear_electric_grid_model_set.define_optimization_variables(optimization_problem, scenarios=stochastic_scenarios)
    linear_electric_grid_model_set.define_optimization_parameters(
        optimization_problem,
        price_data,
        scenarios=stochastic_scenarios
    )
    linear_electric_grid_model_set.define_optimization_constraints(optimization_problem, scenarios=stochastic_scenarios)

    # Define additional variables.
    optimization_problem.define_variable('energy', timestep=linear_electric_grid_model_set.electric_grid_model.timesteps)
    optimization_problem.define_variable('up_reserve', timestep=linear_electric_grid_model_set.electric_grid_model.timesteps)
    optimization_problem.define_variable('down_reserve', timestep=linear_electric_grid_model_set.electric_grid_model.timesteps)

    optimization_problem.define_constraint(
        ('variable', 1.0, dict(name='up_reserve')),
        '>=',
        ('constant', 0.0)
    )
    optimization_problem.define_constraint(
        ('variable', 1.0, dict(name='down_reserve')),
        '>=',
        ('constant', 0.0)
    )

    # Define power balance constraints.
    optimization_problem.define_constraint(
        ('variable', 1.0, dict(name='energy', timestep=der_model_set.timesteps)),
        '==',
        ('variable', (
            -1.0 * np.array([np.real(linear_electric_grid_model_set.electric_grid_model.der_power_vector_reference)])
        ), dict(
            name='der_active_power_vector', scenario='no_reserve', timestep=der_model_set.timesteps,
            der=der_model_set.ders
        )),
        broadcast='timestep'
    )
    optimization_problem.define_constraint(
        ('variable', 1.0, dict(name='energy', timestep=der_model_set.timesteps)),
        ('variable', 1.0, dict(name='up_reserve', timestep=der_model_set.timesteps)),
        '==',
        ('variable', (
            -1.0 * np.array([np.real(linear_electric_grid_model_set.electric_grid_model.der_power_vector_reference)])
        ), dict(
            name='der_active_power_vector', scenario='up_reserve', timestep=der_model_set.timesteps,
            der=der_model_set.ders
        )),
        broadcast='timestep'
    )
    optimization_problem.define_constraint(
        ('variable', 1.0, dict(name='energy', timestep=der_model_set.timesteps)),
        ('variable', -1.0, dict(name='down_reserve', timestep=der_model_set.timesteps)),
        '==',
        ('variable', (
            -1.0 * np.array([np.real(linear_electric_grid_model_set.electric_grid_model.der_power_vector_reference)])
        ), dict(
            name='der_active_power_vector', scenario='down_reserve', timestep=der_model_set.timesteps,
            der=der_model_set.ders
        )),
        broadcast='timestep'
    )

    # Obtain timestep interval in hours, for conversion of power to energy.
    timestep_interval_hours = (der_model_set.timesteps[1] - der_model_set.timesteps[0]) / pd.Timedelta('1h')

    # Obtain energy price timeseries.
    # TODO: Make proper price input data.
    number_of_price = price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].shape
    price_timeseries_energy = (
        dro_data_set.energy_price[0:number_of_price[0]].to_numpy() * timestep_interval_hours
    )
    price_timeseries_reserve = (
        dro_data_set.contingency_reserve_price[0:number_of_price[0]].to_numpy() * timestep_interval_hours
    )

    # Define objective.
    # Active power cost / revenue.
    # - Cost for load / demand, revenue for generation / supply.
    optimization_problem.define_objective(
        ('variable', np.array([price_timeseries_energy]), dict(name='energy', timestep=der_model_set.timesteps)),
        ('variable', (
            -1.0 * dro_data_set.dro_base_data['prob_up_reserve_bidded'].values * np.array([price_timeseries_energy])
        ), dict(name='up_reserve', timestep=der_model_set.timesteps)),
        ('variable', (
            -1.0 * dro_data_set.dro_base_data['prob_down_reserve_bidded'].values * np.array([price_timeseries_reserve])
        ), dict(name='up_reserve', timestep=der_model_set.timesteps))
    )

    # Obtain standard form matrix / vector representation.
    a_matrix = optimization_problem.get_a_matrix()
    b_vector = optimization_problem.get_b_vector()
    # TODO: Relabel f_vector to c_vector.
    f_vector = -1.0 * optimization_problem.get_c_vector()

    return optimization_problem, a_matrix, b_vector, f_vector, stochastic_scenarios, der_model_set


def main():

    # Settings.
    scenario_name = 'singapore_6node_custom'
    fledge.data_interface.recreate_database()

    # Obtain data.
    price_data = fledge.data_interface.PriceData(scenario_name)
    dro_data_set = DRO_data(os.path.join(fledge.config.base_path, '..', 'primo_fledge', 'data', 'dro_data'))

    # Get results path.
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    (
        optimization_problem_stage_1, a_matrix, b_vector, f_vector, stochastic_scenarios, der_model_set
    ) = stage_1_problem_standard_form(scenario_name, dro_data_set)

    # Instantiate optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define optimization problem.
    optimization_problem.define_variable('x_vector', index=range(len(optimization_problem_stage_1.variables)))
    optimization_problem.define_constraint(
        ('variable', a_matrix, dict(name='x_vector', index=range(len(optimization_problem_stage_1.variables)))),
        '<=',
        ('constant', b_vector)
    )
    optimization_problem.define_objective(
        ('variable', -1.0 * f_vector.T, dict(name='x_vector', index=range(len(optimization_problem_stage_1.variables))))
    )

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = optimization_problem_stage_1.get_results(optimization_problem.x_vector)

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
                name='Maximum bound',
                line=go.scatter.Line(shape='hv')
            )
            figure.add_scatter(
                x=der_model.output_minimum_timeseries.index,
                y=der_model.output_minimum_timeseries.loc[:, output].values,
                name='Minimum bound',
                line=go.scatter.Line(shape='hv')
            )
            for number, stochastic_scenario in enumerate(stochastic_scenarios):
                if number == 0:
                    figure.add_scatter(
                        x=output_vector[stochastic_scenario].index,
                        y=output_vector[stochastic_scenario].loc[:, (der_name, output)].values,
                        name=f'optimal value in {stochastic_scenario} scenario',
                        line=go.scatter.Line(shape='hv', width=number + 5)
                    )
                elif number == 1:
                    figure.add_scatter(
                        x=output_vector[stochastic_scenario].index,
                        y=output_vector[stochastic_scenario].loc[:, (der_name, output)].values,
                        name=f'optimal value in {stochastic_scenario} scenario',
                        line=go.scatter.Line(shape='hv', width=number + 4, dash='dashdot')
                    )
                else:
                    figure.add_scatter(
                        x=output_vector[stochastic_scenario].index,
                        y=output_vector[stochastic_scenario].loc[:, (der_name, output)].values,
                        name=f'optimal value in {stochastic_scenario} scenario',
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
