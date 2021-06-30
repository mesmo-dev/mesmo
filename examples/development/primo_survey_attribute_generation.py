"""Script for generating attribute values for the choice experiment in the PRIMO survey."""

import cvxpy as cp
import numpy as np
from multimethod import multimethod
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge


def main():

    # Settings.
    scenario_name = 'primo_survey'
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data.
    der_data = fledge.data_interface.DERData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)
    # price_data_constant = price_data.copy()
    # price_data_constant.price_timeseries.loc[:, :] = price_data_constant.price_timeseries.mean().values[None, :]

    # Obtain building models.
    building_fixed = fledge.der_models.FlexibleBuildingModel(der_data, 'flexible_building')
    outputs_temperature = building_fixed.outputs.str.contains('temperature')
    outputs_heat = building_fixed.outputs.str.contains('_heat_')
    building_fixed.output_minimum_timeseries.loc[:, outputs_temperature] = (
        building_fixed.output_maximum_timeseries.loc[:, outputs_temperature].values - 0.01
    )
    building_fixed.output_maximum_timeseries.loc[:, outputs_heat] = 0.0
    building_smart = fledge.der_models.FlexibleBuildingModel(der_data, 'flexible_building')
    timesteps_nonsmart = building_smart.timesteps.hour > 13
    building_smart.output_minimum_timeseries.loc[timesteps_nonsmart, outputs_temperature] = (
        building_smart.output_maximum_timeseries.loc[timesteps_nonsmart, outputs_temperature].values - 0.01
    )
    building_smart.output_maximum_timeseries.loc[:, outputs_heat] = 0.0
    building_flexi = fledge.der_models.FlexibleBuildingModel(der_data, 'flexible_building')
    building_flexi.output_maximum_timeseries.loc[:, outputs_heat] = 0.0

    # Obtain EV charger models.
    ev_fixed = fledge.der_models.FlexibleEVChargerModel(der_data, 'flexible_ev_charger')
    timesteps_urgent_depart = ev_fixed.timesteps.hour > 10
    ev_fixed.output_maximum_timeseries.loc[timesteps_urgent_depart, 'active_power_charge'] = 0.0
    ev_smart = fledge.der_models.FlexibleEVChargerModel(der_data, 'flexible_ev_charger')
    ev_flexi = fledge.der_models.FlexibleEVChargerModel(der_data, 'flexible_ev_charger')
    ev_flexi.output_maximum_timeseries.loc[:, 'active_power_discharge'] = (
        ev_flexi.output_maximum_timeseries.loc[:, 'active_power_charge'].values
    )

    # Obtain solutions.
    results = {
        'building_fixed': solve_problem(building_fixed, price_data),
        'building_smart': solve_problem(building_smart, price_data),
        'building_flexi': solve_problem(building_flexi, price_data),
        'ev_fixed': solve_problem(ev_fixed, price_data),
        'ev_smart': solve_problem(ev_smart, price_data),
        'ev_flexi': solve_problem(ev_flexi, price_data)
    }

    # Save / plot results.
    save_results(results, results_path)
    plot_results(results, results_path)

    # Plot prices.
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


def solve_problem(
        flexible_der_model: fledge.der_models.FlexibleDERModel,
        price_data: fledge.data_interface.PriceData
) -> fledge.der_models.DERModelOperationResults:

    # Enforce storage states, initial state is linked to final state.
    flexible_der_model.storage_states = flexible_der_model.states

    # Instantiate optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define / solve optimization problem.
    flexible_der_model.define_optimization_variables(optimization_problem)
    flexible_der_model.define_optimization_constraints(optimization_problem)
    flexible_der_model.define_optimization_objective(optimization_problem, price_data)
    optimization_problem.solve()

    # Obtain results.
    results = flexible_der_model.get_optimization_results(optimization_problem)
    results.objective = optimization_problem.objective.value
    results.cost_timeseries = (
        -1.0 * (flexible_der_model.mapping_active_power_by_output @ results.output_vector.T).T
        * price_data.price_timeseries.loc[:, [('active_power', 'source', 'source')]].values
    )

    return results


def save_results(
        results: dict,
        results_path: str
):

    for label, result in results.items():

        # Create folder.
        try:
            os.mkdir(os.path.join(results_path, label))
        except Exception:
            pass

        result.save(os.path.join(results_path, label))


@multimethod
def plot_results(
        results: dict,
        results_path: str
):

    for label, result in results.items():
        plot_results(result, results_path, label)


@multimethod
def plot_results(
        results: fledge.der_models.DERModelOperationResults,
        results_path: str,
        label: str
):

    # Create folder.
    try:
        os.mkdir(os.path.join(results_path, label))
    except Exception:
        pass

    # Plot outputs.
    for output in results.der_model.outputs:

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=results.der_model.output_maximum_timeseries.index,
            y=results.der_model.output_maximum_timeseries.loc[:, output].values,
            name='Maximum',
            line=go.scatter.Line(shape='hv')
        ))
        figure.add_trace(go.Scatter(
            x=results.der_model.output_minimum_timeseries.index,
            y=results.der_model.output_minimum_timeseries.loc[:, output].values,
            name='Minimum',
            line=go.scatter.Line(shape='hv')
        ))
        figure.add_trace(go.Scatter(
            x=results['output_vector'].index,
            y=results['output_vector'].loc[:, output].values,
            name='Optimal',
            line=go.scatter.Line(shape='hv')
        ))
        figure.update_layout(
            title=f'Output: {output}',
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, label, output))

    # Plot disturbances.
    for disturbance in results.der_model.disturbances:

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=results.der_model.disturbance_timeseries.index,
            y=results.der_model.disturbance_timeseries.loc[:, disturbance].values,
            line=go.scatter.Line(shape='hv')
        ))
        figure.update_layout(
            title=f'Disturbance: {disturbance}',
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            showlegend=False
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, label, disturbance))


if __name__ == '__main__':
    main()
