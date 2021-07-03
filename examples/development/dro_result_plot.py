"""Example script for DRO problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import fledge
import statistics



def main():
    scenario_name = 'singapore_6node_custom'

    energy_dro = pd.read_csv(
             "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\1-var-energy-price-DRO_microgrid_main_singapore_6node_custom_2021-07-02_23-13-22\\energy_dro.csv")

    energy_det = pd.read_csv(
        "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\1-var-energy-price-DRO_microgrid_main_singapore_6node_custom_2021-07-02_23-13-22\\energy_det.csv")

    up_reserve_det = pd.read_csv(
        "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\1-var-energy-price-DRO_microgrid_main_singapore_6node_custom_2021-07-02_23-13-22\\up_reserve_det.csv")

    up_reserve_dro = pd.read_csv(
        "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\1-var-energy-price-DRO_microgrid_main_singapore_6node_custom_2021-07-02_23-13-22\\up_reserve_dro.csv")

    down_reserve_det = pd.read_csv(
        "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\1-var-energy-price-DRO_microgrid_main_singapore_6node_custom_2021-07-02_23-13-22\\down_reserve_det.csv")

    down_reserve_dro = pd.read_csv(
        "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\1-var-energy-price-DRO_microgrid_main_singapore_6node_custom_2021-07-02_23-13-22\\down_reserve_dro.csv")

    print()

    number_of_time_steps = len(energy_dro['timestep'])
    X = np.arange(number_of_time_steps)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.bar(X + 0.00, energy_dro['total'], color='blue', width=0.25)
    ax.bar(X + 0.00, up_reserve_dro['total'], bottom=energy_dro['total'], color='r', width=0.25)
    ax.bar(X + 0.00, down_reserve_dro['total'], bottom=up_reserve_dro['total'], color='y', width=0.25)

    ax.bar(X + 0.25, energy_det['total'], color='royalblue', width=0.25)
    ax.bar(X + 0.25, up_reserve_det['total'], bottom=energy_det['total'], color='indianred', width=0.25)
    ax.bar(X + 0.25, down_reserve_det['total'], bottom=up_reserve_det['total'], color='gold', width=0.25)

    ax.legend(labels=['Energy - DRO', 'Up reserve - DRO', 'Down reserve - DRO', 'Energy - det.', 'Up reserve - det.', 'Down reserve - det.'])

    ax.set_ylabel('Offers kWh')
    ax.set_xlabel('time step')
    ax.set_title('Comparison of DRO solution and deterministic solution')
    ax.set_xticks(X)
    ax.set_xticklabels(energy_dro['timestep'].values.tolist())
    fig.savefig('C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\final_plots\\offer_result.pdf')
    plt.show()


    # # Get results path.
    # results_path = fledge.utils.get_results_path(__file__, scenario_name)
    #
    # # Recreate / overwrite database, to incorporate changes in the CSV definition files.
    # fledge.data_interface.recreate_database()
    #
    # dro_data_test = DRO_data("C:\\Users\\kai.zhang\\Desktop\\local_fledge_data\\dro_data\\")
    #
    # dro_price_deviation = dro_data_test.energy_price - dro_data_test.mean_energy_price

    # forecast_data_price = pd.read_csv(
    #     "C:\\Users\\kai.zhang\\Desktop\\local_fledge_data\\dro_data\\price_forecast_2021_26_07.csv")
    # # os.path.join('..', 'bla', 'dro')
    #
    # energy_price = forecast_data_price['USEP($/MWh)']
    #
    # contingency_price = forecast_data_price['Contingency($/MWh)']
    #
    # mean_energy_price = statistics.mean(energy_price)
    #
    # variance_energy_price = statistics.variance(energy_price, mean_energy_price)
    #
    # mean_contingency_price = statistics.mean(contingency_price)

    # variance_contingency_price = statistics.variance(contingency_price, mean_contingency_price)


    #
    # # Obtain price data object.
    # price_data = fledge.data_interface.PriceData(scenario_name)
    #
    # # Obtain DER & grid model objects.
    # der_model_set = fledge.der_models.DERModelSet(scenario_name)
    # # Getting linear electric grid model using "global approximation" method.
    # linear_electric_grid_model = fledge.electric_grid_models.LinearElectricGridModelGlobal(scenario_name)
    #
    # # Instantiate optimization problem.
    # optimization_problem = fledge.utils.OptimizationProblem()
    #
    # # Define model variables.
    # der_model_set.define_optimization_variables(optimization_problem)
    # linear_electric_grid_model.define_optimization_variables(optimization_problem)
    # # Define custom variable.
    # optimization_problem.electric_power_peak = cp.Variable(shape=(1, 1))
    #
    # # Define model constraints.
    # der_model_set.define_optimization_constraints(optimization_problem)
    # linear_electric_grid_model.define_optimization_constraints(optimization_problem)
    # # Define custom constraints.
    # for der_model in der_model_set.flexible_der_models.values():
    #     optimization_problem.constraints.append(
    #         optimization_problem.electric_power_peak[0, 0]
    #         >=
    #         optimization_problem.der_active_power_vector[:, der_model.electric_grid_der_index]
    #         * der_model.active_power_nominal  # Multiplying to convert power from per-unit to Watt values.
    #         * -1.0  # Multiplying to convert power from negative to positive value (loads are negative by default).
    #     )
    #
    # # Define model objective.
    # der_model_set.define_optimization_objective(optimization_problem, price_data)
    # linear_electric_grid_model.define_optimization_objective(optimization_problem, price_data)
    # # Define custom objective.
    # optimization_problem.objective += 1e2 * cp.sum(optimization_problem.electric_power_peak)
    #
    # # Solve optimization problem.
    # optimization_problem.solve()
    #
    # # Get model results.
    # results = fledge.problems.Results()
    # results.update(der_model_set.get_optimization_results(optimization_problem))
    # results.update(linear_electric_grid_model.get_optimization_results(optimization_problem))
    # # Get custom results.
    # electric_power_peak = optimization_problem.electric_power_peak.value
    # print(f"electric_power_peak = {electric_power_peak}")
    #
    # # Store results to CSV.
    # results.save(results_path)
    #
    # # Plot some results.
    # for der_model in der_model_set.flexible_der_models.values():
    #
    #     for output in der_model.outputs:
    #
    #         figure = go.Figure()
    #         figure.add_scatter(
    #             x=der_model.output_maximum_timeseries.index,
    #             y=der_model.output_maximum_timeseries.loc[:, output].values,
    #             name='Maximum',
    #             line=go.scatter.Line(shape='hv')
    #         )
    #         figure.add_scatter(
    #             x=der_model.output_minimum_timeseries.index,
    #             y=der_model.output_minimum_timeseries.loc[:, output].values,
    #             name='Minimum',
    #             line=go.scatter.Line(shape='hv')
    #         )
    #         figure.add_scatter(
    #             x=results.output_vector.index,
    #             y=results.output_vector.loc[:, (der_model.der_name, output)].values,
    #             name='Optimal',
    #             line=go.scatter.Line(shape='hv')
    #         )
    #         figure.update_layout(
    #             title=f'DER: {der_model.der_name} / Output: {output}',
    #             xaxis=go.layout.XAxis(tickformat='%H:%M'),
    #             legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
    #         )
    #         # figure.show()
    #         fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'der_{der_model.der_name}_out_{output}'))
    #
    #     for disturbance in der_model.disturbances:
    #
    #         figure = go.Figure()
    #         figure.add_scatter(
    #             x=der_model.disturbance_timeseries.index,
    #             y=der_model.disturbance_timeseries.loc[:, disturbance].values,
    #             line=go.scatter.Line(shape='hv')
    #         )
    #         figure.update_layout(
    #             title=f'DER: {der_model.der_name} / Disturbance: {disturbance}',
    #             xaxis=go.layout.XAxis(tickformat='%H:%M'),
    #             showlegend=False
    #         )
    #         # figure.show()
    #         fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'der_{der_model.der_name}_dis_{disturbance}'))
    #
    # for commodity_type in ['active_power', 'reactive_power']:
    #
    #     if commodity_type in price_data.price_timeseries.columns.get_level_values('commodity_type'):
    #         figure = go.Figure()
    #         figure.add_scatter(
    #             x=price_data.price_timeseries.index,
    #             y=price_data.price_timeseries.loc[:, (commodity_type, 'source', 'source')].values,
    #             line=go.scatter.Line(shape='hv')
    #         )
    #         figure.update_layout(
    #             title=f'Price: {commodity_type}',
    #             xaxis=go.layout.XAxis(tickformat='%H:%M')
    #         )
    #         # figure.show()
    #         fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'price_{commodity_type}'))
    #
    # # Print results path.
    # fledge.utils.launch(results_path)
    # print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
