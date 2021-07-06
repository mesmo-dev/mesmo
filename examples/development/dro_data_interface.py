"""Example script for DRO problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge
import statistics


class DRO_data(object):
    """DRO data object."""
    energy_price: pd.DataFrame
    contingency_reserve_price: pd.DataFrame
    forecast_price_raw: pd.DataFrame
    dro_base_data: pd.DataFrame
    energy_price_deviation: pd.DataFrame
    contingency_reserve_price_deviation: pd.DataFrame

    mean_energy_price: float
    variance_energy_price: float
    mean_contingency_price: float
    variance_contingency_price: float

    def __init__(
            self,
            data_path: str,
    ):
        self.forecast_price_raw = \
            pd.read_csv(data_path + "price_forecast_2021_26_07.csv")
        # forecast_price_raw = pd.read_csv(
        #     "C:\\Users\\kai.zhang\\Desktop\\local_fledge_data\\dro_data\\price_forecast_2021_26_07.csv")
        self.energy_price = self.forecast_price_raw['USEP($/MWh)'] / 100

        self.contingency_reserve_price = self.forecast_price_raw['Contingency($/MWh)'] / 100

        self.dro_base_data = \
            pd.read_csv(data_path + "dro_base_data.csv" )

        self.mean_energy_price = statistics.mean(self.energy_price)

        self.variance_energy_price = statistics.variance(self.energy_price-self.mean_energy_price, 0)

        self.mean_contingency_price = statistics.mean(self.contingency_reserve_price)

        self.variance_contingency_price = statistics.variance(
            self.contingency_reserve_price - self.mean_contingency_price, 0
        )

        self.energy_price_deviation = self.energy_price - self.mean_energy_price

        self.contingency_reserve_price_deviation = self.contingency_reserve_price - self.mean_contingency_price

class DRO_ambiguity_set(object):
    gamma: np.array
    delta_lower_bound: np.array
    delta_upper_bound: np.array
    u_upper_bound: np.array

    def __init__(
            self,
            scenario_name: str,
            standard_form_stage_2: fledge.utils.StandardForm,
            delta_indices_stage2: float,
            dro_data_set: DRO_data
    ):

        number_of_uncertainties = len(delta_indices_stage2)

        self.gamma = np.zeros((number_of_uncertainties, 1))

        self.delta_lower_bound = np.zeros((number_of_uncertainties, 1))

        self.delta_upper_bound = np.zeros((number_of_uncertainties, 1))

        self.u_upper_bound = np.zeros((number_of_uncertainties, 1))

        der_model_set = fledge.der_models.DERModelSet(scenario_name)

        linear_electric_grid_model = fledge.electric_grid_models.LinearElectricGridModelGlobal(scenario_name)

        temp_indices = fledge.utils.get_index(
            standard_form_stage_2.variables, name='uncertainty_energy_price_deviation_s2',
            timestep=linear_electric_grid_model.electric_grid_model.timesteps
        )

        self.gamma[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = 1*dro_data_set.variance_energy_price

        self.delta_lower_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = \
            10*dro_data_set.energy_price_deviation.min()

        self.delta_upper_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = \
            10*dro_data_set.energy_price_deviation.max()

        self.u_upper_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = \
            100*max(dro_data_set.energy_price_deviation.min()**2, dro_data_set.energy_price_deviation.max()**2)

        temp_indices = fledge.utils.get_index(
            standard_form_stage_2.variables, name='uncertainty_up_reserve_price_deviation_s2',
            timestep=linear_electric_grid_model.electric_grid_model.timesteps
        )

        self.gamma[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = dro_data_set.variance_contingency_price

        self.delta_lower_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = \
            dro_data_set.contingency_reserve_price_deviation.min()

        self.delta_upper_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = \
            dro_data_set.contingency_reserve_price_deviation.max()

        self.u_upper_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = \
            max(
                    dro_data_set.contingency_reserve_price_deviation.min()**2,
                    dro_data_set.contingency_reserve_price_deviation.max()**2
                )

        temp_indices = fledge.utils.get_index(
            standard_form_stage_2.variables, name='uncertainty_down_reserve_price_deviation_s2',
            timestep=linear_electric_grid_model.electric_grid_model.timesteps
        )

        self.gamma[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = dro_data_set.variance_contingency_price

        self.delta_lower_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = \
            dro_data_set.contingency_reserve_price_deviation.min()

        self.delta_upper_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = \
            dro_data_set.contingency_reserve_price_deviation.max()

        self.u_upper_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = \
            max(
                    dro_data_set.contingency_reserve_price_deviation.min()**2,
                    dro_data_set.contingency_reserve_price_deviation.max()**2
                )


        for der_name, der_model in der_model_set.flexible_der_models.items():
            if not der_model.disturbances.empty:
                temp_indices = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                    timestep=der_model.timesteps, der_name=[der_model.der_name],
                    disturbance=der_model.disturbances,
                )

                self.gamma[np.where(
                    pd.Index(delta_indices_stage2).isin(temp_indices))] = 20

                self.delta_lower_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = -5

                self.delta_upper_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = 5

                self.u_upper_bound[np.where(pd.Index(delta_indices_stage2).isin(temp_indices))] = 100



def main():
    scenario_name = 'singapore_6node_custom'

    # Get results path.
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV definition files.
    fledge.data_interface.recreate_database()

    dro_data_test = DRO_data("C:\\Users\\kai.zhang\\Desktop\\local_fledge_data\\dro_data\\")

    dro_price_deviation = dro_data_test.energy_price - dro_data_test.mean_energy_price

    print()
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
