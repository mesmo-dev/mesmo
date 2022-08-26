import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp
import mesmo

# To be commented out if deployed as a module
from bscs_data_interface import data_bscs
import numpy as np

class deterministic_acopf_battery_placement_sizing(object):

    def __init__(
            self,
            scenario_name,
            data_set,
            enable_electric_grid_model=True
    ):

        mesmo.utils.logger.info('Initializing battery sizing placement')

        # Obtain price data object.
        price_data = mesmo.data_interface.PriceData(scenario_name)

        # Obtain DER & grid model objects.
        self.der_model_set = mesmo.der_models.DERModelSet(scenario_name)

        linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(scenario_name)

        # settings
        self.scenarios = ['deterministic']
        self.timesteps = self.der_model_set.timesteps
        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (
            (self.der_model_set.timesteps[1] - self.der_model_set.timesteps[0]) / pd.Timedelta('1h')
        )

        # Manipulate building model to avoid over-consumption for up-reserves.
        for der_name, der_model in self.der_model_set.der_models.items():
            if isinstance(der_model, mesmo.der_models.FlexibleBuildingModel):
                der_model.output_maximum_timeseries.loc[
                    :, der_model.output_maximum_timeseries.columns.str.contains('_heat_')
                ] = 0.0
                der_model.output_maximum_timeseries.iloc[
                    -1, der_model.output_maximum_timeseries.columns.str.contains('_cool_')
                ] = 0.0
                der_model.output_minimum_timeseries.iloc[
                    -1, der_model.output_minimum_timeseries.columns.str.contains('_air_flow')
                ] = 0.0

        # Instantiate optimization problem.
        self.optimization_problem = mesmo.utils.OptimizationProblem()

        # Define BESS placement variables/constraints
        self.optimization_problem.define_variable(
            "battery_capacity",
            scenario=self.scenarios,
            node=linear_electric_grid_model_set.electric_grid_model.nodes,
        )

        self.optimization_problem.define_variable(
            "battery_charge_power",
            scenario=self.scenarios,
            timestep=self.der_model_set.timesteps,
            node=linear_electric_grid_model_set.electric_grid_model.nodes,
        )

        self.optimization_problem.define_variable(
            "battery_discharge_power",
            scenario=self.scenarios,
            timestep=self.der_model_set.timesteps,
            node=linear_electric_grid_model_set.electric_grid_model.nodes,
        )

        self.optimization_problem.define_variable(
            "battery_placement_binary",
            variable_type='binary',
            scenario=self.scenarios,
            node=linear_electric_grid_model_set.electric_grid_model.nodes,
        )

        # Define price arbitrage variables.
        self.optimization_problem.define_variable(
            'energy_root_node', timestep=self.timesteps, scenario=self.scenarios,
        )

        # Define DER problem.
        self.der_model_set.define_optimization_variables(
            self.optimization_problem, scenarios=self.scenarios
        )
        self.der_model_set.define_optimization_parameters(
            self.optimization_problem, price_data, scenarios=self.scenarios
        )
        self.der_model_set.define_optimization_constraints(
            self.optimization_problem, scenarios=self.scenarios
        )

        # Define electric grid problem.
        if enable_electric_grid_model:
            linear_electric_grid_model_set.define_optimization_variables(
                self.optimization_problem, scenarios=self.scenarios
            )
            linear_electric_grid_model_set.define_optimization_parameters(
                self.optimization_problem, price_data, scenarios=self.scenarios
            )

            # Define active loss equation.
            self.optimization_problem.define_constraint(
                ("variable", 1.0, dict(name="loss_active", scenario=self.scenarios, timestep=self.timesteps)),
                "==",
                (
                    "variable",
                    "loss_active_active_term",
                    dict(
                        name="der_active_power_vector",
                        scenario=self.scenarios,
                        timestep=self.timesteps,
                        der=linear_electric_grid_model_set.electric_grid_model.ders,
                    ),
                ),
                (
                    "variable",
                    "loss_active_reactive_term",
                    dict(
                        name="der_reactive_power_vector",
                        scenario=self.scenarios,
                        timestep=self.timesteps,
                        der=linear_electric_grid_model_set.electric_grid_model.ders,
                    ),
                ),
                (
                    "variable",
                    sp.block_diag(
                        [
                            linear_electric_grid_model.sensitivity_loss_active_by_power_wye_active
                            for linear_electric_grid_model in linear_electric_grid_model_set.linear_electric_grid_models.values()
                        ]
                    ),
                    dict(
                            name="battery_discharge_power",
                            scenario=self.scenarios,
                            timestep=self.timesteps,
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    ),
                ),
                (
                    "variable",
                    -sp.block_diag(
                        [
                            linear_electric_grid_model.sensitivity_loss_active_by_power_wye_active
                            for linear_electric_grid_model in linear_electric_grid_model_set.linear_electric_grid_models.values()
                        ]
                    ),
                    dict(
                            name="battery_charge_power",
                            scenario=self.scenarios,
                            timestep=self.timesteps,
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    ),
                ),
                ("constant", "loss_active_constant", dict(scenario=self.scenarios, timestep=self.timesteps)),
                broadcast=["scenario"],
            )


        # Define BESS placement constraints
        # BESS placement: SOC
        for t in range(self.timesteps.size-1):
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    data_set.battery_data['soc min'].values,
                    dict(
                            name="battery_capacity", scenario=self.scenarios,
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    )
                ),
                "<=",
                (
                    "variable",
                    data_set.battery_data['soc init'].values,
                    dict(
                            name="battery_capacity", scenario=self.scenarios,
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    )
                ),
                (
                    "variable",
                    data_set.battery_data['battery efficiency'].values * timestep_interval_hours * np.ones(t+1),
                    dict(
                            name="battery_charge_power", scenario=self.scenarios, timestep=self.timesteps[:t+1],
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    ),
                ),
                (
                    "variable",
                    -1/data_set.battery_data['battery efficiency'].values * timestep_interval_hours * np.ones(t+1),
                    dict(
                            name="battery_discharge_power", scenario=self.scenarios, timestep=self.timesteps[:t+1],
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    ),
                ),
                broadcast=["node", "scenario"],
            )

            self.optimization_problem.define_constraint(
                (
                    "variable",
                    data_set.battery_data['soc max'].values,
                    dict(
                            name="battery_capacity", scenario=self.scenarios,
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    )
                ),
                ">=",
                (
                    "variable",
                    data_set.battery_data['soc init'].values,
                    dict(
                            name="battery_capacity", scenario=self.scenarios,
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    )
                ),
                (
                    "variable",
                    data_set.battery_data['battery efficiency'].values * timestep_interval_hours * np.ones(t+1),
                    dict(
                            name="battery_charge_power", scenario=self.scenarios, timestep=self.timesteps[:t+1],
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    ),
                ),
                (
                    "variable",
                    -1/data_set.battery_data['battery efficiency'].values * timestep_interval_hours * np.ones(t+1),
                    dict(
                            name="battery_discharge_power", scenario=self.scenarios, timestep=self.timesteps[:t+1],
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    ),
                ),
                broadcast=["node", "scenario"],
            )

        # Placement decision variable constraint
        self.optimization_problem.define_parameter(
            "maximal_number_of_battery",
            data_set.battery_data['max battery number'].values[0]
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                np.ones(self.optimization_problem.get_variable_index(name="battery_placement_binary").size),
                dict(
                        name="battery_placement_binary", scenario=self.scenarios,
                        node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            "<=",
            (
                "constant",
                "maximal_number_of_battery",
                dict(
                        scenario=self.scenarios,
                )
            ),
            broadcast=["scenario"],
        )

        # Battery capacity constraint
        self.optimization_problem.define_parameter(
            "maximal_battery_capacity",
            np.ones(self.optimization_problem.get_variable_index(name="battery_placement_binary").size) *
            data_set.battery_data['min battery capacity (kWh)'].values
        )

        self.optimization_problem.define_parameter(
            "minimal_battery_capacity",
            np.ones(self.optimization_problem.get_variable_index(name="battery_placement_binary").size) *
            data_set.battery_data['max battery capacity (kWh)'].values
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_capacity", scenario=self.scenarios,
                        node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            "<=",
            (
                "constant",
                #"maximal_battery_capacity",
                data_set.battery_data['max battery capacity (kWh)'].values[0],
                dict(
                        scenario=self.scenarios, node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            broadcast=["scenario", "node"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_capacity", scenario=self.scenarios,
                    node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            ">=",
            (
                "constant",
                #"minimal_battery_capacity",
                data_set.battery_data['min battery capacity (kWh)'].values[0],
                dict(
                    scenario=self.scenarios, node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            broadcast=["scenario", "node"],
        )

        # Power charge/discharge constraints
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_charge_power", scenario=self.scenarios, timestep=self.timesteps,
                        node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            ">=",
            (
                "constant",
                0,
                dict(
                        scenario=self.scenarios, timestep=self.timesteps,
                        node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            broadcast=["scenario", "timestep", "node"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_discharge_power", scenario=self.scenarios, timestep=self.timesteps,
                    node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            ">=",
            (
                "constant",
                0,
                dict(
                    scenario=self.scenarios, timestep=self.timesteps,
                    node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            broadcast=["scenario", "timestep", "node"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_charge_power", scenario=self.scenarios, timestep=self.timesteps,
                    node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            "<=",
            (
                "constant",
                data_set.battery_data['charging power max (kW)'].values[0],
                dict(
                    scenario=self.scenarios, timestep=self.timesteps,
                    node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            broadcast=["scenario", "timestep", "node"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_discharge_power", scenario=self.scenarios, timestep=self.timesteps,
                    node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            "<=",
            (
                "constant",
                data_set.battery_data['charging power max (kW)'].values[0],
                dict(
                    scenario=self.scenarios, timestep=self.timesteps,
                    node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            broadcast=["scenario", "timestep", "node"],
        )

        # Big M reformulation of bi-linear terms into linear constraints
        # self.optimization_problem.define_parameter("big_M", np.array([1e5]))
        self.big_M_constant = 1e3

        for t in range(self.timesteps.size):
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="battery_charge_power", scenario=self.scenarios, timestep=self.timesteps[t],
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    ),
                ),
                "<=",
                (
                    "variable",
                    self.big_M_constant,
                    dict(
                            name="battery_placement_binary", scenario=self.scenarios,
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    )
                ),
                broadcast=["scenario", "node"],
            )

            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="battery_discharge_power", scenario=self.scenarios, timestep=self.timesteps[t],
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    ),
                ),
                "<=",
                (
                    "variable",
                    self.big_M_constant,
                    dict(
                            name="battery_placement_binary", scenario=self.scenarios,
                            node=linear_electric_grid_model_set.electric_grid_model.nodes,
                    )
                ),
                broadcast=["scenario", "node"],
            )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_charge_power", scenario=self.scenarios, timestep=self.timesteps,
                        node=linear_electric_grid_model_set.electric_grid_model.nodes,
                ),
            ),
            ">=",
            (
                "constant",
                0,
                dict(
                        scenario=self.scenarios, timestep=self.timesteps,
                        node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            broadcast=["timestep", "scenario", "node"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_discharge_power", scenario=self.scenarios, timestep=self.timesteps,
                        node=linear_electric_grid_model_set.electric_grid_model.nodes,
                ),
            ),
            ">=",
            (
                "constant",
                0,
                dict(
                        scenario=self.scenarios, timestep=self.timesteps,
                        node=linear_electric_grid_model_set.electric_grid_model.nodes,
                )
            ),
            broadcast=["timestep", "scenario", "node"],
        )

        # self.optimization_problem.define_constraint(
        #     (
        #         "variable",
        #         1,
        #         dict(
        #             name="battery_capacity", scenario=self.scenarios,
        #             node=linear_electric_grid_model_set.electric_grid_model.nodes,
        #         ),
        #     ),
        #     "<=",
        #     (
        #         "variable",
        #         self.big_M_constant,
        #         dict(
        #             name="battery_placement_binary", scenario=self.scenarios,
        #             node=linear_electric_grid_model_set.electric_grid_model.nodes,
        #         )
        #     ),
        #     broadcast=["scenario", "node"],
        # )

        # Define power balance constraints.
        self.optimization_problem.define_constraint(
            ('variable', 1.0, dict(name='energy_root_node', scenario=self.scenarios, timestep=self.timesteps)),
            '==',
            ('variable', (
                -1.0
                * np.array([np.real(linear_electric_grid_model_set.electric_grid_model.der_power_vector_reference)])
            ), dict(
                        name='der_active_power_vector',  scenario=self.scenarios, timestep=self.timesteps,
                        der=self.der_model_set.ders
            )),
            ('variable',
             -np.ones(self.optimization_problem.get_variable_index(name="battery_placement_binary").size),
             dict(
                     name="battery_discharge_power", scenario=self.scenarios, timestep=self.timesteps,
                     node=linear_electric_grid_model_set.electric_grid_model.nodes,
             ),
            ),
            ('variable',
             np.ones(self.optimization_problem.get_variable_index(name="battery_placement_binary").size),
             dict(
                     name="battery_charge_power", scenario=self.scenarios, timestep=self.timesteps,
                     node=linear_electric_grid_model_set.electric_grid_model.nodes,
             ),
            ),
            ("variable", 1.0, dict(name="loss_active", scenario=self.scenarios, timestep=self.timesteps)),
            broadcast=["timestep", "scenario"],
        )

        # Define objective.
        # Obtain energy price timeseries.
        price_timeseries_energy = (
            data_set.annual_average_energy_price[0:len(self.der_model_set.timesteps)].to_numpy()
        )

        self.optimization_problem.define_objective(
            ('variable', (
                    1.0 * np.array([price_timeseries_energy]) * timestep_interval_hours
            ), dict(
                        name='energy_root_node', timestep=self.der_model_set.timesteps
            )),
            # ('variable', (
            #                 data_set.battery_data['battery_investment cost ($/kWh)'].values[0] *
            #                 np.ones(len(linear_electric_grid_model_set.electric_grid_model.nodes))
            # ), dict(
            #             name='battery_capacity'
            # )),
            ('variable', (
                    sp.diags(data_set.battery_data['battery_investment cost ($/kWh)'].values[0] *
                    np.ones(len(linear_electric_grid_model_set.electric_grid_model.nodes)))
            ), dict(name='battery_capacity'), dict(name='battery_placement_binary')),
            ('variable', (
                data_set.battery_data['battery degradation coefficient ($/kWh^2)'].values[0] * sp.block_diag(
                    [sp.diags(np.ones(len(linear_electric_grid_model_set.electric_grid_model.nodes)))] * len(self.timesteps)
             ))
            , dict(name='battery_charge_power'), dict(name='battery_charge_power')),
            ('variable', (
                    data_set.battery_data['battery degradation coefficient ($/kWh^2)'].values[0] * sp.block_diag(
                [sp.diags(np.ones(len(linear_electric_grid_model_set.electric_grid_model.nodes)))] * len(self.timesteps)
            ))
             , dict(name='battery_discharge_power'), dict(name='battery_discharge_power')),
        )


def main():

    # regulation signal time step
    reg_time_constant = 0.02
    # Settings.
    scenario_name = 'paper_2021_zhang_dro'
    mesmo.data_interface.recreate_database()

    # Obtain data.
    data_set = data_bscs(os.path.join(os.path.dirname(os.path.normpath(__file__)), 'Dataset'))

    samples_to_plot = data_set.reg_d_data_40min_sample.iloc[0:-1]
    fig = px.line(samples_to_plot['RegDTest'], labels=dict(x="time step (0.2s)", value="CRS", variable="Day index"))
    fig.show()

    # RegD data per day
    day_1_reD = data_set.reg_d_data_whole_day[pd.datetime(2020, 1, 1, 0, 0)]
    day_2_reD = data_set.reg_d_data_whole_day[pd.datetime(2020, 1, 2, 0, 0)]
    day_3_reD = data_set.reg_d_data_whole_day[pd.datetime(2020, 1, 3, 0, 0)]
    day_4_reD = data_set.reg_d_data_whole_day[pd.datetime(2020, 1, 4, 0, 0)]
    day_5_reD = data_set.reg_d_data_whole_day[pd.datetime(2020, 1, 5, 0, 0)]

    # dict for the dataframes and their names
    dfs = {"day_1_CRS": day_1_reD.cumsum().values * reg_time_constant,
           "day_2_CRS": day_2_reD.cumsum().values * reg_time_constant,
           "day_3_CRS": day_3_reD.cumsum().values * reg_time_constant,
           "day_4_CRS": day_4_reD.cumsum().values * reg_time_constant,
           "day_5_CRS": day_5_reD.cumsum().values * reg_time_constant}

    dfs = pd.DataFrame(dfs)

    # plot the data
    fig = go.Figure()

    fig = px.line(dfs, x=dfs.index.values, y=["day_1_CRS", "day_2_CRS", "day_3_CRS", "day_4_CRS", "day_5_CRS"],
                  labels=dict(x="time step (0.2s)", value="CRS", variable="Day index"))
    fig.show()

    # Get results path.
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Get standard form of stage 1.
    # optimal_sizing_problem = deterministic_acopf_battery_placement_sizing(scenario_name, data_set)
    #
    # optimal_sizing_problem.optimization_problem.solve()
    # results = optimal_sizing_problem.optimization_problem.get_results()
    #
    # a_matrix = optimal_sizing_problem.optimization_problem.get_a_matrix()
    # b_vector = optimal_sizing_problem.optimization_problem.get_b_vector().transpose()[0]
    # c_vector = optimal_sizing_problem.optimization_problem.get_c_vector()[0]
    # q_matrix = optimal_sizing_problem.optimization_problem.get_q_matrix()
    # d_vector = np.array([optimal_sizing_problem.optimization_problem.get_d_constant()])


if __name__ == '__main__':
    main()