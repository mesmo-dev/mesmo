import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp
import mesmo
from bscs_data_interface import data_bscs


class bscs_wep_optimization_model(object):
    def __init__(
            self,
            scenario_name,
            data_set,
            data_set_swapping_demand,
            time_step,
            enable_electric_grid_model=False
    ):
        big_M_constant = 10000

        mesmo.utils.logger.info('Initializing BSCS model...')

        # Obtain price data object.
        price_data = mesmo.data_interface.PriceData(scenario_name)

        # Obtain DER & grid model objects.
        self.der_model_set = mesmo.der_models.DERModelSet(scenario_name)
        linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(scenario_name)

        # settings
        # for stochastic optimisation
        self.scenarios = ['deterministic']

        # battery slot index
        number_of_battery_slot = data_set.bscs_data['number_of_battery_slots'].values[0]
        battery_slot_index = list(range(number_of_battery_slot))
        self.battery_slot = ['battery_slot_no_{}'.format(x) for x in battery_slot_index]

        # time steps for electricity market participation
        self.timesteps = time_step
        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (
            (self.timesteps[1] - self.timesteps[0]) / pd.Timedelta('1h')
        )

        # Instantiate optimization problem.
        self.optimization_problem = mesmo.utils.OptimizationProblem()

        # self.optimization_problem.define_variable(
        #     "battery_regulation_power",
        #     scenario=self.scenarios,
        #     timestep=self.der_model_set.timesteps,
        #     battery_slot=self.battery_slot,
        # )

        self.optimization_problem.define_variable(
            "battery_charge_power",
            scenario=self.scenarios,
            timestep=self.timesteps,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_discharge_power",
            scenario=self.scenarios,
            timestep=self.timesteps,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_slot_energy",
            scenario=self.scenarios,
            timestep=self.timesteps,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_slot_soc",
            scenario=self.scenarios,
            timestep=self.timesteps,
            battery_slot_indices=self.battery_slot,
        )

        # Define price arbitrage variables.
        self.optimization_problem.define_variable(
            'swapping_station_total_energy',
            timestep=self.timesteps,
            scenario=self.scenarios,
        )

        # Define A_t matrix
        # number of incoming batteries:  data_set_swapping_demand.data_number_ev_to_be_swapped_dict[time_step[3]]
        for time_step_temp in self.timesteps:
            self.optimization_problem.define_variable(
                'A_matrix',
                variable_type='binary',
                timestep=time_step_temp,
                scenario=self.scenarios,
                battery_slot_indices=self.battery_slot,
                ev_indices=np.array(list(range(int(data_set_swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])))),
            )

        # Define Z_t matrix
        for time_step_temp in self.timesteps:
            self.optimization_problem.define_variable(
                'Z_matrix',
                timestep=time_step_temp,
                scenario=self.scenarios,
                battery_slot_indices=self.battery_slot,
                ev_indices=np.array(list(range(int(data_set_swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])))),
            )

        # Define constraints
        # TODO
        # Define energy constraints
        for timestep_temp in time_step:
            # number of incoming EVs per timestep
            number_ev_per_step = data_set_swapping_demand.data_SOC_ev_to_be_swapped_dict[timestep_temp].size
            A_coefficient_matrix = np.zeros([number_of_battery_slot, number_of_battery_slot*number_ev_per_step])

            for index_temp in range(number_of_battery_slot):
                A_coefficient_matrix[index_temp, index_temp*number_ev_per_step: (index_temp+1)*number_ev_per_step] = \
                    data_set_swapping_demand.data_energy_ev_to_be_swapped_dict[timestep_temp]

            ## Debug print('debug')
            # self.optimization_problem.get_variable_index(name='A_matrix', timestep=timestep_temp)

            # TODO + add time step plus+
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_energy", scenario=self.scenarios,
                            timestep=timestep_temp, #TODO
                    )
                ),
                "==",
                (
                    "variable",
                    A_coefficient_matrix,
                    dict(
                            name="battery_slot_energy",
                            scenario=self.scenarios,
                            timestep=timestep_temp,
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




        # Define BESS placement variables/constraints
        self.optimization_problem.define_variable(
            "battery_capacity",
            scenario=self.scenarios,
            battery_slot=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_charge_power",
            scenario=self.scenarios,
            timestep=self.der_model_set.timesteps,
            battery_slot=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_discharge_power",
            scenario=self.scenarios,
            timestep=self.der_model_set.timesteps,
            battery_slot=self.battery_slot,
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
    ...


if __name__ == '__main__':
    main()
