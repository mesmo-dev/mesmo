import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp

import mesmo

# To be commented out if deployed as a module
from data_interface import data_battery_sizing_placement

# from pymoo.algorithms.soo.nonconvex.de import DE
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.core.problem import Problem, ElementwiseProblem
# from pymoo.operators.sampling.lhs import LHS
# from pymoo.optimize import minimize
# from pymoo.factory import get_sampling, get_crossover, get_mutation
# from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

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

        # # self.optimization_problem.variables.iloc[self.optimization_problem.get_variable_index(name="node_voltage_magnitude_vector",
        #     # scenario=self.scenarios,timestep=self.timesteps, node=linear_electric_grid_model_set.electric_grid_model.nodes,)]
        #     # Define voltage equation.
        #     self.optimization_problem.define_constraint(
        #         (
        #             "variable",
        #             1.0,
        #             dict(
        #                 name="node_voltage_magnitude_vector",
        #                 scenario=self.scenarios,
        #                 timestep=self.timesteps,
        #                 node=linear_electric_grid_model_set.electric_grid_model.nodes,
        #             ),
        #         ),
        #         "==",
        #         (
        #             "variable",
        #             "voltage_active_term",
        #             dict(
        #                 name="der_active_power_vector",
        #                 scenario=self.scenarios,
        #                 timestep=self.timesteps,
        #                 der=linear_electric_grid_model_set.electric_grid_model.ders,
        #             ),
        #         ),
        #         (
        #             "variable",
        #             "voltage_reactive_term",
        #             dict(
        #                 name="der_reactive_power_vector",
        #                 scenario=self.scenarios,
        #                 timestep=self.timesteps,
        #                 der=linear_electric_grid_model_set.electric_grid_model.ders,
        #             ),
        #         ),
        #         ("constant", "voltage_constant", dict(scenario=self.scenarios, timestep=self.timesteps)),
        #         (
        #             "variable",
        #             -1.0e-8, ## need to revise to: linear_electric_grid_model_set.linear_electric_grid_models.sensitivity?
        #             dict(
        #                 name="battery_charge_power",
        #                 scenario=self.scenarios,
        #                 timestep=self.timesteps,
        #                 node=linear_electric_grid_model_set.electric_grid_model.nodes,
        #             ),
        #         ),
        #         (
        #             "variable",
        #             1.0e-8,  ## need to revise to: linear_electric_grid_model_set.linear_electric_grid_models.sensitivity?
        #             dict(
        #                     name="battery_discharge_power",
        #                     scenario=self.scenarios,
        #                     timestep=self.timesteps,
        #                     node=linear_electric_grid_model_set.electric_grid_model.nodes,
        #             ),
        #         ),
        #         broadcast="scenario",
        #     )
        #     # Define voltage limits.
        #     self.optimization_problem.define_constraint(
        #         (
        #             "variable",
        #             1.0,
        #             dict(
        #                 name="node_voltage_magnitude_vector",
        #                 scenario=self.scenarios,
        #                 timestep=self.timesteps,
        #                 node=linear_electric_grid_model_set.electric_grid_model.nodes,
        #             ),
        #         ),
        #         ">=",
        #         ("constant", "voltage_limit_minimum", dict(scenario=self.scenarios, timestep=self.timesteps)),
        #         keys=dict(
        #             name="voltage_magnitude_vector_minimum_constraint",
        #             scenario=self.scenarios,
        #             timestep=self.timesteps,
        #             node=linear_electric_grid_model_set.electric_grid_model.nodes,
        #         ),
        #         broadcast="scenario",
        #     )
        #
        #     self.optimization_problem.define_constraint(
        #         (
        #             "variable",
        #             1.0,
        #             dict(
        #                 name="node_voltage_magnitude_vector",
        #                 scenario=self.scenarios,
        #                 timestep=self.timesteps,
        #                 node=linear_electric_grid_model_set.electric_grid_model.nodes,
        #             ),
        #         ),
        #         "<=",
        #         ("constant", "voltage_limit_maximum", dict(scenario=self.scenarios, timestep=self.timesteps)),
        #         keys=dict(
        #             name="voltage_magnitude_vector_maximum_constraint",
        #             scenario=self.scenarios,
        #             timestep=self.timesteps,
        #             node=linear_electric_grid_model_set.electric_grid_model.nodes,
        #         ),
        #         broadcast="scenario",
        #     )

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

        #self.optimization_problem.solve()


# class ConstrainedQuadraticProblem(ElementwiseProblem):
#
#     def __init__(self, Q, c, d, A, b):
#         self.Q = Q
#         self.c = c
#         self.d = d
#         self.A = A
#         self.b = b
#
#         super().__init__(n_var=len(Q), n_obj=1, n_constr=1, xl=-1e5, xu=1e5)
#
#     def _evaluate(self, x, out, *args, **kwargs):
#         out["F"] = x.T @ (0.5 * self.Q) @ x + x @ self.c + self.d
#         out["G"] = self.A @ x - self.b


def main():

    # Settings.
    scenario_name = 'paper_2021_zhang_dro'
    mesmo.data_interface.recreate_database()

    # Obtain data.
    data_set = data_battery_sizing_placement(os.path.join(os.path.dirname(os.path.normpath(__file__)), 'test_case_customized'))

    # Get results path.
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Get standard form of stage 1.
    optimal_sizing_problem = deterministic_acopf_battery_placement_sizing(scenario_name, data_set)

    optimal_sizing_problem.optimization_problem.solve()
    results = optimal_sizing_problem.optimization_problem.get_results()

    a_matrix = optimal_sizing_problem.optimization_problem.get_a_matrix()
    b_vector = optimal_sizing_problem.optimization_problem.get_b_vector().transpose()[0]
    c_vector = optimal_sizing_problem.optimization_problem.get_c_vector()[0]
    q_matrix = optimal_sizing_problem.optimization_problem.get_q_matrix()
    d_vector = np.array([optimal_sizing_problem.optimization_problem.get_d_constant()])

    # problem = ConstrainedQuadraticProblem(q_matrix.toarray(), c_vector, d_vector, a_matrix.toarray(), b_vector)
    #
    # # algorithm = DE(
    # #     pop_size=100,
    # #     sampling=LHS(),
    # #     variant="DE/rand/1/bin",
    # #     CR=0.3,
    # #     dither="vector",
    # #     jitter=False
    # # )
    #
    # # algorithm = GA()
    #
    # binary_indices = optimal_sizing_problem.optimization_problem.variables.loc[:, "variable_type"] == "binary"
    # # mask = ["int", "real"]
    # mask = pd.array(["real" for i in range(binary_indices.size)])
    # mask[binary_indices.values.nonzero()] = 'int'
    #
    # mask_list = mask.tolist()
    # #
    # sampling = MixedVariableSampling(mask_list, {
    #     "real": get_sampling("real_random"),
    #     "int": get_sampling("bin_random")
    # })
    #
    # crossover = MixedVariableCrossover(mask_list, {
    #     "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    #     "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
    # })
    #
    # mutation = MixedVariableMutation(mask_list, {
    #     "real": get_mutation("real_pm", eta=3.0),
    #     "int": get_mutation("int_pm", eta=3.0)
    # })
    # #
    # algorithm = GA(
    #     pop_size=200,
    #     sampling=sampling,
    #     crossover=crossover,
    #     mutation=mutation,
    #     eliminate_duplicates=True,
    # )
    # #
    # res = minimize(problem,
    #                algorithm,
    #                seed=1,
    #                verbose=True)

    # # solve the problem with GA
    # file = 'a_matrix'
    # np.save(file, a_matrix)
    #
    # file = 'b_vector'
    # np.save(file, b_vector)
    #
    # file = 'c_vector'
    # np.save(file, c_vector)
    #
    # file = 'q_matrix'
    # np.save(file, q_matrix)
    #
    # file = 'd_vector'
    # np.save(file, d_vector)
    #
    # a_matrix = np.load('a_matrix.npy', allow_pickle=True)
    # b_vector = np.load('b_vector.npy', allow_pickle=True)
    # c_vector = np.load('c_vector.npy', allow_pickle=True)
    # q_matrix = np.load('q_matrix.npy', allow_pickle=True)
    # d_vector = np.load('d_vector.npy', allow_pickle=True)

    # objs = [
    #     lambda x:  c_vector @ x + x.T @ (0.5 * q_matrix) @ x + d_vector
    # ]
    #
    # constr_ieq = [lambda x: a_matrix @ x - b_vector]
    #
    # n_var = c_vector.size
    #
    # problem = FunctionalProblem(n_var,
    #                             objs,
    #                             constr_ieq=constr_ieq,
    #                             )
    #
    # F, CV = problem.evaluate(np.random.rand(2, n_var))
    #
    # algorithm = GA(
    #     pop_size=100,
    #     eliminate_duplicates=True)
    #
    # res = minimize(problem,
    #                algorithm,
    #                seed=1,
    #                verbose=False)
    #
    # print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))



if __name__ == '__main__':
    main()
