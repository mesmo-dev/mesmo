import numpy as np
import scipy.sparse as sp
import pandas as pd
import typing
import mesmo
from mesmo.der_models import DERModelSet
from mesmo.electric_grid_models import LinearElectricGridModelSet, ElectricGridModelDefault, PowerFlowSolutionFixedPoint


class StrategicMarket(object):
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain electric grid model & reference power flow solution.
        electric_grid_model = ElectricGridModelDefault(scenario_name)
        power_flow_solution = PowerFlowSolutionFixedPoint(electric_grid_model)
        self.linear_electric_grid_model_set = LinearElectricGridModelSet(electric_grid_model, power_flow_solution)
        self.der_model_set = DERModelSet(scenario_name)

        if len(self.der_model_set.electric_ders) > 0:
            self.timestep_interval_hours = (self.der_model_set.timesteps[1] -
                                            self.der_model_set.timesteps[0]) / pd.Timedelta('1h')
            # _____________________________
        flexible_generator_index = [der for der in self.der_model_set.electric_ders if 'flexible_generator' in der]
        self.strategic_generator_set_to_zero_map = pd.DataFrame(0, index=self.der_model_set.electric_ders,
                                                                columns=self.der_model_set.electric_ders)
        for i in self.strategic_generator_set_to_zero_map.index:
            for c in self.strategic_generator_set_to_zero_map.columns:
                if i == c and '4_5' not in i:
                    self.strategic_generator_set_to_zero_map.at[i, c] = 1

        self.timesteps = self.linear_electric_grid_model_set.timesteps
        self.ders = self.linear_electric_grid_model_set.electric_grid_model.ders
        self.nodes = self.linear_electric_grid_model_set.electric_grid_model.nodes
        self.branches = self.linear_electric_grid_model_set.electric_grid_model.branches
        self.flexible_load_index = [der for der in self.ders if 'flexible_load' in der]
        self.flexible_load_map = pd.DataFrame(0, index=self.flexible_load_index, columns=self.ders)
        for i in self.flexible_load_map.index:
            for c in self.flexible_load_map.columns:
                if i == c:
                    self.flexible_load_map.at[i, c] = 1

        self.strategic_generator_index = [der for der in self.ders if '4_5' in der]
        self.flexible_generator_map = pd.DataFrame(0, index=self.ders, columns=self.strategic_generator_index, )
        for i in self.flexible_generator_map.index:
            for c in self.flexible_generator_map.columns:
                if i == c:
                    self.flexible_generator_map.at[i, c] = 1

        self.flexible_generator_index = [der for der in self.ders if 'flexible_generator' in der]
        flexible_der_index = self.flexible_generator_index + self.flexible_load_index
        self.fixed_der_index = [der for der in self.ders if der not in flexible_der_index]
        self.flexible_der_index = [der for der in self.ders if der not in self.fixed_der_index]
        self.flexible_der_map = pd.DataFrame(0, index=self.ders, columns=self.flexible_der_index)
        for i in self.flexible_der_map.index:
            for c in self.flexible_der_map.columns:
                if i == c:
                    self.flexible_der_map.at[i, c] = 1

        self.non_flexible_der_set_to_zero_map = pd.DataFrame(0, index=self.ders, columns=self.ders)
        for i in self.non_flexible_der_set_to_zero_map.index:
            for c in self.non_flexible_der_set_to_zero_map.columns:
                if i == c and i not in self.fixed_der_index:
                    self.non_flexible_der_set_to_zero_map.at[i, c] = 1

    def strategic_optimization_problem(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            price_data,
            scenarios: typing.Union[list, pd.Index] = None,
            **kwargs
    ):
        self.define_optimization_variables(optimization_problem, scenarios)
        self.define_optimization_parameters(
            optimization_problem,
            price_data,
            scenarios=scenarios,
            **kwargs
        )
        self.define_optimization_constraints(optimization_problem, scenarios)
        self.define_objective_function(optimization_problem, scenarios)

    def define_optimization_variables(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            scenarios: typing.Union[list, pd.Index] = None
    ):
        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Defining strategic variables:
        optimization_problem.define_variable(
            'flexible_generator_strategic_offer', scenario=scenarios, timestep=self.timesteps,
            fg=self.strategic_generator_index
        )
        optimization_problem.define_variable(
            'der_strategic_offer', scenario=scenarios, timestep=self.timesteps,
            der=self.ders
        )
        optimization_problem.define_variable(
            'active_equal_to_reactive_power_mu', scenario=scenarios, timestep=self.timesteps,
            der=self.ders
        )
        optimization_problem.define_variable(
            'node_voltage_mu_maximum', scenario=scenarios, timestep=self.timesteps,
            node=self.nodes
        )
        optimization_problem.define_variable(
            'node_voltage_mu_maximum_binary', variable_type='binary', scenario=scenarios, timestep=self.timesteps,
            node=self.nodes
        )
        optimization_problem.define_variable(
            'node_voltage_mu_minimum', scenario=scenarios, timestep=self.timesteps,
            node=self.nodes
        )
        optimization_problem.define_variable(
            'node_voltage_mu_minimum_binary', variable_type='binary', scenario=scenarios, timestep=self.timesteps,
            node=self.nodes
        )
        optimization_problem.define_variable(
            'branch_1_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
            branch=self.branches
        )
        optimization_problem.define_variable(
            'branch_1_power_mu_maximum_binary', variable_type='binary', scenario=scenarios, timestep=self.timesteps,
            branch=self.branches
        )
        optimization_problem.define_variable(
            'branch_1_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
            branch=self.branches
        )
        optimization_problem.define_variable(
            'branch_1_power_mu_minimum_binary', variable_type='binary', scenario=scenarios, timestep=self.timesteps,
            branch=self.branches
        )
        optimization_problem.define_variable(
            'branch_2_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
            branch=self.branches
        )
        optimization_problem.define_variable(
            'branch_2_power_mu_maximum_binary', variable_type='binary', scenario=scenarios, timestep=self.timesteps,
            branch=self.branches
        )
        optimization_problem.define_variable(
            'branch_2_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
            branch=self.branches
        )
        optimization_problem.define_variable(
            'branch_2_power_mu_minimum_binary', variable_type='binary', scenario=scenarios, timestep=self.timesteps,
            branch=self.branches
        )
        optimization_problem.define_variable(
            'active_loss_mu', scenario=scenarios, timestep=self.timesteps,
        )
        optimization_problem.define_variable(
            'reactive_loss_mu', scenario=scenarios, timestep=self.timesteps,
        )
        optimization_problem.define_variable(
            'der_active_power_vector_mu_minimum', scenario=scenarios, timestep=self.timesteps,
            der=self.ders
        )
        optimization_problem.define_variable(
            'der_active_power_vector_mu_minimum_binary', variable_type='binary', scenario=scenarios,
            timestep=self.timesteps, der=self.ders
        )
        optimization_problem.define_variable(
            'der_active_power_vector_mu_maximum', scenario=scenarios, timestep=self.timesteps,
            der=self.ders
        )
        optimization_problem.define_variable(
            'der_active_power_vector_mu_maximum_binary', variable_type='binary', scenario=scenarios,
            timestep=self.timesteps, der=self.ders
        )
        optimization_problem.define_variable(
            'der_reactive_power_vector_mu_minimum', scenario=scenarios, timestep=self.timesteps,
            der=self.ders
        )
        optimization_problem.define_variable(
            'der_reactive_power_vector_mu_minimum_binary', variable_type='binary', scenario=scenarios,
            timestep=self.timesteps, der=self.ders
        )
        optimization_problem.define_variable(
            'der_reactive_power_vector_mu_maximum', scenario=scenarios, timestep=self.timesteps,
            der=self.ders
        )
        optimization_problem.define_variable(
            'der_reactive_power_vector_mu_maximum_binary', variable_type='binary', scenario=scenarios,
            timestep=self.timesteps, der=self.ders
        )

    def define_optimization_parameters(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            price_data: mesmo.data_interface.PriceData,
            node_voltage_magnitude_vector_minimum: np.ndarray = None,
            node_voltage_magnitude_vector_maximum: np.ndarray = None,
            branch_power_magnitude_vector_maximum: np.ndarray = None,
            active_power_vector_minimum: np.ndarray = None,
            active_power_vector_maximum: np.ndarray = None,
            reactive_power_vector_minimum: np.ndarray = None,
            reactive_power_vector_maximum: np.ndarray = None,
            big_m: int = 100,
            scenarios: typing.Union[list, pd.Index] = None,
    ):
        if scenarios is None:
            scenarios = [None]
        # ===================================================
        # flexible_der_type = ['flexible_generator', 'flexible_load']
        #
        # flexible_der_index = [(der_type, der_name) for der_type, der_name in self.ders if
        #                       der_type in flexible_der_type]
        # flexible_der_power_variable_map = pd.DataFrame(0.0, index=self.ders, columns=flexible_der_index)
        # der_maximum_limit = pd.Series(1, index=self.ders)
        # for i in der_maximum_limit.index:
        #     if 'flexible_load' in i:
        #         der_maximum_limit.at[i] = 1.2
        #
        #     elif 'flexible_generator' in i:
        #         der_maximum_limit.at[i] = 0.91
        #
        # der_minimum_limit = pd.Series(0, index=self.ders)
        # for i in der_minimum_limit.index:
        #     if 'flexible_load' in i:
        #         der_minimum_limit.at[i] = 0.51
        #
        # for i in flexible_der_power_variable_map.index:
        #     for c in flexible_der_power_variable_map.columns:
        #         if i == c:
        #             flexible_der_power_variable_map.at[i, c] = 1
        #
        # optimization_problem.define_parameter(
        #     'der_active_power_vector_maximum_limit',
        #     np.transpose([np.concatenate([der_maximum_limit.values] * len(self.timesteps))])
        # )
        # optimization_problem.define_parameter(
        #     'der_reactive_power_vector_maximum_limit',
        #     np.transpose([np.concatenate([der_maximum_limit.values] * len(self.timesteps))])
        # )
        # optimization_problem.define_parameter(
        #     'der_reactive_power_vector_maximum_limit_transposed',
        #     1.0 * np.array([np.concatenate([der_maximum_limit.values] * len(self.timesteps))])
        # )
        #
        # optimization_problem.define_parameter(
        #     'der_active_power_vector_minimum_limit',
        #     np.transpose([np.concatenate([der_minimum_limit.values] * len(self.timesteps))])
        # )
        # optimization_problem.define_parameter(
        #     'der_reactive_power_vector_minimum_limit',
        #     np.transpose([np.concatenate([der_minimum_limit.values] * len(self.timesteps))])
        # )
        # optimization_problem.define_parameter(
        #     'minus_der_reactive_power_vector_minimum_limit_transposed',
        #     - 1.0 * np.array([np.concatenate([der_minimum_limit.values] * len(self.timesteps))])
        # )
        #
        # optimization_problem.define_variable(
        #     'flexible_der_active_power',
        #     scenario=scenarios, timestep=self.timesteps, fd=flexible_der_index
        # )
        # optimization_problem.define_variable(
        #     'flexible_der_reactive_power',
        #     scenario=scenarios, timestep=self.timesteps, fd=flexible_der_index
        # )
        #
        # optimization_problem.define_parameter(
        #     'power_map_to_flexible_der_variable',
        #     sp.block_diag([flexible_der_power_variable_map.values] * 1)
        # )

        # optimization_problem.define_constraint(
        #     ('variable', 1.0, dict(
        #         name='der_active_power_vector', scenario=scenarios, timestep=self.timesteps,
        #         der=self.ders
        #     )),
        #     '==',
        #     ('constant', 'active_power_constant', dict(scenario=scenarios)),
        #     ('variable', 'power_map_to_flexible_der_variable', dict(
        #         name='flexible_der_active_power', scenario=scenarios, timestep=self.timesteps, fd=flexible_der_index
        #     )),
        #     broadcast=['timestep', 'scenario']
        # )

        # optimization_problem.define_constraint(
        #     ('variable', 1.0, dict(
        #         name='der_reactive_power_vector', scenario=scenarios, timestep=self.timesteps,
        #         der=self.ders
        #     )),
        #     '==',
        #     ('constant', 'reactive_power_constant', dict(scenario=scenarios)),
        #     ('variable', 'power_map_to_flexible_der_variable', dict(
        #         name='flexible_der_reactive_power', scenario=scenarios, timestep=self.timesteps, fd=flexible_der_index
        #     )),
        #     broadcast=['timestep', 'scenario']
        # )

        # optimization_problem.define_constraint(
        #     ('variable', 1.0, dict(
        #         name='der_active_power_vector', scenario=scenarios, timestep=self.timesteps,
        #         der=self.ders
        #     )),
        #     '==',
        #     ('variable', 1.0, dict(
        #         name='der_reactive_power_vector', scenario=scenarios, timestep=self.timesteps,
        #         der=self.ders
        #     )),
        #     broadcast='scenario'
        # )

        # optimization_problem.define_constraint(
        #     ('variable', 1.0, dict(
        #         name='der_active_power_vector',
        #         scenario=scenarios, timestep=self.timesteps, der=self.ders
        #     )),
        #     '<=',
        #     ('constant', 'der_active_power_vector_maximum_limit', dict(scenario=scenarios))
        # )
        # optimization_problem.define_constraint(
        #     ('variable', 1, dict(
        #         name='der_reactive_power_vector',
        #         scenario=scenarios, timestep=self.timesteps, der=self.ders
        #     )),
        #     '<=',
        #     ('constant', 'der_reactive_power_vector_maximum_limit', dict(scenario=scenarios))
        # )
        # optimization_problem.define_constraint(
        #     ('variable', 1, dict(
        #         name='der_active_power_vector',
        #         scenario=scenarios, timestep=self.timesteps, der=self.ders
        #     )),
        #     '>=',
        #     ('constant', 'der_active_power_vector_minimum_limit', dict(scenario=scenarios))
        # )
        # optimization_problem.define_constraint(
        #     ('variable', 1, dict(
        #         name='der_reactive_power_vector',
        #         scenario=scenarios, timestep=self.timesteps, der=self.ders
        #     )),
        #     '>=',
        #     ('constant', 'der_reactive_power_vector_minimum_limit', dict(scenario=scenarios))
        # )
        # ===============================================================

        optimization_problem.define_parameter(
            'flexible_generator_mapping_matrix',
            sp.block_diag([self.flexible_generator_map.values] * len(self.timesteps))
        )

        optimization_problem.define_parameter(
            'non_flexible_der_variable_set_to_zero',
            sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
        )
        # =====================================
        optimization_problem.define_parameter(
            'minus_electric_grid_active_power_cost_flexible_der',
            sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            @ np.transpose(np.array([price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values])
                           * -1.0 * self.timestep_interval_hours  # In Wh.
                           @ sp.block_diag(
                [np.array([np.real(
                    self.linear_electric_grid_model_set.electric_grid_model.der_power_vector_reference)])] * len(
                    self.timesteps)
            ))
        )
        optimization_problem.define_parameter(
            'minus_electric_grid_reactive_power_cost_flexible_der',
            sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            @ np.transpose(np.array([price_data.price_timeseries.loc[:, ('reactive_power', 'source', 'source')].values])
                           * -1.0 * self.timestep_interval_hours  # In Wh.
                           @ sp.block_diag(
                [np.array([np.imag(
                    self.linear_electric_grid_model_set.electric_grid_model.der_power_vector_reference)])] * len(
                    self.timesteps)
            ))
        )
        # =====================================

        optimization_problem.define_parameter(
            'loss_active_active_term_transposed',
            sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            @ sp.block_diag([
                linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
                @ sp.diags(np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                # @ self.flexible_generator_map.values.transpose()
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]).transpose()
        )
        optimization_problem.define_parameter(
            'loss_active_reactive_term_transposed',
            sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            @ sp.block_diag([
                linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
                @ sp.diags(np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                # @ self.flexible_generator_map.values.transpose()
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]).transpose()
        )
        optimization_problem.define_parameter(
            'loss_reactive_active_term_transposed',
            sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            @ sp.block_diag([
                linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
                @ sp.diags(np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                # @ self.flexible_generator_map.values.transpose()
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]).transpose()
        )
        optimization_problem.define_parameter(
            'loss_reactive_reactive_term_transposed',
            sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            @ sp.block_diag([
                linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
                @ sp.diags(np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                # @ self.flexible_generator_map.values.transpose()
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]).transpose()
        )
        optimization_problem.define_parameter(
            'voltage_active_term_transposed',
            # sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            # @
            sp.block_diag([
                sp.diags(np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference) ** -1)
                @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                @ sp.diags(np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                # @ self.flexible_generator_map.values.transpose()
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]).transpose()
        )

        optimization_problem.define_parameter(
            'voltage_reactive_term_transposed',
            # sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            # @
            sp.block_diag([
                sp.diags(np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference) ** -1)
                @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                @ sp.diags(np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                # @ self.flexible_generator_map.values.transpose()
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]).transpose()
        )

        optimization_problem.define_parameter(
            'branch_power_1_active_term_transposed',
            # sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            # @
            sp.block_diag([
                sp.diags(linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference ** -1)
                @ linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                @ sp.diags(np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                # @ self.flexible_load_map.values.transpose()
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]).transpose()
        )

        optimization_problem.define_parameter(
            'branch_power_1_reactive_term_transposed',
            # sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            # @
            sp.block_diag([
                sp.diags(linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference ** -1)
                @ linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                @ sp.diags(np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]).transpose()
        )

        optimization_problem.define_parameter(
            'branch_power_2_active_term_transposed',
            # sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            # @
            sp.block_diag([
                sp.diags(linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference ** -1)
                @ linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                @ sp.diags(np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                # @ self.flexible_load_map.values.transpose()
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]).transpose()
        )

        optimization_problem.define_parameter(
            'branch_power_2_reactive_term_transposed',
            # sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            # @
            sp.block_diag([
                sp.diags(linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference ** -1)
                @ linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                @ sp.diags(np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                # @ self.flexible_load_map.values.transpose()
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]).transpose()
        )

        # optimization_problem.define_parameter(
        #     'offer_zeros',
        #     np.zeros([len(self.timesteps) * len(self.ders), 1])
        # )
        optimization_problem.define_parameter(
            'offer_zeros',
            np.zeros([len(self.timesteps) * len(self.strategic_generator_index), 1])
        )
        # optimization_problem.define_parameter(
        #     'offer_zeros',
        #     8.0 * np.ones([len(self.timesteps) * len(self.strategic_generator_index), 1])
        # )
        optimization_problem.define_parameter(
            'power_vector_mu_zeros',
            np.zeros([len(self.timesteps) * len(self.ders), 1])
        )
        optimization_problem.define_parameter(
            'node_voltage_mu_zeros',
            np.zeros([len(self.timesteps) * len(self.nodes), 1])
        )
        optimization_problem.define_parameter(
            'branch_power_mu_zeros',
            np.zeros([len(self.timesteps) * len(self.branches), 1])
        )
        optimization_problem.define_parameter(
            'power_vector_big_m',
            big_m * sp.diags(np.ones(len(self.ders) * len(self.timesteps)))
        )
        optimization_problem.define_parameter(
            'power_vector_big_m_ones',
            big_m * np.ones([len(self.ders) * len(self.timesteps), 1])
        )
        optimization_problem.define_parameter(
            'voltage_big_m',
            big_m * sp.diags(np.ones(len(self.nodes) * len(self.timesteps)))
        )
        optimization_problem.define_parameter(
            'voltage_big_m_ones',
            big_m * np.ones([len(self.nodes) * len(self.timesteps), 1])
        )
        optimization_problem.define_parameter(
            'branch_power_big_m',
            big_m * sp.diags(np.ones(len(self.branches) * len(self.timesteps)))
        )
        optimization_problem.define_parameter(
            'branch_power_big_m_ones',
            big_m * np.ones([len(self.branches) * len(self.timesteps), 1])
        )

        optimization_problem.define_parameter(
            'der_ones',
            sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            @ sp.block_diag([np.ones([len(self.ders), 1])] * len(self.timesteps))
        )

        optimization_problem.define_parameter(
            'minus_voltage_constant_plus_voltage_maximum_limit',
            1.0 * np.transpose(np.transpose([np.concatenate([
                node_voltage_magnitude_vector_maximum.ravel()
                / np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ])]) - 1.0 * np.concatenate([
                sp.diags(np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference) ** -1)
                @ (
                        np.transpose([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector)])
                        - linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                        - linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                ) for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]))
        )

        optimization_problem.define_parameter(
            'voltage_constant_minus_voltage_limit_minimum',
            1.0 * np.transpose(-1.0 * np.transpose([np.concatenate([
                node_voltage_magnitude_vector_minimum.ravel()
                / np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ])]) + np.concatenate([
                sp.diags(np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference) ** -1)
                @ (
                        np.transpose([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector)])
                        - linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                        - linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                ) for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]))
        )

        optimization_problem.define_parameter(
            'minus_branch_power_1_constant_plus_branch_power_maximum',
            1.0 * np.transpose(np.transpose([np.concatenate([
                branch_power_magnitude_vector_maximum.ravel()
                / linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ])]) - 1.0 * np.concatenate([
                sp.diags(linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference ** -1)
                @ (
                        np.transpose([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1)])
                        - linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                        - linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                ) for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]))
        )

        optimization_problem.define_parameter(
            'branch_power_1_constant_minus_branch_power_minimum',
            1.0 * np.transpose(np.transpose([np.concatenate([
                branch_power_magnitude_vector_maximum.ravel()
                / linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ])]) + np.concatenate([
                sp.diags(linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference ** -1)
                @ (
                        np.transpose([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1)])
                        - linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                        - linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                ) for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]))
        )

        optimization_problem.define_parameter(
            'minus_branch_power_2_constant_plus_branch_power_maximum',
            1.0 * np.transpose(np.transpose([np.concatenate([
                branch_power_magnitude_vector_maximum.ravel()
                / linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ])]) - 1.0 * np.concatenate([
                sp.diags(linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference ** -1)
                @ (
                        np.transpose([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2)])
                        - linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                        - linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                ) for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]))
        )

        optimization_problem.define_parameter(
            'branch_power_2_constant_minus_branch_power_minimum',
            1.0 * np.transpose(np.transpose([np.concatenate([
                branch_power_magnitude_vector_maximum.ravel()
                / linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference
                for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ])]) + np.concatenate([
                sp.diags(linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference ** -1)
                @ (
                        np.transpose([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2)])
                        - linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                        - linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                ) for linear_electric_grid_model in
                self.linear_electric_grid_model_set.linear_electric_grid_models.values()
            ]))
        )

        optimization_problem.define_parameter(
            'non_strategic_der_active_power_marginal_cost',
            sp.block_diag([self.non_flexible_der_set_to_zero_map.values] * len(self.timesteps))
            # @ sp.block_diag([self.strategic_generator_set_to_zero_map.values] * len(self.der_model_set.timesteps))
            @ np.transpose(np.concatenate([[[
                                                self.der_model_set.der_models[der_name].marginal_cost
                                                * self.timestep_interval_hours  # In Wh.
                                                * self.der_model_set.der_models[der_name].active_power_nominal
                                                for der_type, der_name in self.der_model_set.electric_ders
                                            ] * len(self.der_model_set.timesteps)]], axis=1))
        )

        optimization_problem.define_parameter(
            'der_reactive_power_marginal_cost_transposed',
            np.concatenate([[[
                                 0.0
                                 # self.der_models[der_name].marginal_cost
                                 # * timestep_interval_hours  # In Wh.
                                 # * self.der_models[der_name].reactive_power_nominal
                                 for der_type, der_name in self.der_model_set.electric_ders
                             ] * len(self.der_model_set.timesteps)]], axis=1).transpose()
        )

    def define_optimization_constraints(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            scenarios: typing.Union[list, pd.Index] = None
    ):
        if scenarios is None:
            scenarios = [None]

        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='active_loss_mu', scenario=scenarios, timestep=self.timesteps
            )),
            '==',
            ('constant', 'electric_grid_loss_active_cost', dict(scenario=scenarios)),
            broadcast=['scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='reactive_loss_mu', scenario=scenarios, timestep=self.timesteps
            )),
            '==',
            ('constant', 'electric_grid_loss_reactive_cost', dict(scenario=scenarios)),
            broadcast=['scenario']
        )
        # optimization_problem.define_constraint(
        #     ('variable', 1.0, dict(
        #         name='der_strategic_offer', scenario=scenarios, timestep=self.timesteps,
        #         der=self.ders
        #     )),
        #     '==',
        #     ('constant', 'non_strategic_der_active_power_marginal_cost', dict(scenario=scenarios)),
        #     # ('variable', 'flexible_generator_mapping_matrix', dict(
        #     #     name='flexible_generator_strategic_offer', timestep=self.timesteps, scenario=scenarios,
        #     #     fg=self.strategic_generator_index
        #     # )),
        #     broadcast=['scenario']
        # )

        optimization_problem.define_constraint(
            # ('variable', 1.0, dict(
            #     name='der_strategic_offer', scenario=scenarios, timestep=self.timesteps,
            #     der=self.ders
            # )),
            ('constant', 'non_strategic_der_active_power_marginal_cost', dict(scenario=scenarios)),
            ('constant', 'minus_electric_grid_active_power_cost_flexible_der', dict(scenario=scenarios)),
            ('variable', 'non_flexible_der_variable_set_to_zero', dict(
                name='active_equal_to_reactive_power_mu', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            # -----------------------------------
            ('variable', 'non_flexible_der_variable_set_to_zero', dict(
                name='der_active_power_vector_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            ('variable', 'voltage_active_term_transposed', dict(
                name='node_voltage_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            ('variable', 'branch_power_1_active_term_transposed', dict(
                name='branch_1_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            ('variable', 'branch_power_2_active_term_transposed', dict(
                name='branch_2_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            ('variable', 'loss_active_active_term_transposed', dict(
                name='active_loss_mu', scenario=scenarios, timestep=self.timesteps
            )),
            ('variable', 'loss_reactive_active_term_transposed', dict(
                name='reactive_loss_mu', scenario=scenarios, timestep=self.timesteps
            )),
            '==',
            # ------------------
            ('variable', 'non_flexible_der_variable_set_to_zero', dict(
                name='der_active_power_vector_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            ('variable', 'voltage_active_term_transposed', dict(
                name='node_voltage_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            ('variable', 'branch_power_1_active_term_transposed', dict(
                name='branch_1_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            ('variable', 'branch_power_2_active_term_transposed', dict(
                name='branch_2_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            broadcast=['scenario']
        )

        optimization_problem.define_constraint(
            ('constant', 'der_reactive_power_marginal_cost_transposed', dict(scenario=scenarios)),
            ('constant', 'minus_electric_grid_reactive_power_cost_flexible_der', dict(scenario=scenarios)),
            # -------------------------------------
            ('variable', 'non_flexible_der_variable_set_to_zero', dict(
                name='der_reactive_power_vector_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            ('variable', 'voltage_reactive_term_transposed', dict(
                name='node_voltage_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            ('variable', 'branch_power_1_reactive_term_transposed', dict(
                name='branch_1_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            ('variable', 'branch_power_2_reactive_term_transposed', dict(
                name='branch_2_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            ('variable', 'loss_active_reactive_term_transposed', dict(
                name='active_loss_mu', scenario=scenarios, timestep=self.timesteps
            )),
            ('variable', 'loss_reactive_reactive_term_transposed', dict(
                name='reactive_loss_mu', scenario=scenarios, timestep=self.timesteps
            )),
            '==',
            ('variable', 'non_flexible_der_variable_set_to_zero', dict(
                name='active_equal_to_reactive_power_mu', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            # ---------------------------------
            ('variable', 'non_flexible_der_variable_set_to_zero', dict(
                name='der_reactive_power_vector_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            ('variable', 'voltage_reactive_term_transposed', dict(
                name='node_voltage_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            ('variable', 'branch_power_1_reactive_term_transposed', dict(
                name='branch_1_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            ('variable', 'branch_power_2_reactive_term_transposed', dict(
                name='branch_2_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            broadcast=['scenario']
        )

        # Defining the complementary conditions:
        if len(self.nodes) >= 0:
            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='node_voltage_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                    node=self.nodes
                )),
                '>=',
                ('constant', 'node_voltage_mu_zeros', dict(scenario=scenarios)),
                broadcast=['scenario']
            )

            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='node_voltage_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                    node=self.nodes
                )),
                '>=',
                ('constant', 'node_voltage_mu_zeros', dict(scenario=scenarios)),
                broadcast=['scenario']
            )

        if len(self.branches) >= 0:
            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='branch_1_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                    branch=self.branches
                )),
                '>=',
                ('constant', 'branch_power_mu_zeros', dict(scenario=scenarios)),
                broadcast=['scenario']
            )

            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='branch_1_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                    branch=self.branches
                )),
                '>=',
                ('constant', 'branch_power_mu_zeros', dict(scenario=scenarios)),
                broadcast=['scenario']
            )

        if len(self.branches) >= 0:
            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='branch_2_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                    branch=self.branches
                )),
                '>=',
                ('constant', 'branch_power_mu_zeros', dict(scenario=scenarios)),
                broadcast=['scenario']
            )

            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='branch_2_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                    branch=self.branches
                )),
                '>=',
                ('constant', 'branch_power_mu_zeros', dict(scenario=scenarios)),
                broadcast=['scenario']
            )

        # """
        if len(self.ders) >= 0:
            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='der_active_power_vector_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                    der=self.ders
                )),
                '>=',
                ('constant', 'power_vector_mu_zeros', dict(scenario=scenarios)),
                broadcast=['scenario']
            )

            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='der_active_power_vector_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                    der=self.ders
                )),
                '>=',
                ('constant', 'power_vector_mu_zeros', dict(scenario=scenarios)),
                broadcast=['scenario']
            )
            # --------------------------------------------------------
            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='der_reactive_power_vector_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                    der=self.ders
                )),
                '>=',
                ('constant', 'power_vector_mu_zeros', dict(scenario=scenarios)),
                broadcast=['scenario']
            )

            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='der_reactive_power_vector_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                    der=self.ders
                )),
                '>=',
                ('constant', 'power_vector_mu_zeros', dict(scenario=scenarios)),
                broadcast=['scenario']
            )

        # """
        # Complementarities
        """
        optimization_problem.define_constraint(
            ('constant', 'voltage_limit_maximum', dict(scenario=scenarios)),
            ('variable', -1.0, dict(
                name='node_voltage_magnitude_vector', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            '<=',
            ('variable', 'voltage_big_m', dict(
                name='node_voltage_mu_maximum_binary', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            broadcast=['scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='node_voltage_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            ('variable', 'voltage_big_m', dict(
                name='node_voltage_mu_maximum_binary', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            '<=',
            ('constant', 'voltage_big_m_ones', dict(scenario=scenarios)),
            broadcast=['scenario']
        )

        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='node_voltage_magnitude_vector', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            '<=',
            ('constant', 'voltage_limit_minimum', dict(scenario=scenarios)),
            ('variable', 'voltage_big_m', dict(
                name='node_voltage_mu_minimum_binary', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            broadcast=['scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='node_voltage_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            ('variable', 'voltage_big_m', dict(
                name='node_voltage_mu_minimum_binary', scenario=scenarios, timestep=self.timesteps,
                node=self.nodes
            )),
            '<=',
            ('constant', 'voltage_big_m_ones', dict(scenario=scenarios)),
            broadcast=['scenario']
        )

        optimization_problem.define_constraint(
            ('constant', 'branch_power_maximum', dict(scenario=scenarios)),
            ('variable', -1.0, dict(
                name='branch_power_magnitude_vector_1', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            '<=',
            ('variable', 'branch_power_big_m', dict(
                name='branch_1_power_mu_maximum_binary', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            broadcast=['scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1, dict(
                name='branch_1_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            ('variable', 'branch_power_big_m', dict(
                name='branch_1_power_mu_maximum_binary', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            '<=',
            ('constant', 'branch_power_big_m_ones', dict(scenario=scenarios)),
            broadcast=['scenario']
        )

        optimization_problem.define_constraint(
            ('variable', -1.0, dict(
                name='branch_power_magnitude_vector_1', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            '<=',
            ('constant', 'branch_power_minimum', dict(scenario=scenarios)),
            ('variable', 'branch_power_big_m', dict(
                name='branch_1_power_mu_minimum_binary', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            broadcast=['scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1, dict(
                name='branch_1_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            ('variable', 'branch_power_big_m', dict(
                name='branch_1_power_mu_minimum_binary', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            '<=',
            ('constant', 'branch_power_big_m_ones', dict(scenario=scenarios)),
            broadcast=['scenario']
        )

        optimization_problem.define_constraint(
            ('constant', 'branch_power_maximum', dict(scenario=scenarios)),
            ('variable', -1.0, dict(
                name='branch_power_magnitude_vector_2', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            '<=',
            ('variable', 'branch_power_big_m', dict(
                name='branch_2_power_mu_maximum_binary', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            broadcast=['scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1, dict(
                name='branch_2_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            ('variable', 'branch_power_big_m', dict(
                name='branch_2_power_mu_maximum_binary', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            '<=',
            ('constant', 'branch_power_big_m_ones', dict(scenario=scenarios)),
            broadcast=['scenario']
        )

        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='branch_power_magnitude_vector_2', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            '<=',
            ('constant', 'branch_power_minimum', dict(scenario=scenarios)),
            ('variable', 'branch_power_big_m', dict(
                name='branch_2_power_mu_minimum_binary', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            broadcast=['scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1, dict(
                name='branch_2_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            ('variable', 'branch_power_big_m', dict(
                name='branch_2_power_mu_minimum_binary', scenario=scenarios, timestep=self.timesteps,
                branch=self.branches
            )),
            '<=',
            ('constant', 'branch_power_big_m_ones', dict(scenario=scenarios)),
            broadcast=['scenario']
        )
    #     _____________________________________---
        optimization_problem.define_constraint(
            ('constant', 'der_active_power_vector_maximum_limit', dict(scenario=scenarios)),
            ('variable', -1.0, dict(
                name='der_active_power_vector', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            '<=',
            ('variable', 'power_vector_big_m', dict(
                name='der_active_power_vector_mu_maximum_binary', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            broadcast=['scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='der_active_power_vector_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            ('variable', 'power_vector_big_m', dict(
                name='der_active_power_vector_mu_maximum_binary', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            '<=',
            ('constant', 'power_vector_big_m_ones', dict(scenario=scenarios)),
            broadcast=['scenario']
        )

        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='der_active_power_vector', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            '<=',
            ('constant', 'der_active_power_vector_minimum_limit', dict(scenario=scenarios)),
            ('variable', 'power_vector_big_m', dict(
                name='der_active_power_vector_mu_minimum_binary', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            broadcast=['scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='der_active_power_vector_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            ('variable', 'power_vector_big_m', dict(
                name='der_active_power_vector_mu_minimum_binary', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            '<=',
            ('constant', 'power_vector_big_m_ones', dict(scenario=scenarios)),
            broadcast=['scenario']
        )

        optimization_problem.define_constraint(
            ('constant', 'der_reactive_power_vector_maximum_limit', dict(scenario=scenarios)),
            ('variable', -1.0, dict(
                name='der_reactive_power_vector', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            '<=',
            ('variable', 'power_vector_big_m', dict(
                name='der_reactive_power_vector_mu_maximum_binary', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            broadcast=['scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='der_reactive_power_vector_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            ('variable', 'power_vector_big_m', dict(
                name='der_reactive_power_vector_mu_maximum_binary', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            '<=',
            ('constant', 'power_vector_big_m_ones', dict(scenario=scenarios)),
            broadcast=['scenario']
        )

        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='der_reactive_power_vector', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            '<=',
            ('constant', 'der_reactive_power_vector_minimum_limit', dict(scenario=scenarios)),
            ('variable', 'power_vector_big_m', dict(
                name='der_reactive_power_vector_mu_minimum_binary', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            broadcast=['scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='der_reactive_power_vector_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            ('variable', 'power_vector_big_m', dict(
                name='der_reactive_power_vector_mu_minimum_binary', scenario=scenarios, timestep=self.timesteps,
                der=self.ders
            )),
            '<=',
            ('constant', 'power_vector_big_m_ones', dict(scenario=scenarios)),
            broadcast=['scenario']
        )
        """

    def define_objective_function(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            scenarios: typing.Union[list, pd.Index] = None
    ):
        if scenarios is None:
            scenarios = [None]
        # Defining strategic objective function:
        kkt = True
        if not kkt:
            optimization_problem.define_objective(
                ('variable', 'minus_voltage_constant_plus_voltage_maximum_limit', dict(
                    name='node_voltage_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                    node=self.nodes
                ))
            )
            optimization_problem.define_objective(
                ('variable', 'voltage_constant_minus_voltage_limit_minimum', dict(
                    name='node_voltage_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                    node=self.nodes
                ))
            )
            optimization_problem.define_objective(
                ('variable', 'minus_branch_power_1_constant_plus_branch_power_maximum', dict(
                    name='branch_1_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                    branch=self.branches
                ))
            )
            optimization_problem.define_objective(
                ('variable', 'minus_branch_power_2_constant_plus_branch_power_maximum', dict(
                    name='branch_2_power_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                    branch=self.branches
                ))
            )
            optimization_problem.define_objective(
                ('variable', 'branch_power_1_constant_minus_branch_power_minimum', dict(
                    name='branch_1_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                    branch=self.branches
                ))
            )
            optimization_problem.define_objective(
                ('variable', 'branch_power_2_constant_minus_branch_power_minimum', dict(
                    name='branch_2_power_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                    branch=self.branches
                ))
            )
            optimization_problem.define_objective(
                ('variable', 'minus_der_reactive_power_vector_minimum_limit_transposed', dict(
                    name='der_reactive_power_vector_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                    der=self.ders
                ))
            )
            optimization_problem.define_objective(
                ('variable', 'der_reactive_power_vector_maximum_limit_transposed', dict(
                    name='der_reactive_power_vector_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                    der=self.ders
                ))
            )

            optimization_problem.define_objective(
                ('variable', 'minus_der_reactive_power_vector_minimum_limit_transposed', dict(
                    name='der_active_power_vector_mu_minimum', scenario=scenarios, timestep=self.timesteps,
                    der=self.ders
                ))
            )
            optimization_problem.define_objective(
                ('variable', 'der_reactive_power_vector_maximum_limit_transposed', dict(
                    name='der_active_power_vector_mu_maximum', scenario=scenarios, timestep=self.timesteps,
                    der=self.ders
                ))
            )
        else:
            optimization_problem.define_variable(
                'kkt_objective', scenario=scenarios
            )
            optimization_problem.define_parameter(
                'kkt_value', 1
            )
            optimization_problem.define_constraint(
                ('variable', 1, dict(name='kkt_objective', scenario=scenarios)),
                '==',
                ('constant', 'kkt_value', dict())
            )
            optimization_problem.define_objective(
                ('variable', 1, dict(name='kkt_objective', scenario=scenarios))
            )
