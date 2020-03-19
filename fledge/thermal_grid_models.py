"""Thermal grid models module."""

from multimethod import multimethod
import numpy as np
import pandas as pd
import pyomo.core
import pyomo.environ as pyo
import scipy.sparse
import scipy.sparse.linalg

import fledge.config
import fledge.database_interface
import fledge.utils

logger = fledge.config.get_logger(__name__)


class ThermalGridModel(object):
    """Thermal grid model object."""

    node_names: pd.Index
    line_names: pd.Index
    der_names: pd.Index
    der_types: pd.Index
    nodes: pd.Index
    branches: pd.Index
    ders = pd.Index
    branch_node_incidence_matrix: scipy.sparse.spmatrix
    der_node_incidence_matrix: scipy.sparse.spmatrix
    der_thermal_power_vector_nominal: np.ndarray

    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain thermal grid data.
        thermal_grid_data = fledge.database_interface.ThermalGridData(scenario_name)

        # Obtain node / line / DER names.
        self.node_names = pd.Index(thermal_grid_data.thermal_grid_nodes['node_name'])
        self.line_names = pd.Index(thermal_grid_data.thermal_grid_lines['line_name'])
        self.der_names = pd.Index(thermal_grid_data.thermal_grid_ders['der_name'])
        self.der_types = pd.Index(thermal_grid_data.thermal_grid_ders['der_type']).unique()

        # Obtain node / branch / DER index set.
        self.nodes = pd.MultiIndex.from_frame(thermal_grid_data.thermal_grid_nodes[['node_type', 'node_name']])
        self.branches = self.line_names
        self.ders = pd.MultiIndex.from_frame(thermal_grid_data.thermal_grid_ders[['der_type', 'der_name']])

        # Define branch to node incidence matrix.
        self.branch_node_incidence_matrix = (
            scipy.sparse.dok_matrix((len(self.nodes), len(self.branches)), dtype=np.int)
        )
        for node_index, node_name in enumerate(self.nodes.get_level_values('node_name')):
            for branch_index, branch in enumerate(self.branches):
                if node_name == thermal_grid_data.thermal_grid_lines.at[branch, 'node_1_name']:
                    self.branch_node_incidence_matrix[node_index, branch_index] += +1.0
                elif node_name == thermal_grid_data.thermal_grid_lines.at[branch, 'node_2_name']:
                    self.branch_node_incidence_matrix[node_index, branch_index] += -1.0
        self.branch_node_incidence_matrix = self.branch_node_incidence_matrix.tocsr()

        # Define DER to node incidence matrix.
        self.der_node_incidence_matrix = (
            scipy.sparse.dok_matrix((len(self.nodes), len(self.ders)), dtype=np.int)
        )
        for node_index, node_name in enumerate(self.nodes.get_level_values('node_name')):
            for der_index, der_name in enumerate(self.der_names):
                if node_name == thermal_grid_data.thermal_grid_ders.at[der_name, 'node_name']:
                    self.der_node_incidence_matrix[node_index, der_index] = 1.0
        self.der_node_incidence_matrix = self.der_node_incidence_matrix.tocsr()

        # Obtain DER nominal thermal power vector.
        self.der_thermal_power_vector_nominal = (
            thermal_grid_data.thermal_grid_ders.loc[:, 'thermal_power_nominal'].values
        )

        # Obtain line parameters.
        self.line_length_vector = thermal_grid_data.thermal_grid_lines['length'].values
        self.line_diameter_vector = thermal_grid_data.thermal_grid_lines['diameter'].values
        self.line_roughness_vector = thermal_grid_data.thermal_grid_lines['absolute_roughness'].values

        # Obtain other system parameters.
        self.ets_head_loss = (
            np.float(thermal_grid_data.thermal_grid['ets_head_loss'])
        )
        self.enthalpy_difference_distribution_water = (
            np.float(thermal_grid_data.thermal_grid['enthalpy_difference_distribution_water'])
        )
        self.pump_efficiency_secondary_pump = (  # TODO: Rename to `secondary_pump_efficiency`.
            np.float(thermal_grid_data.thermal_grid['pump_efficiency_secondary_pump'])
        )
        self.cooling_plant_efficiency = 5.0  # TODO: Define cooling plant model.


class ThermalPowerFlowSolution(object):
    """Thermal grid power flow solution object."""

    der_thermal_power_vector: np.ndarray
    der_flow_vector: np.ndarray
    source_flow: np.float
    branch_flow_vector: np.ndarray
    branch_velocity_vector: np.ndarray
    branch_reynold_vector: np.ndarray
    branch_friction_factor_vector: np.ndarray
    branch_head_vector: np.ndarray
    source_head: np.float
    node_head_vector: np.ndarray
    source_electric_power_secondary_pump: np.float
    source_electric_power_cooling_plant: np.float

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain thermal grid model.
        thermal_grid_model = ThermalGridModel(scenario_name)

        self.__init__(
            thermal_grid_model
        )

    @multimethod
    def __init__(
            self,
            thermal_grid_model: ThermalGridModel
    ):

        # Obtain DER thermal power vector.
        der_thermal_power_vector = thermal_grid_model.der_thermal_power_vector_nominal

        self.__init__(
            thermal_grid_model,
            der_thermal_power_vector
        )

    @multimethod
    def __init__(
            self,
            thermal_grid_model: ThermalGridModel,
            der_thermal_power_vector: np.ndarray
    ):

        # Obtain DER thermal power vector.
        self.der_thermal_power_vector = der_thermal_power_vector

        # Obtain DER / source volume flow vector.
        self.der_flow_vector = (
            self.der_thermal_power_vector
            / fledge.config.water_density
            / thermal_grid_model.enthalpy_difference_distribution_water
        )
        self.source_flow = (
            -1.0 * np.sum(self.der_flow_vector)
        )

        # Obtain branch volume flow vector.
        self.branch_flow_vector = (
            scipy.sparse.linalg.spsolve(
                thermal_grid_model.branch_node_incidence_matrix[
                    fledge.utils.get_index(thermal_grid_model.nodes, node_type='no_source'),
                    :
                ],
                thermal_grid_model.der_node_incidence_matrix[
                    fledge.utils.get_index(thermal_grid_model.nodes, node_type='no_source'),
                    :
                ]
                @ np.transpose([self.der_flow_vector])
            )
        )

        # Obtain branch velocity vector.
        self.branch_velocity_vector = (
            4.0 * self.branch_flow_vector
            / (np.pi * thermal_grid_model.line_diameter_vector ** 2)
        )

        # Obtain branch Reynolds coefficient vector.
        self.branch_reynold_vector = (
            np.abs(self.branch_velocity_vector)
            * thermal_grid_model.line_diameter_vector
            / fledge.config.water_kinematic_viscosity
        )

        # Obtain branch friction factor vector.
        @np.vectorize
        def get_friction_factor(
                reynold,
                roughness,
                diameter
        ):

            # No flow.
            if reynold == 0:
                friction_factor = 0

            # Laminar Flow, based on Hagen-Poiseuille velocity profile, analytical correlation.
            elif 0 < reynold < 4000:
                friction_factor = 64 / reynold

            # Turbulent flow, Swamee-Jain formula, approximating correlation of Colebrook-White equation.
            elif 4000 <= reynold:
                if not (reynold <= 100000000 and 0.000001 <= ((roughness / 1000) / diameter) <= 0.01):
                    logger.warn("Exceeding validity range of Swamee-Jain formula for calculation of friction factor.")
                friction_factor = 1.325 / (
                    np.log(
                        (roughness / 1000) / (3.7 * diameter) + 5.74 / (reynold ** 0.9)
                    )
                ) ** 2

            else:
                logger.error(f"Invalid Reynolds coefficient: {reynold}")
                raise ValueError

            return friction_factor

        self.branch_friction_factor_vector = (
            get_friction_factor(
                self.branch_reynold_vector,
                thermal_grid_model.line_roughness_vector,
                thermal_grid_model.line_diameter_vector
            )
        )

        # Obtain branch head loss vector.
        # - Darcy-Weisbach Equation.
        self.branch_head_vector = (
            self.branch_friction_factor_vector
            * self.branch_flow_vector
            * np.abs(self.branch_flow_vector)  # TODO: Check if absolute value needed.
            * 8.0 * thermal_grid_model.line_length_vector
            / (
                fledge.config.gravitational_acceleration
                * thermal_grid_model.line_diameter_vector ** 5
                * np.pi ** 2
            )
        )

        # Obtain node / source head vector.
        node_head_vector_no_source = (
            scipy.sparse.linalg.spsolve(
                np.transpose(
                    thermal_grid_model.branch_node_incidence_matrix[
                        fledge.utils.get_index(thermal_grid_model.nodes, node_type='no_source'),
                        :
                    ]
                ),
                self.branch_head_vector
            )
        )
        self.source_head = (
            np.max(np.abs(node_head_vector_no_source))
        )
        self.node_head_vector = np.zeros(len(thermal_grid_model.nodes), dtype=np.float)
        self.node_head_vector[fledge.utils.get_index(thermal_grid_model.nodes, node_type='no_source')] = (
            node_head_vector_no_source
        )

        # Obtain secondary pump / cooling plant electric power.
        self.source_electric_power_secondary_pump = (
            (
                2.0 * self.source_head
                + thermal_grid_model.ets_head_loss
            )
            * self.source_flow
            * fledge.config.water_density
            * fledge.config.gravitational_acceleration
            / thermal_grid_model.pump_efficiency_secondary_pump
        )
        self.source_electric_power_cooling_plant = (
            self.source_flow
            * thermal_grid_model.enthalpy_difference_distribution_water
            * fledge.config.water_density
            / thermal_grid_model.cooling_plant_efficiency
        )


class LinearThermalGridModel(object):
    """Linear thermal grid model object."""

    thermal_grid_model: ThermalGridModel
    thermal_power_flow_solution: ThermalPowerFlowSolution
    sensitivity_branch_flow_by_der_power: scipy.sparse.spmatrix
    sensitivity_node_head_by_der_power: scipy.sparse.spmatrix
    sensitivity_pump_power_by_der_power: scipy.sparse.spmatrix

    def __init__(
            self,
            thermal_grid_model: ThermalGridModel,
            thermal_power_flow_solution: ThermalPowerFlowSolution,
    ):

        # Store thermal grid model.
        self.thermal_grid_model = thermal_grid_model

        # Store thermal power flow solution.
        self.thermal_power_flow_solution = thermal_power_flow_solution

        # Obtain inverse / transpose incidence matrices.
        node_index_no_source = (
            fledge.utils.get_index(self.thermal_grid_model.nodes, node_type='no_source')  # Define shorthand.
        )
        branch_node_incidence_matrix_tranpose_inverse = (
            scipy.sparse.dok_matrix(
                (len(self.thermal_grid_model.nodes), len(self.thermal_grid_model.branches)),
                dtype=np.float
            )
        )
        branch_node_incidence_matrix_tranpose_inverse[np.ix_(
            node_index_no_source,
            range(len(self.thermal_grid_model.branches))
        )] = (
            scipy.sparse.linalg.inv(
                self.thermal_grid_model.branch_node_incidence_matrix[node_index_no_source, :].transpose()
            )
        )
        branch_node_incidence_matrix_tranpose_inverse = branch_node_incidence_matrix_tranpose_inverse.tocsr()
        branch_node_incidence_matrix_inverse = (
            scipy.sparse.dok_matrix(
                (len(self.thermal_grid_model.branches), len(self.thermal_grid_model.nodes)),
                dtype=np.float
            )
        )
        branch_node_incidence_matrix_inverse[np.ix_(
            range(len(self.thermal_grid_model.branches)),
            node_index_no_source
        )] = (
            scipy.sparse.linalg.inv(
                self.thermal_grid_model.branch_node_incidence_matrix[node_index_no_source, :].tocsc()
            )
        )
        branch_node_incidence_matrix_inverse = branch_node_incidence_matrix_inverse.tocsr()
        der_node_incidence_matrix_transpose = np.transpose(self.thermal_grid_model.der_node_incidence_matrix)

        # Obtain sensitivity matrices.
        self.sensitivity_branch_flow_by_der_power = (
            branch_node_incidence_matrix_inverse
            @ self.thermal_grid_model.der_node_incidence_matrix
            / fledge.config.water_density
            / self.thermal_grid_model.enthalpy_difference_distribution_water
        )
        self.sensitivity_node_head_by_der_power = (
            branch_node_incidence_matrix_tranpose_inverse
            @ scipy.sparse.diags(
                thermal_power_flow_solution.branch_flow_vector
                * thermal_power_flow_solution.branch_friction_factor_vector
                * 8.0 * self.thermal_grid_model.line_length_vector
                / (
                    fledge.config.gravitational_acceleration
                    * self.thermal_grid_model.line_diameter_vector ** 5
                    * np.pi ** 2
                )
            )
            @ self.sensitivity_branch_flow_by_der_power
        )
        self.sensitivity_pump_power_by_der_power = (
            (-1.0 * thermal_power_flow_solution.der_flow_vector)
            @ (-2.0 * der_node_incidence_matrix_transpose)
            @ self.sensitivity_node_head_by_der_power
            * fledge.config.water_density
            * fledge.config.gravitational_acceleration
            / self.thermal_grid_model.pump_efficiency_secondary_pump
        )

    def define_optimization_variables(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            timesteps=pd.Index([0], name='timestep')
    ):
        """Define decision variables for given `optimization_problem`."""

        optimization_problem.der_thermal_power_vector = (
            pyo.Var(timesteps.to_list(), self.thermal_grid_model.ders.to_list())
        )
        optimization_problem.node_head_vector = (
            pyo.Var(timesteps.to_list(), self.thermal_grid_model.nodes.to_list())
        )
        optimization_problem.branch_flow_vector = (
            pyo.Var(timesteps.to_list(), self.thermal_grid_model.branches.to_list())
        )
        optimization_problem.pump_power = (
            pyo.Var(timesteps.to_list())
        )

    def define_optimization_constraints(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            timesteps=pd.Index([0], name='timestep')
    ):

        # Define constraints.
        optimization_problem.thermal_grid_constraints = pyo.ConstraintList()
        for timestep in timesteps:

            for node_index, node in enumerate(self.thermal_grid_model.nodes):
                optimization_problem.thermal_grid_constraints.add(
                    # TODO: First column should not be NaN.
                    optimization_problem.node_head_vector[timestep, node]
                    ==
                    sum(
                        self.sensitivity_node_head_by_der_power[node_index, der_index]
                        * optimization_problem.der_thermal_power_vector[timestep, der]
                        for der_index, der in enumerate(self.thermal_grid_model.ders)
                    )
                )

            for branch_index, branch in enumerate(self.thermal_grid_model.branches):
                optimization_problem.thermal_grid_constraints.add(
                    optimization_problem.branch_flow_vector[timestep, branch]
                    ==
                    sum(
                        self.sensitivity_branch_flow_by_der_power[branch_index, der_index]
                        * optimization_problem.der_thermal_power_vector[timestep, der]
                        for der_index, der in enumerate(self.thermal_grid_model.ders)
                    )
                )

            optimization_problem.thermal_grid_constraints.add(
                optimization_problem.pump_power[timestep]
                ==
                sum(
                    self.sensitivity_pump_power_by_der_power[der_index]
                    * optimization_problem.der_thermal_power_vector[timestep, der]
                    for der_index, der in enumerate(self.thermal_grid_model.ders)
                )
                + sum(
                    -1.0 * self.thermal_power_flow_solution.der_flow_vector[der_index]
                    * self.thermal_grid_model.ets_head_loss
                    * fledge.config.water_density
                    * fledge.config.gravitational_acceleration
                    / self.thermal_grid_model.pump_efficiency_secondary_pump
                    for der_index, der in enumerate(self.thermal_grid_model.ders)
                )
            )

    def define_optimization_limits(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            node_head_vector_minimum: np.ndarray = None,
            branch_flow_vector_maximum: np.ndarray = None,
            timesteps=pd.Index([0], name='timestep')
    ):

        # Node head.
        if node_head_vector_minimum is not None:
            # node_head_vector = (  # Define shorthand.
            #     lambda node:
            #     self.thermal_power_flow_solution.node_head_vector.ravel()[
            #         self.thermal_grid_model.nodes.get_loc(node)
            #     ]
            # )
            optimization_problem.node_head_vector_minimum_constraint = pyo.Constraint(
                timesteps.to_list(),
                self.thermal_grid_model.nodes.to_list(),
                rule=lambda optimization_problem, timestep, *node: (
                    optimization_problem.node_head_vector[timestep, node]
                    # + node_head_vector(node)
                    >=
                    node_head_vector_minimum.ravel()[self.thermal_grid_model.nodes.get_loc(node)]
                )
            )

        # Branch flow.
        if branch_flow_vector_maximum is not None:
            # branch_flow_vector = (  # Define shorthand.
            #     lambda branch:
            #     self.thermal_power_flow_solution.branch_flow_vector.ravel()[
            #         self.thermal_grid_model.branches.get_loc(branch)
            #     ]
            # )
            optimization_problem.branch_flow_vector_maximum_constraint = pyo.Constraint(
                timesteps.to_list(),
                self.thermal_grid_model.branches.to_list(),
                rule=lambda optimization_problem, timestep, branch: (  # Will not work if `branches` becomes MultiIndex.
                    optimization_problem.branch_flow_vector[timestep, branch]
                    # + branch_flow_vector(branch)
                    <=
                    branch_flow_vector_maximum.ravel()[self.thermal_grid_model.branches.get_loc(branch)]
                )
            )

    def define_optimization_objective(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            price_timeseries=pd.DataFrame(1.0, columns=['price_value'], index=[0]),
            timesteps=pd.Index([0], name='timestep')
    ):

        # Define objective.
        if optimization_problem.find_component('objective') is None:
            optimization_problem.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
        optimization_problem.objective.expr += (
            sum(
                price_timeseries.at[timestep, 'price_value']
                * -1.0 * optimization_problem.der_thermal_power_vector[timestep, der]
                / self.thermal_grid_model.cooling_plant_efficiency
                for der_index, der in enumerate(self.thermal_grid_model.ders)
                for timestep in timesteps
            )
        )
        optimization_problem.objective.expr += (
            sum(
                price_timeseries.at[timestep, 'price_value']
                * optimization_problem.pump_power[timestep]
                for timestep in timesteps
            )
        )

    def get_optimization_limits_duals(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            timesteps=pd.Index([0], name='timestep')
    ):

        # Instantiate dual variables.
        node_head_vector_minimum_dual = (
            pd.DataFrame(0.0, columns=self.thermal_grid_model.nodes, index=timesteps, dtype=np.float)
        )
        branch_flow_vector_maximum_dual = (
            pd.DataFrame(0.0, columns=self.thermal_grid_model.branches, index=timesteps, dtype=np.float)
        )

        # Obtain duals.
        for timestep in timesteps:

            if optimization_problem.find_component('node_head_vector_minimum_constraint') is not None:
                for node_index, node in enumerate(self.thermal_grid_model.nodes):
                    node_head_vector_minimum_dual.at[timestep, node] = (
                        optimization_problem.dual[
                            optimization_problem.node_head_vector_minimum_constraint[timestep, node]
                        ]
                    )

            if optimization_problem.find_component('branch_flow_vector_maximum_constraint') is not None:
                for branch_index, branch in enumerate(self.thermal_grid_model.branches):
                    branch_flow_vector_maximum_dual.at[timestep, branch] = (
                        optimization_problem.dual[
                            optimization_problem.branch_flow_vector_maximum_constraint[timestep, branch]
                        ]
                    )

        return (
            node_head_vector_minimum_dual,
            branch_flow_vector_maximum_dual
        )

    def get_optimization_results(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            timesteps=pd.Index([0], name='timestep'),
            in_per_unit=False,
            with_mean=False,
    ):

        # Instantiate results variables.
        der_thermal_power_vector = (
            pd.DataFrame(columns=self.thermal_grid_model.ders, index=timesteps, dtype=np.float)
        )
        branch_flow_vector = (
            pd.DataFrame(columns=self.thermal_grid_model.branches, index=timesteps, dtype=np.float)
        )
        node_head_vector = (
            pd.DataFrame(columns=self.thermal_grid_model.nodes, index=timesteps, dtype=np.float)
        )
        pump_power = (
            pd.DataFrame(columns=['total'], index=timesteps, dtype=np.float)
        )

        # Obtain results.
        for timestep in timesteps:

            for der in self.thermal_grid_model.ders:
                der_thermal_power_vector.at[timestep, der] = (
                    optimization_problem.der_thermal_power_vector[timestep, der].value
                )

            for branch in self.thermal_grid_model.branches:
                branch_flow_vector.at[timestep, branch] = (
                    optimization_problem.branch_flow_vector[timestep, branch].value
                )

            for node in self.thermal_grid_model.nodes:
                node_head_vector.at[timestep, node] = (
                    optimization_problem.node_head_vector[timestep, node].value
                )

            pump_power.at[timestep, 'total'] = (
                optimization_problem.pump_power[timestep].value
            )

        # Convert in per-unit values.
        if in_per_unit:
            der_thermal_power_vector = (
                der_thermal_power_vector
                / self.thermal_grid_model.der_thermal_power_vector_nominal.ravel()
            )
            branch_flow_vector = (
                branch_flow_vector
                / self.thermal_power_flow_solution.branch_flow_vector.ravel()
            )
            node_head_vector = (
                node_head_vector
                / self.thermal_power_flow_solution.node_head_vector.ravel()
            )
            node_head_vector.iloc[
                :,
                fledge.utils.get_index(self.thermal_grid_model.nodes, node_type='source')
            ] = 0.0  # Reset source head.
            pump_power = (
                pump_power
                / self.thermal_power_flow_solution.source_electric_power_secondary_pump
            )

        # Add mean column.
        if with_mean:
            der_thermal_power_vector['mean'] = der_thermal_power_vector.mean(axis=1)
            branch_flow_vector['mean'] = branch_flow_vector.mean(axis=1)
            node_head_vector['mean'] = node_head_vector.mean(axis=1)

        return (
            der_thermal_power_vector,
            node_head_vector,
            branch_flow_vector,
            pump_power
        )
