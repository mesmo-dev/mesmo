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
                    self.branch_node_incidence_matrix[node_index, branch_index] += -1.0
                elif node_name == thermal_grid_data.thermal_grid_lines.at[branch, 'node_2_name']:
                    self.branch_node_incidence_matrix[node_index, branch_index] += +1.0
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
        self.enthalpy_difference_distribution_water = (
            float(thermal_grid_data.thermal_grid['enthalpy_difference_distribution_water'])
        )

    def define_optimization_variables(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            timesteps=pd.Index([0], name='timestep')
    ):
        """Define decision variables for given `optimization_problem`."""

        optimization_problem.der_thermal_power_vector = (
            pyo.Var(timesteps.to_list(), self.ders.to_list())
        )
        optimization_problem.branch_flow_vector = (
            pyo.Var(timesteps.to_list(), self.branches.to_list())
        )
        optimization_problem.source_flow = (
            pyo.Var(timesteps.to_list())
        )

    def define_optimization_constraints(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            timesteps=pd.Index([0], name='timestep')
    ):
        """Define constraints to express the thermal grid model equations for given `optimization_problem`."""

        optimization_problem.thermal_grid_constraints = pyo.ConstraintList()
        for timestep in timesteps:
            for node_index, node_type in enumerate(self.nodes.get_level_values('node_type')):
                if node_type == 'source':
                    optimization_problem.thermal_grid_constraints.add(
                        -1.0 * optimization_problem.source_flow[timestep]
                        ==
                        sum(
                            self.branch_node_incidence_matrix[node_index, branch_index]
                            * optimization_problem.branch_flow_vector[timestep, branch]
                            for branch_index, branch in enumerate(self.branches)
                        )
                    )
                else:
                    optimization_problem.thermal_grid_constraints.add(
                        sum(
                            self.der_node_incidence_matrix[node_index, der_index]
                            * optimization_problem.der_thermal_power_vector[timestep, der]
                            * self.enthalpy_difference_distribution_water
                            / fledge.config.water_density
                            for der_index, der in enumerate(self.ders)
                        )
                        ==
                        sum(
                            self.branch_node_incidence_matrix[node_index, branch_index]
                            * optimization_problem.branch_flow_vector[timestep, branch]
                            for branch_index, branch in enumerate(self.branches)
                        )
                    )

    def get_optimization_results(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            timesteps=pd.Index([0], name='timestep')
    ):

        # Instantiate results variables.
        der_thermal_power_vector = (
            pd.DataFrame(columns=self.ders, index=timesteps, dtype=np.float)
        )
        branch_flow_vector = (
            pd.DataFrame(columns=self.branches, index=timesteps, dtype=np.float)
        )
        source_flow = (
            pd.DataFrame(columns=['total'], index=timesteps, dtype=np.float)
        )

        # Obtain results.
        for timestep in timesteps:

            for der in self.ders:
                der_thermal_power_vector.at[timestep, der] = (
                    optimization_problem.der_thermal_power_vector[timestep, der].value
                )

            for branch in self.branches:
                branch_flow_vector.at[timestep, branch] = (
                    optimization_problem.branch_flow_vector[timestep, branch].value
                )

            source_flow.at[timestep, 'total'] = (
                optimization_problem.source_flow[timestep].value
            )

        return (
            der_thermal_power_vector,
            branch_flow_vector,
            source_flow
        )


class ThermalPowerFlowSolution(object):
    """Thermal grid power flow solution object."""

    der_thermal_power_vector: np.ndarray
    der_flow_vector: np.ndarray
    branch_flow_vector: np.ndarray
    branch_velocity_vector: np.ndarray
    branch_reynold_vector: np.ndarray
    branch_friction_factor_vector: np.ndarray
    branch_head_vector: np.ndarray
    node_head_vector: np.ndarray

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

        # Obtain DER volume flow vector.
        self.der_flow_vector = (
            self.der_thermal_power_vector
            / fledge.config.water_density
            / thermal_grid_model.enthalpy_difference_distribution_water
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

        # Obtain nodal head vector.
        self.node_head_vector = (
            scipy.sparse.linalg.spsolve(
                np.transpose(
                    thermal_grid_model.branch_node_incidence_matrix[
                        fledge.utils.get_index(thermal_grid_model.nodes, node_type='no_source'),
                        :
                    ]
                ),
                -1.0 * self.branch_head_vector
            )
        )
