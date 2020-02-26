"""Linear electric grid models module."""

from multimethod import multimethod
import numpy as np
import pandas as pd
import pyomo.core
import pyomo.environ as pyo
import scipy.sparse
import scipy.sparse.linalg

import fledge.config
import fledge.electric_grid_models
import fledge.power_flow_solvers
import fledge.utils

logger = fledge.config.get_logger(__name__)


class LinearElectricGridModel(object):
    """Abstract linear electric model object, consisting of the sensitivity matrices for
    voltage / voltage magnitude / squared branch power / active loss / reactive loss by changes in nodal wye power /
    nodal delta power.

    Note:
        This abstract class only defines the expected variables of linear electric grid model objects,
        but does not implement any functionality.

    Attributes:
        electric_grid_model (fledge.electric_grid_models.ElectricGridModel): Electric grid model object.
        power_flow_solution (fledge.power_flow_solvers.PowerFlowSolution): Reference power flow solution object.
        sensitivity_voltage_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage vector
            by active wye power vector.
        sensitivity_voltage_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage
            vector by reactive wye power vector.
        sensitivity_voltage_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage vector
            by active delta power vector.
        sensitivity_voltage_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage
            vector by reactive delta power vector.
        sensitivity_voltage_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            complex voltage vector by DER active power vector.
        sensitivity_voltage_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            complex voltage vector by DER reactive power vector.
        sensitivity_voltage_magnitude_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for voltage
            magnitude vector by active wye power vector.
        sensitivity_voltage_magnitude_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by reactive wye power vector.
        sensitivity_voltage_magnitude_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by active delta power vector.
        sensitivity_voltage_magnitude_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by reactive delta power vector.
        sensitivity_voltage_magnitude_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by DER active power vector.
        sensitivity_voltage_magnitude_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by DER reactive power vector.
        sensitivity_branch_power_1_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by active wye power vector.
        sensitivity_branch_power_1_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by reactive wye power vector.
        sensitivity_branch_power_1_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by active delta power vector.
        sensitivity_branch_power_1_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by reactive delta power vector.
        sensitivity_branch_power_1_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 by DER active power vector.
        sensitivity_branch_power_1_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 by DER reactive power vector.
        sensitivity_branch_power_2_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by active wye power vector.
        sensitivity_branch_power_2_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by reactive wye power vector.
        sensitivity_branch_power_2_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by active delta power vector.
        sensitivity_branch_power_2_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by reactive delta power vector.
        sensitivity_branch_power_2_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 by DER active power vector.
        sensitivity_branch_power_2_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 by DER reactive power vector.
        sensitivity_loss_active_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by active wye power vector.
        sensitivity_loss_active_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by reactive wye power vector.
        sensitivity_loss_active_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by active delta power vector.
        sensitivity_loss_active_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by reactive delta power vector.
        sensitivity_loss_active_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by DER active power vector.
        sensitivity_loss_active_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by DER reactive power vector.
        sensitivity_loss_reactive_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by active wye power vector.
        sensitivity_loss_reactive_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by reactive wye power vector.
        sensitivity_loss_reactive_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by active delta power vector.
        sensitivity_loss_reactive_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by reactive delta power vector.
        sensitivity_loss_reactive_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by DER active power vector.
        sensitivity_loss_reactive_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by DER reactive power vector.
    """

    electric_grid_model: fledge.electric_grid_models.ElectricGridModel
    power_flow_solution: fledge.power_flow_solvers.PowerFlowSolution
    sensitivity_voltage_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_voltage_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_voltage_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_voltage_by_der_power_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_der_power_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_der_power_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_der_power_reactive: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_loss_active_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_loss_active_by_der_power_reactive: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_der_power_reactive: scipy.sparse.spmatrix

    def define_optimization_variables(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            timesteps=pd.Index([0], name='timestep')
    ):
        """Define decision variables for given `optimization_problem`."""

        # DERs.
        optimization_problem.der_active_power_vector_change = (
            pyo.Var(timesteps.to_list(), self.electric_grid_model.ders.to_list())
        )
        optimization_problem.der_reactive_power_vector_change = (
            pyo.Var(timesteps.to_list(), self.electric_grid_model.ders.to_list())
        )

        # Voltage.
        optimization_problem.voltage_magnitude_vector_change = (
            pyo.Var(timesteps.to_list(), self.electric_grid_model.nodes.to_list())
        )

        # Branch flows.
        optimization_problem.branch_power_vector_1_squared_change = (
            pyo.Var(timesteps.to_list(), self.electric_grid_model.branches.to_list())
        )
        optimization_problem.branch_power_vector_2_squared_change = (
            pyo.Var(timesteps.to_list(), self.electric_grid_model.branches.to_list())
        )

        # Loss.
        optimization_problem.loss_active_change = pyo.Var(timesteps.to_list())
        optimization_problem.loss_reactive_change = pyo.Var(timesteps.to_list())

    def define_optimization_constraints(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            timesteps=pd.Index([0], name='timestep')
    ):
        """Define constraints to express the linear electric grid model equations for given `optimization_problem`."""

        # Instantiate constraint list.
        optimization_problem.linear_electric_grid_model_constraints = pyo.ConstraintList()

        for timestep in timesteps:

            # Voltage.
            for node_index, node in enumerate(self.electric_grid_model.nodes):
                optimization_problem.linear_electric_grid_model_constraints.add(
                    optimization_problem.voltage_magnitude_vector_change[timestep, node]
                    ==
                    sum(
                        self.sensitivity_voltage_magnitude_by_der_power_active[node_index, der_index]
                        * optimization_problem.der_active_power_vector_change[timestep, der]
                        + self.sensitivity_voltage_magnitude_by_der_power_reactive[node_index, der_index]
                        * optimization_problem.der_reactive_power_vector_change[timestep, der]
                        for der_index, der in enumerate(self.electric_grid_model.ders)
                    )
                )

            # Branch flows.
            for branch_index, branch in enumerate(self.electric_grid_model.branches):
                optimization_problem.linear_electric_grid_model_constraints.add(
                    optimization_problem.branch_power_vector_1_squared_change[timestep, branch]
                    ==
                    sum(
                        self.sensitivity_branch_power_1_by_der_power_active[branch_index, der_index]
                        * optimization_problem.der_active_power_vector_change[timestep, der]
                        + self.sensitivity_branch_power_1_by_der_power_reactive[branch_index, der_index]
                        * optimization_problem.der_reactive_power_vector_change[timestep, der]
                        for der_index, der in enumerate(self.electric_grid_model.ders)
                    )
                )
                optimization_problem.linear_electric_grid_model_constraints.add(
                    optimization_problem.branch_power_vector_2_squared_change[timestep, branch]
                    ==
                    sum(
                        self.sensitivity_branch_power_2_by_der_power_active[branch_index, der_index]
                        * optimization_problem.der_active_power_vector_change[timestep, der]
                        + self.sensitivity_branch_power_2_by_der_power_reactive[branch_index, der_index]
                        * optimization_problem.der_reactive_power_vector_change[timestep, der]
                        for der_index, der in enumerate(self.electric_grid_model.ders)
                    )
                )

            # Loss.
            optimization_problem.linear_electric_grid_model_constraints.add(
                optimization_problem.loss_active_change[timestep]
                ==
                sum(
                    self.sensitivity_loss_active_by_der_power_active[0, der_index]
                    * optimization_problem.der_active_power_vector_change[timestep, der]
                    + self.sensitivity_loss_active_by_der_power_reactive[0, der_index]
                    * optimization_problem.der_reactive_power_vector_change[timestep, der]
                    for der_index, der in enumerate(self.electric_grid_model.ders)
                )
            )
            optimization_problem.linear_electric_grid_model_constraints.add(
                optimization_problem.loss_reactive_change[timestep]
                ==
                sum(
                    self.sensitivity_loss_reactive_by_der_power_active[0, der_index]
                    * optimization_problem.der_active_power_vector_change[timestep, der]
                    + self.sensitivity_loss_reactive_by_der_power_reactive[0, der_index]
                    * optimization_problem.der_reactive_power_vector_change[timestep, der]
                    for der_index, der in enumerate(self.electric_grid_model.ders)
                )
            )

    def get_optimization_results(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            timesteps=pd.Index([0], name='timestep')
    ):

        # Instantiate results variables.

        # DER.
        der_active_power_vector = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )
        der_reactive_power_vector = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )

        # Voltage.
        voltage_magnitude_vector = (
            pd.DataFrame(columns=self.electric_grid_model.nodes, index=timesteps, dtype=np.float)
        )

        # Branch flows.
        branch_power_vector_1_squared = (
            pd.DataFrame(columns=self.electric_grid_model.branches, index=timesteps, dtype=np.float)
        )
        branch_power_vector_2_squared = (
            pd.DataFrame(columns=self.electric_grid_model.branches, index=timesteps, dtype=np.float)
        )

        # Loss.
        loss_active = pd.DataFrame(columns=['total'], index=timesteps, dtype=np.float)
        loss_reactive = pd.DataFrame(columns=['total'], index=timesteps, dtype=np.float)

        # Obtain results.
        for timestep in timesteps:

            # DER.
            for der_index, der in enumerate(self.electric_grid_model.ders):
                der_active_power_vector.at[timestep, der] = (
                    optimization_problem.der_active_power_vector_change[timestep, der].value
                    + np.real(self.power_flow_solution.der_power_vector[der_index])
                )
                der_reactive_power_vector.at[timestep, der] = (
                    optimization_problem.der_reactive_power_vector_change[timestep, der].value
                    + np.imag(self.power_flow_solution.der_power_vector[der_index])
                )

            # Voltage.
            for node_index, node in enumerate(self.electric_grid_model.nodes):
                voltage_magnitude_vector.at[timestep, node] = (
                    optimization_problem.voltage_magnitude_vector_change[timestep, node].value
                    + np.abs(self.power_flow_solution.node_voltage_vector[node_index])
                )

            # Branch flows.
            for branch_index, branch in enumerate(self.electric_grid_model.branches):
                branch_power_vector_1_squared.at[timestep, branch] = (
                    optimization_problem.branch_power_vector_1_squared_change[timestep, branch].value
                    + np.abs(self.power_flow_solution.branch_power_vector_1[branch_index] ** 2)
                )
                branch_power_vector_2_squared.at[timestep, branch] = (
                    optimization_problem.branch_power_vector_2_squared_change[timestep, branch].value
                    + np.abs(self.power_flow_solution.branch_power_vector_2[branch_index] ** 2)
                )

            # Loss.
            loss_active.at[timestep, 'total'] = (
                optimization_problem.loss_active_change[timestep].value
                + np.real(self.power_flow_solution.loss)
            )
            loss_reactive.at[timestep, 'total'] = (
                optimization_problem.loss_reactive_change[timestep].value
                + np.imag(self.power_flow_solution.loss)
            )

        return (
            der_active_power_vector,
            der_reactive_power_vector,
            voltage_magnitude_vector,
            branch_power_vector_1_squared,
            branch_power_vector_2_squared,
            loss_active,
            loss_reactive
        )


class LinearElectricGridModelGlobal(LinearElectricGridModel):
    """Linear electric grid model object based on global approximations, consisting of the sensitivity matrices for
    voltage / voltage magnitude / squared branch power / active loss / reactive loss by changes in nodal wye power /
    nodal delta power.

    :syntax:
        - ``LinearElectricGridModelGlobal(electric_grid_model, power_flow_solution)``: Instantiate linear electric grid
          model object for given `electric_grid_model` and `power_flow_solution`.
        - ``LinearElectricGridModelGlobal(scenario_name)``: Instantiate linear electric grid model for given
          `scenario_name`. The required `electric_grid_model` is obtained for given `scenario_name` and the
          `power_flow_solution` is obtained for nominal power conditions.

    Parameters:
        electric_grid_model (fledge.electric_grid_models.ElectricGridModel): Electric grid model object.
        power_flow_solution (fledge.power_flow_solvers.PowerFlowSolution): Power flow solution object.
        scenario_name (str): FLEDGE scenario name.

    Attributes:
        electric_grid_model (fledge.electric_grid_models.ElectricGridModel): Electric grid model object.
        power_flow_solution (fledge.power_flow_solvers.PowerFlowSolution): Reference power flow solution object.
        sensitivity_voltage_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage vector
            by active wye power vector.
        sensitivity_voltage_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage
            vector by reactive wye power vector.
        sensitivity_voltage_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage vector
            by active delta power vector.
        sensitivity_voltage_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage
            vector by reactive delta power vector.
        sensitivity_voltage_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            complex voltage vector by DER active power vector.
        sensitivity_voltage_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            complex voltage vector by DER reactive power vector.
        sensitivity_voltage_magnitude_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for voltage
            magnitude vector by active wye power vector.
        sensitivity_voltage_magnitude_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by reactive wye power vector.
        sensitivity_voltage_magnitude_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by active delta power vector.
        sensitivity_voltage_magnitude_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by reactive delta power vector.
        sensitivity_voltage_magnitude_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by DER active power vector.
        sensitivity_voltage_magnitude_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by DER reactive power vector.
        sensitivity_branch_power_1_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by active wye power vector.
        sensitivity_branch_power_1_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by reactive wye power vector.
        sensitivity_branch_power_1_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by active delta power vector.
        sensitivity_branch_power_1_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by reactive delta power vector.
        sensitivity_branch_power_1_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 by DER active power vector.
        sensitivity_branch_power_1_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 by DER reactive power vector.
        sensitivity_branch_power_2_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by active wye power vector.
        sensitivity_branch_power_2_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by reactive wye power vector.
        sensitivity_branch_power_2_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by active delta power vector.
        sensitivity_branch_power_2_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by reactive delta power vector.
        sensitivity_branch_power_2_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 by DER active power vector.
        sensitivity_branch_power_2_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 by DER reactive power vector.
        sensitivity_loss_active_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by active wye power vector.
        sensitivity_loss_active_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by reactive wye power vector.
        sensitivity_loss_active_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by active delta power vector.
        sensitivity_loss_active_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by reactive delta power vector.
        sensitivity_loss_active_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by DER active power vector.
        sensitivity_loss_active_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by DER reactive power vector.
        sensitivity_loss_reactive_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by active wye power vector.
        sensitivity_loss_reactive_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by reactive wye power vector.
        sensitivity_loss_reactive_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by active delta power vector.
        sensitivity_loss_reactive_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by reactive delta power vector.
        sensitivity_loss_reactive_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by DER active power vector.
        sensitivity_loss_reactive_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by DER reactive power vector.
    """

    @multimethod
    def __init__(
            self,
            scenario_name: str,
    ):

        # Obtain electric grid model.
        electric_grid_model = (
            fledge.electric_grid_models.ElectricGridModel(scenario_name)
        )

        # Obtain der power vector.
        der_power_vector = (
            electric_grid_model.der_power_vector_nominal
        )

        # Obtain power flow solution.
        power_flow_solution = (
            fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(
                electric_grid_model,
                der_power_vector
            )
        )

        self.__init__(
            electric_grid_model,
            power_flow_solution
        )

    @multimethod
    def __init__(
            self,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModel,
            power_flow_solution: fledge.power_flow_solvers.PowerFlowSolution
    ):
        # TODO: Validate linear model with delta DERs.

        # Store power flow solution.
        self.power_flow_solution = power_flow_solution

        # Store electric grid model.
        self.electric_grid_model = electric_grid_model

        # Obtain shorthands for no-source matrices and vectors.
        node_admittance_matrix_no_source = (
            electric_grid_model.node_admittance_matrix[np.ix_(
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
            )]
        )
        node_transformation_matrix_no_source = (
            electric_grid_model.node_transformation_matrix[np.ix_(
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
            )]
        )
        node_voltage_no_source = (
            self.power_flow_solution.node_voltage_vector[
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
            ]
        )

        # Instantiate voltage sensitivity matrices.
        self.sensitivity_voltage_by_power_wye_active = (
            scipy.sparse.dok_matrix(
                (len(electric_grid_model.nodes), len(electric_grid_model.nodes)),
                dtype=complex
            )
        )
        self.sensitivity_voltage_by_power_wye_reactive = (
            scipy.sparse.dok_matrix(
                (len(electric_grid_model.nodes), len(electric_grid_model.nodes)),
                dtype=complex
            )
        )
        self.sensitivity_voltage_by_power_delta_active = (
            scipy.sparse.dok_matrix(
                (len(electric_grid_model.nodes), len(electric_grid_model.nodes)),
                dtype=complex
            )
        )
        self.sensitivity_voltage_by_power_delta_reactive = (
            scipy.sparse.dok_matrix(
                (len(electric_grid_model.nodes), len(electric_grid_model.nodes)),
                dtype=complex
            )
        )

        # Calculate voltage sensitivity matrices.
        # TODO: Document the change in sign in the reactive part compared to Hanif.
        self.sensitivity_voltage_by_power_wye_active[np.ix_(
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        )] = (
            scipy.sparse.linalg.spsolve(
                node_admittance_matrix_no_source.tocsc(),
                scipy.sparse.diags(np.conj(node_voltage_no_source).ravel() ** -1, format='csc')
            )
        )
        self.sensitivity_voltage_by_power_wye_reactive[np.ix_(
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        )] = (
            scipy.sparse.linalg.spsolve(
                1.0j * node_admittance_matrix_no_source.tocsc(),
                scipy.sparse.diags(np.conj(node_voltage_no_source).ravel() ** -1, format='csc')
            )
        )
        self.sensitivity_voltage_by_power_delta_active[np.ix_(
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        )] = (
            scipy.sparse.linalg.spsolve(
                node_admittance_matrix_no_source.tocsc(),
                np.transpose(node_transformation_matrix_no_source)
            )
            @ scipy.sparse.diags(
                (
                    (
                        node_transformation_matrix_no_source
                        @ np.conj(node_voltage_no_source)
                    ) ** -1
                ).ravel()
            )
        )
        self.sensitivity_voltage_by_power_delta_reactive[np.ix_(
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        )] = (
            scipy.sparse.linalg.spsolve(
                1.0j * node_admittance_matrix_no_source.tocsc(),
                np.transpose(node_transformation_matrix_no_source)
            )
            @ scipy.sparse.diags(
                (
                    (
                        node_transformation_matrix_no_source
                        * np.conj(node_voltage_no_source)
                    ) ** -1
                ).ravel()
            )
        )

        self.sensitivity_voltage_by_der_power_active = (
            self.sensitivity_voltage_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_voltage_by_der_power_reactive = (
            self.sensitivity_voltage_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )

        self.sensitivity_voltage_magnitude_by_power_wye_active = (
            scipy.sparse.diags(abs(self.power_flow_solution.node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(self.power_flow_solution.node_voltage_vector).ravel())
                @ self.sensitivity_voltage_by_power_wye_active
            )
        )
        self.sensitivity_voltage_magnitude_by_power_wye_reactive = (
            scipy.sparse.diags(abs(self.power_flow_solution.node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(self.power_flow_solution.node_voltage_vector).ravel())
                @ self.sensitivity_voltage_by_power_wye_reactive
            )
        )
        self.sensitivity_voltage_magnitude_by_power_delta_active = (
            scipy.sparse.diags(abs(self.power_flow_solution.node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(self.power_flow_solution.node_voltage_vector).ravel())
                @ self.sensitivity_voltage_by_power_delta_active
            )
        )
        self.sensitivity_voltage_magnitude_by_power_delta_reactive = (
            scipy.sparse.diags(abs(self.power_flow_solution.node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(self.power_flow_solution.node_voltage_vector).ravel())
                @ self.sensitivity_voltage_by_power_delta_reactive
            )
        )

        self.sensitivity_voltage_magnitude_by_der_power_active = (
            self.sensitivity_voltage_magnitude_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_magnitude_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_voltage_magnitude_by_der_power_reactive = (
            self.sensitivity_voltage_magnitude_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_magnitude_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )

        # Caculate branch flow sensitivity matrices.
        # TODO: Document the removed factor two compared to Hanif.
        sensitivity_branch_power_1_by_voltage = (
            scipy.sparse.diags(np.conj(
                electric_grid_model.branch_admittance_1_matrix
                @ self.power_flow_solution.node_voltage_vector
            ).ravel())
            @ electric_grid_model.branch_incidence_1_matrix
            + scipy.sparse.diags((
                electric_grid_model.branch_incidence_1_matrix
                @ self.power_flow_solution.node_voltage_vector
            ).ravel())
            @ np.conj(electric_grid_model.branch_admittance_1_matrix)
        )
        sensitivity_branch_power_2_by_voltage = (
            scipy.sparse.diags(np.conj(
                electric_grid_model.branch_admittance_2_matrix
                @ self.power_flow_solution.node_voltage_vector
            ).ravel())
            @ electric_grid_model.branch_incidence_2_matrix
            + scipy.sparse.diags((
                electric_grid_model.branch_incidence_2_matrix
                @ self.power_flow_solution.node_voltage_vector
            ).ravel())
            @ np.conj(electric_grid_model.branch_admittance_2_matrix)
        )

        self.sensitivity_branch_power_1_by_power_wye_active = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_1).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_active)
                ),
                np.imag(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_active)
                )
            ])
        )
        self.sensitivity_branch_power_1_by_power_wye_reactive = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_1).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_reactive)
                ),
                np.imag(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_reactive)
                )
            ])
        )
        self.sensitivity_branch_power_1_by_power_delta_active = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_1).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_active)
                ),
                np.imag(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_active)
                )
            ])
        )
        self.sensitivity_branch_power_1_by_power_delta_reactive = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_1).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_reactive)
                ),
                np.imag(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_reactive)
                )
            ])
        )
        self.sensitivity_branch_power_2_by_power_wye_active = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_2).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_active)
                ),
                np.imag(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_active)
                )
            ])
        )
        self.sensitivity_branch_power_2_by_power_wye_reactive = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_2).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_reactive)
                ),
                np.imag(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_reactive)
                )
            ])
        )
        self.sensitivity_branch_power_2_by_power_delta_active = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_2).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_active)
                ),
                np.imag(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_active)
                )
            ])
        )
        self.sensitivity_branch_power_2_by_power_delta_reactive = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_2).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_reactive)
                ),
                np.imag(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_reactive)
                )
            ])
        )

        self.sensitivity_branch_power_1_by_der_power_active = (
            self.sensitivity_branch_power_1_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_1_by_der_power_reactive = (
            self.sensitivity_branch_power_1_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_by_der_power_active = (
            self.sensitivity_branch_power_2_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_by_der_power_reactive = (
            self.sensitivity_branch_power_2_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )

        # Caculate loss sensitivity matrices.
        # TODO: Document the inverted real / imag parts compared to Hanif.
        sensitivity_loss_by_voltage = (
            np.transpose(self.power_flow_solution.node_voltage_vector)
            @ np.conj(electric_grid_model.node_admittance_matrix)
            + np.transpose(
                electric_grid_model.node_admittance_matrix
                @ self.power_flow_solution.node_voltage_vector
            )
        )

        self.sensitivity_loss_active_by_power_wye_active = (
            np.imag(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_wye_active
            )
        )
        self.sensitivity_loss_active_by_power_wye_reactive = (
            np.imag(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_wye_reactive
            )
        )
        self.sensitivity_loss_active_by_power_delta_active = (
            np.imag(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_delta_active
            )
        )
        self.sensitivity_loss_active_by_power_delta_reactive = (
            np.imag(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_delta_reactive
            )
        )

        self.sensitivity_loss_reactive_by_power_wye_active = (
            np.real(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_wye_active
            )
        )
        self.sensitivity_loss_reactive_by_power_wye_reactive = (
            np.real(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_wye_reactive
            )
        )
        self.sensitivity_loss_reactive_by_power_delta_active = (
            np.real(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_delta_active
            )
        )
        self.sensitivity_loss_reactive_by_power_delta_reactive = (
            np.real(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_delta_reactive
            )
        )

        self.sensitivity_loss_active_by_der_power_active = (
            self.sensitivity_loss_active_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_active_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_active_by_der_power_reactive = (
            self.sensitivity_loss_active_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_active_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_reactive_by_der_power_active = (
            self.sensitivity_loss_reactive_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_reactive_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_reactive_by_der_power_reactive = (
            self.sensitivity_loss_reactive_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_reactive_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )
