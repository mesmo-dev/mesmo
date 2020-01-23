"""Linear electric grid models module."""

from multimethod import multimethod
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg

import fledge.config
import fledge.electric_grid_models
import fledge.power_flow_solvers

logger = fledge.config.get_logger(__name__)


class LinearElectricGridModel(object):
    """Abstract linear electric model object, consisting of the sensitivity matrices for
    voltage / voltage magnitude / squared branch power / active loss / reactive loss by changes in nodal wye power /
    nodal delta power.

    Note:
        This abstract class only defines the expected variables of linear electric grid model objects,
        but does not implement any functionality.

    Attributes:
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
        # TODO: Validate linear model with delta ders.

        # Obtain vectors.
        node_voltage_vector = power_flow_solution.node_voltage_vector
        branch_power_vector_1 = power_flow_solution.branch_power_vector_1
        branch_power_vector_2 = power_flow_solution.branch_power_vector_2

        # Obtain no_source matrices and vectors.
        node_admittance_matrix_no_source = (
            electric_grid_model.node_admittance_matrix[np.ix_(
                electric_grid_model.index.node_by_node_type['no_source'],
                electric_grid_model.index.node_by_node_type['no_source']
            )]
        )
        node_transformation_matrix_no_source = (
            electric_grid_model.node_transformation_matrix[np.ix_(
                electric_grid_model.index.node_by_node_type['no_source'],
                electric_grid_model.index.node_by_node_type['no_source']
            )]
        )
        node_voltage_no_source = (
            node_voltage_vector[
                electric_grid_model.index.node_by_node_type['no_source']
            ]
        )
    
        # Instantiate voltage sensitivity matrices.
        self.sensitivity_voltage_by_power_wye_active = (
            scipy.sparse.dok_matrix(
                (electric_grid_model.index.node_dimension, electric_grid_model.index.node_dimension),
                dtype=complex
            )
        )
        self.sensitivity_voltage_by_power_wye_reactive = (
            scipy.sparse.dok_matrix(
                (electric_grid_model.index.node_dimension, electric_grid_model.index.node_dimension),
                dtype=complex
            )
        )
        self.sensitivity_voltage_by_power_delta_active = (
            scipy.sparse.dok_matrix(
                (electric_grid_model.index.node_dimension, electric_grid_model.index.node_dimension),
                dtype=complex
            )
        )
        self.sensitivity_voltage_by_power_delta_reactive = (
            scipy.sparse.dok_matrix(
                (electric_grid_model.index.node_dimension, electric_grid_model.index.node_dimension),
                dtype=complex
            )
        )
    
        # Calculate voltage sensitivity matrices.
        # TODO: Document the change in sign in the reactive part compared to Hanif.
        self.sensitivity_voltage_by_power_wye_active[np.ix_(
            electric_grid_model.index.node_by_node_type['no_source'],
            electric_grid_model.index.node_by_node_type['no_source']
        )] = (
            scipy.sparse.linalg.spsolve(
                node_admittance_matrix_no_source.tocsc(),
                scipy.sparse.diags(np.conj(node_voltage_no_source).ravel() ** -1, format='csc')
            )
        )
        self.sensitivity_voltage_by_power_wye_reactive[np.ix_(
            electric_grid_model.index.node_by_node_type['no_source'],
            electric_grid_model.index.node_by_node_type['no_source']
        )] = (
            scipy.sparse.linalg.spsolve(
                1.0j * node_admittance_matrix_no_source.tocsc(),
                scipy.sparse.diags(np.conj(node_voltage_no_source).ravel() ** -1, format='csc')
            )
        )
        self.sensitivity_voltage_by_power_delta_active[np.ix_(
            electric_grid_model.index.node_by_node_type['no_source'],
            electric_grid_model.index.node_by_node_type['no_source']
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
            electric_grid_model.index.node_by_node_type['no_source'],
            electric_grid_model.index.node_by_node_type['no_source']
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
            scipy.sparse.diags(abs(node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(node_voltage_vector).ravel())
                @ self.sensitivity_voltage_by_power_wye_active
            )
        )
        self.sensitivity_voltage_magnitude_by_power_wye_reactive = (
            scipy.sparse.diags(abs(node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(node_voltage_vector).ravel())
                @ self.sensitivity_voltage_by_power_wye_reactive
            )
        )
        self.sensitivity_voltage_magnitude_by_power_delta_active = (
            scipy.sparse.diags(abs(node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(node_voltage_vector).ravel())
                @ self.sensitivity_voltage_by_power_delta_active
            )
        )
        self.sensitivity_voltage_magnitude_by_power_delta_reactive = (
            scipy.sparse.diags(abs(node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(node_voltage_vector).ravel())
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
                @ node_voltage_vector
            ).ravel())
            @ electric_grid_model.branch_incidence_1_matrix
            + scipy.sparse.diags((
                electric_grid_model.branch_incidence_1_matrix
                @ node_voltage_vector
            ).ravel())
            @ np.conj(electric_grid_model.branch_admittance_1_matrix)
        )
        sensitivity_branch_power_2_by_voltage = (
            scipy.sparse.diags(np.conj(
                electric_grid_model.branch_admittance_2_matrix
                @ node_voltage_vector
            ).ravel())
            @ electric_grid_model.branch_incidence_2_matrix
            + scipy.sparse.diags((
                electric_grid_model.branch_incidence_2_matrix
                @ node_voltage_vector
            ).ravel())
            @ np.conj(electric_grid_model.branch_admittance_2_matrix)
        )

        self.sensitivity_branch_power_1_by_power_wye_active = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(branch_power_vector_1).ravel())
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
                scipy.sparse.diags(np.real(branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(branch_power_vector_1).ravel())
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
                scipy.sparse.diags(np.real(branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(branch_power_vector_1).ravel())
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
                scipy.sparse.diags(np.real(branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(branch_power_vector_1).ravel())
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
                scipy.sparse.diags(np.real(branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(branch_power_vector_2).ravel())
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
                scipy.sparse.diags(np.real(branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(branch_power_vector_2).ravel())
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
                scipy.sparse.diags(np.real(branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(branch_power_vector_2).ravel())
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
                scipy.sparse.diags(np.real(branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(branch_power_vector_2).ravel())
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
            np.transpose(node_voltage_vector)
            @ np.conj(electric_grid_model.node_admittance_matrix)
            + np.transpose(
                electric_grid_model.node_admittance_matrix
                @ node_voltage_vector
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
