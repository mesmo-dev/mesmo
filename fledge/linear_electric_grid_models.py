"""Linear electric grid models."""

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
    """Linear electric grid model object."""

    sensitivity_voltage_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_voltage_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_voltage_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_delta_reactive: scipy.sparse.spmatrix

    @multimethod
    def __init__(
            self,
            scenario_name: str,
    ):
        """Instantiate linear electric grid model object for given `scenario_name`.

        - Power flow solution is obtained for nominal loading conditions
          via fixed point solution.
        """

        # Obtain electric grid model.
        electric_grid_model = (
            fledge.electric_grid_models.ElectricGridModel(scenario_name)
        )

        # Obtain load power vector.
        load_power_vector = (
            electric_grid_model.load_power_vector_nominal
        )

        # Obtain power flow solution.
        power_flow_solution = (
            fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(
                electric_grid_model,
                load_power_vector
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
        """Instantiate linear electric grid model object for given `electric_grid_model`
        and `power_flow_solution`.
        """

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
        # TODO: Address sparse solve efficiency warning.
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
        # TODO: Testing of delta-loads with linear model.
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

        # Caculate branch flow sensitivity matrices.
        # TODO: Validate branch flow sensitivity equations.
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
            2.0
            * scipy.sparse.hstack([
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
            2.0
            * scipy.sparse.hstack([
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
            2.0
            * scipy.sparse.hstack([
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
            2.0
            * scipy.sparse.hstack([
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
            2.0
            * scipy.sparse.hstack([
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
            2.0
            * scipy.sparse.hstack([
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
            2.0
            * scipy.sparse.hstack([
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
            2.0
            * scipy.sparse.hstack([
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

        # Caculate loss sensitivity matrices.
        # TODO: Validate loss sensitivity equations.
        sensitivity_loss_active_by_voltage = (
            np.conj(
                np.transpose(node_voltage_vector)
                @ np.real(electric_grid_model.node_admittance_matrix)
            )
            + np.transpose(np.conj(
                np.real(electric_grid_model.node_admittance_matrix)
                @ node_voltage_vector
            ))
        )
        sensitivity_loss_reactive_by_voltage = (
            np.conj(
                np.transpose(node_voltage_vector)
                @ np.imag(- electric_grid_model.node_admittance_matrix)
            )
            + np.transpose(np.conj(
                np.imag(- electric_grid_model.node_admittance_matrix)
                @ node_voltage_vector
            ))
        )

        self.sensitivity_loss_active_by_power_wye_active = (
            np.real(sensitivity_loss_active_by_voltage)
            @ np.real(self.sensitivity_voltage_by_power_wye_active)
            + np.imag(sensitivity_loss_active_by_voltage)
            @ np.imag(self.sensitivity_voltage_by_power_wye_active)
        )
        self.sensitivity_loss_active_by_power_wye_reactive = (
            np.real(sensitivity_loss_active_by_voltage)
            @ np.real(self.sensitivity_voltage_by_power_wye_reactive)
            + np.imag(sensitivity_loss_active_by_voltage)
            @ np.imag(self.sensitivity_voltage_by_power_wye_reactive)
        )
        self.sensitivity_loss_active_by_power_delta_active = (
            np.real(sensitivity_loss_active_by_voltage)
            @ np.real(self.sensitivity_voltage_by_power_delta_active)
            + np.imag(sensitivity_loss_active_by_voltage)
            @ np.imag(self.sensitivity_voltage_by_power_delta_active)
        )
        self.sensitivity_loss_active_by_power_delta_reactive = (
            np.real(sensitivity_loss_active_by_voltage)
            @ np.real(self.sensitivity_voltage_by_power_delta_reactive)
            + np.imag(sensitivity_loss_active_by_voltage)
            @ np.imag(self.sensitivity_voltage_by_power_delta_reactive)
        )

        self.sensitivity_loss_reactive_by_power_wye_active = (
            np.real(sensitivity_loss_reactive_by_voltage)
            @ np.real(self.sensitivity_voltage_by_power_wye_active)
            + np.imag(sensitivity_loss_reactive_by_voltage)
            @ np.imag(self.sensitivity_voltage_by_power_wye_active)
        )
        self.sensitivity_loss_reactive_by_power_wye_reactive = (
            np.real(sensitivity_loss_reactive_by_voltage)
            @ np.real(self.sensitivity_voltage_by_power_wye_reactive)
            + np.imag(sensitivity_loss_reactive_by_voltage)
            @ np.imag(self.sensitivity_voltage_by_power_wye_reactive)
        )
        self.sensitivity_loss_reactive_by_power_delta_active = (
            np.real(sensitivity_loss_reactive_by_voltage)
            @ np.real(self.sensitivity_voltage_by_power_delta_active)
            + np.imag(sensitivity_loss_reactive_by_voltage)
            @ np.imag(self.sensitivity_voltage_by_power_delta_active)
        )
        self.sensitivity_loss_reactive_by_power_delta_reactive = (
            np.real(sensitivity_loss_reactive_by_voltage)
            @ np.real(self.sensitivity_voltage_by_power_delta_reactive)
            + np.imag(sensitivity_loss_reactive_by_voltage)
            @ np.imag(self.sensitivity_voltage_by_power_delta_reactive)
        )
