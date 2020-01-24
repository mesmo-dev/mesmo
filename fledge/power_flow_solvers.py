"""Power flow solvers."""

from multimethod import multimethod
import natsort
import numpy as np
import opendssdirect
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg

import fledge.config
import fledge.database_interface
import fledge.electric_grid_models

logger = fledge.config.get_logger(__name__)


@multimethod
def get_voltage_fixed_point(
    electric_grid_model: fledge.electric_grid_models.ElectricGridModel,
    **kwargs
):
    """
    Get nodal voltage vector by solving with the fixed point algorithm.

    - Obtains nodal power vectors assuming nominal power conditions from an
      `electric_grid_model` object.
    """

    # Obtain nodal power vectors assuming nominal power conditions.
    node_power_vector_wye = (
        np.transpose([
            electric_grid_model.der_incidence_wye_matrix
            @ electric_grid_model.der_power_vector_nominal
        ])
    )
    node_power_vector_delta = (
        np.transpose([
            electric_grid_model.der_incidence_delta_matrix
            @ electric_grid_model.der_power_vector_nominal
        ])
    )

    # Get fixed point solution.
    node_voltage_vector_solution = get_voltage_fixed_point(
        electric_grid_model,
        node_power_vector_wye,
        node_power_vector_delta,
        **kwargs
    )
    return node_voltage_vector_solution


@multimethod
def get_voltage_fixed_point(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModel,
        node_power_vector_wye: np.ndarray,
        node_power_vector_delta: np.ndarray,
        **kwargs
):
    """
    Get nodal voltage vector by solving with the fixed point algorithm.

    - Takes nodal wye-power, delta-power vectors as inputs.
    - Obtains the nodal admittance, transformation matrices and
      initial nodal wye-power, delta-power and voltage vectors as well as
      nodal no load voltage vector without source nodes from an
      `electric_grid_model` object.
    - Assumes no load conditions for initial nodal power and voltage vectors.
    """

    # Obtain no-source variables for fixed point equation.
    node_admittance_matrix_no_source = (
        electric_grid_model.node_admittance_matrix[
            electric_grid_model.index.node_by_node_type['no_source'], :
        ][
            :, electric_grid_model.index.node_by_node_type['no_source']
        ]
    )
    node_transformation_matrix_no_source = (
        electric_grid_model.node_transformation_matrix[
            electric_grid_model.index.node_by_node_type['no_source'], :
        ][
            :, electric_grid_model.index.node_by_node_type['no_source']
        ]
    )
    node_power_vector_wye_no_source = (
        node_power_vector_wye[electric_grid_model.index.node_by_node_type['no_source']]
    )
    node_power_vector_delta_no_source = (
        node_power_vector_delta[electric_grid_model.index.node_by_node_type['no_source']]
    )
    node_voltage_vector_no_load_no_source = (
        electric_grid_model.node_voltage_vector_no_load[electric_grid_model.index.node_by_node_type['no_source']]
    )

    # Define initial nodal power and voltage vectors as no load conditions.
    node_power_vector_wye_initial_no_source = np.zeros(node_power_vector_wye_no_source.shape, dtype=complex)
    node_power_vector_delta_initial_no_source = np.zeros(node_power_vector_delta_no_source.shape, dtype=complex)
    node_voltage_vector_initial_no_source = node_voltage_vector_no_load_no_source

    # Get fixed point solution.
    node_voltage_vector_solution = get_voltage_fixed_point(
        node_admittance_matrix_no_source,
        node_transformation_matrix_no_source,
        node_power_vector_wye_no_source,
        node_power_vector_delta_no_source,
        node_power_vector_wye_initial_no_source,
        node_power_vector_delta_initial_no_source,
        node_voltage_vector_no_load_no_source,
        node_voltage_vector_initial_no_source,
        **kwargs
    )

    # Get full voltage vector by concatenating source and calculated voltage.
    node_voltage_vector_solution = (
        np.vstack([
            electric_grid_model.node_voltage_vector_no_load[electric_grid_model.index.node_by_node_type['source']],
            node_voltage_vector_solution
        ])
    )
    return node_voltage_vector_solution


@multimethod
def get_voltage_fixed_point(
        node_admittance_matrix_no_source: scipy.sparse.spmatrix,
        node_transformation_matrix_no_source: scipy.sparse.spmatrix,
        node_power_vector_wye_no_source: np.ndarray,
        node_power_vector_delta_no_source: np.ndarray,
        node_power_vector_wye_initial_no_source: np.ndarray,
        node_power_vector_delta_initial_no_source: np.ndarray,
        node_voltage_vector_no_load_no_source: np.ndarray,
        node_voltage_vector_initial_no_source: np.ndarray,
        voltage_iteration_limit=100,
        voltage_tolerance=1e-2
):
    """Get nodal voltage vector by solving with the fixed point algorithm.

    - Takes the nodal admittance, transformation matrices and
      nodal wye-power, delta-power, voltage vectors without source nodes, i.e.,
      source nodes must be removed from the arrays before passing to this function.
    - Initial nodal wye-power, delta-power, voltage vectors must be a valid
      solution to te fixed-point equation, e.g., a previous solution from a past
      operation point.
    - Fixed point equation according to: <https://arxiv.org/pdf/1702.03310.pdf>
    """

    # Instantiate fixed point iteration variables.
    voltage_iteration = 1
    voltage_change = np.inf

    while (
            (voltage_iteration < voltage_iteration_limit)
            & (voltage_change > voltage_tolerance)
    ):
        # Calculate fixed point equation.
        node_voltage_vector_solution_no_source = (
            node_voltage_vector_no_load_no_source
            + np.transpose([
                scipy.sparse.linalg.spsolve(
                    node_admittance_matrix_no_source,
                    (
                        (
                            (
                                np.conj(node_voltage_vector_initial_no_source) ** -1
                            )
                            * np.conj(node_power_vector_wye_no_source)
                        )
                        + (
                            np.transpose(node_transformation_matrix_no_source)
                            @ (
                                (
                                    (
                                        node_transformation_matrix_no_source
                                        @ np.conj(node_voltage_vector_initial_no_source)
                                    ) ** -1
                                )
                                * np.conj(node_power_vector_delta_no_source)
                            )
                        )
                    )
                )
            ])
        )

        # Calculate voltage change from previous iteration.
        voltage_change = (
            np.max(abs(
                node_voltage_vector_solution_no_source
                - node_voltage_vector_initial_no_source
            ))
        )

        # Set voltage solution as initial voltage for next iteration.
        node_voltage_vector_initial_no_source = (
            node_voltage_vector_solution_no_source
        )

        # Increment voltage iteration counter.
        voltage_iteration += 1

    if voltage_iteration == voltage_iteration_limit:
        # Reaching the iteration limit is considered undesired and therefore triggers a warning.
        logger.warning(
            f"Fixed point voltage solution algorithm reached maximum limit of {voltage_iteration_limit} iterations."
        )

    return node_voltage_vector_solution_no_source


def get_voltage_opendss():
    """Get nodal voltage vector by solving OpenDSS model.

    - OpenDSS model must be readily set up, with the desired power being set for all ders.
    """

    # Solve OpenDSS model.
    opendssdirect.run_command("solve")

    # Extract nodal voltage vector.
    # - Voltages are sorted by node names in the fashion as nodes are sorted in
    #   `nodes` in `fledge.electric_grid_models.ElectricGridModelIndex()`.
    node_voltage_vector_solution = (
        np.transpose([
            pd.Series(
                (
                    np.array(opendssdirect.Circuit.AllBusVolts()[0::2])
                    + 1j * np.array(opendssdirect.Circuit.AllBusVolts()[1::2])
                ),
                index=opendssdirect.Circuit.AllNodeNames()
            ).reindex(
                natsort.natsorted(opendssdirect.Circuit.AllNodeNames())
            ).values
        ])
    )

    return node_voltage_vector_solution


@multimethod
def get_branch_power_fixed_point(
    electric_grid_model: fledge.electric_grid_models.ElectricGridModel,
    node_voltage_vector: np.ndarray
):
    """Get branch power vectors by calculating power flow with given nodal voltage.

    - Obtains the needed matrices from an `electric_grid_model` object.
    """

    # Obtain branch admittance and incidence matrices.
    branch_admittance_1_matrix = (
        electric_grid_model.branch_admittance_1_matrix
    )
    branch_admittance_2_matrix = (
        electric_grid_model.branch_admittance_2_matrix
    )
    branch_incidence_1_matrix = (
        electric_grid_model.branch_incidence_1_matrix
    )
    branch_incidence_2_matrix = (
        electric_grid_model.branch_incidence_2_matrix
    )

    # Calculate branch power vectors.
    return get_branch_power_fixed_point(
        branch_admittance_1_matrix,
        branch_admittance_2_matrix,
        branch_incidence_1_matrix,
        branch_incidence_2_matrix,
        node_voltage_vector
    )


@multimethod
def get_branch_power_fixed_point(
    branch_admittance_1_matrix: scipy.sparse.spmatrix,
    branch_admittance_2_matrix: scipy.sparse.spmatrix,
    branch_incidence_1_matrix: scipy.sparse.spmatrix,
    branch_incidence_2_matrix: scipy.sparse.spmatrix,
    node_voltage_vector: np.ndarray
):
    """Get branch power vectors by calculating power flow with given nodal voltage.

    - Returns two branch power vectors, where `branch_power_vector_1` represents the
      "from"-direction and `branch_power_vector_2` represents the "to"-direction.
    - Nodal voltage vector is assumed to be obtained from fixed-point solution,
      therefore this function is associated with the fixed-point solver.
    - This function directly takes branch admittance and incidence matrices as
      inputs, which can be obtained from an `electric_grid_model` object.
    """

    # Calculate branch power vectors.
    branch_power_vector_1 = (
        (
            branch_incidence_1_matrix
            @ node_voltage_vector
        )
        * np.conj(
            branch_admittance_1_matrix
            @ node_voltage_vector
        )
    )
    branch_power_vector_2 = (
        (
            branch_incidence_2_matrix
            @ node_voltage_vector
        )
        * np.conj(
            branch_admittance_2_matrix
            @ node_voltage_vector
        )
    )

    return (
        branch_power_vector_1,
        branch_power_vector_2
    )


def get_branch_power_opendss():
    """Get branch power vectors by solving OpenDSS model.

    - OpenDSS model must be readily set up, with the desired power being set for all ders.
    """

    # Solve OpenDSS model.
    opendssdirect.run_command("solve")

    # Instantiate branch vectors.
    branch_power_vector_1 = (
        np.full(((opendssdirect.Lines.Count() + opendssdirect.Transformers.Count()), 3), np.nan, dtype=np.complex)
    )
    branch_power_vector_2 = (
        np.full(((opendssdirect.Lines.Count() + opendssdirect.Transformers.Count()), 3), np.nan, dtype=np.complex)
    )

    # Instantiate iteration variables.
    branch_vector_index = 0
    line_index = opendssdirect.Lines.First()

    # Obtain line branch power vectors.
    while line_index > 0:
        branch_power = np.array(opendssdirect.CktElement.Powers())
        branch_phase_count = opendssdirect.CktElement.NumPhases()
        branch_power_vector_1[branch_vector_index, :branch_phase_count] = (
            branch_power[0:(branch_phase_count * 2):2]
            + 1.0j * branch_power[1:(branch_phase_count * 2):2]
        )
        branch_power_vector_2[branch_vector_index, :branch_phase_count] = (
            branch_power[0 + (branch_phase_count * 2)::2]
            + 1.0j * branch_power[1 + (branch_phase_count * 2)::2]
        )

        branch_vector_index += 1
        line_index = opendssdirect.Lines.Next()

    # Obtain transformer branch power vectors.
    transformer_index = opendssdirect.Transformers.First()
    while transformer_index > 0:
        branch_power = np.array(opendssdirect.CktElement.Powers())
        branch_phase_count = opendssdirect.CktElement.NumPhases()
        skip_phase = 2 if 0 in opendssdirect.CktElement.NodeOrder() else 0  # Ignore ground nodes.
        branch_power_vector_1[branch_vector_index, :branch_phase_count] = (
            branch_power[0:(branch_phase_count * 2):2]
            + 1.0j * branch_power[1:(branch_phase_count * 2):2]
        )
        branch_power_vector_2[branch_vector_index, :branch_phase_count] = (
            branch_power[0 + (branch_phase_count * 2) + skip_phase:-skip_phase:2]
            + 1.0j * branch_power[1 + (branch_phase_count * 2) + skip_phase:-skip_phase:2]
        )

        branch_vector_index += 1
        transformer_index = opendssdirect.Transformers.Next()

    # Reshape branch power vectors to appropriate size and remove entries for nonexistent phases.
    # TODO: Sort vector by branch name if not in order.
    branch_power_vector_1 = branch_power_vector_1.flatten()
    branch_power_vector_2 = branch_power_vector_2.flatten()
    branch_power_vector_1 = np.transpose([branch_power_vector_1[~np.isnan(branch_power_vector_1)]])
    branch_power_vector_2 = np.transpose([branch_power_vector_2[~np.isnan(branch_power_vector_2)]])

    return (
        branch_power_vector_1,
        branch_power_vector_2
    )


@multimethod
def get_loss_fixed_point(
    electric_grid_model: fledge.electric_grid_models.ElectricGridModel,
    node_voltage_vector: np.ndarray
):
    """
    Get total electric losses with given nodal voltage.

    - Obtains the nodal admittance matrix from an `electric_grid_model` object.
    """

    # Obtain total losses with admittance matrix from electric grid model.
    return (
        get_loss_fixed_point(
            electric_grid_model.node_admittance_matrix,
            node_voltage_vector
        )
    )


@multimethod
def get_loss_fixed_point(
    node_admittance_matrix: scipy.sparse.spmatrix,
    node_voltage_vector: np.ndarray
):
    """
    Get total electric losses with given nodal voltage.

    - Nodal voltage vector is assumed to be obtained from fixed-point solution,
      therefore this function is associated with the fixed-point solver.
    - This function directly takes the nodal admittance matrix as
      input, which can be obtained from an `electric_grid_model` object.
    """

    # Calculate total losses.
    # TODO: Validate loss solution.
    loss = (
        np.conj(
            np.transpose(node_voltage_vector)
            @ (
                node_admittance_matrix
                @ node_voltage_vector
            )
        )
    )

    return loss


def get_loss_opendss():
    """Get total loss by solving OpenDSS model.

    - OpenDSS model must be readily set up, with the desired power being set for all ders.
    """

    # Solve OpenDSS model.
    opendssdirect.run_command("solve")

    # Obtain loss.
    loss = opendssdirect.Circuit.Losses()[0] + 1.0j * opendssdirect.Circuit.Losses()[1]

    return loss


class PowerFlowSolution(object):
    """Power flow solution object."""

    der_power_vector: np.ndarray
    node_power_vector_wye: np.ndarray
    node_power_vector_delta: np.ndarray
    node_voltage_vector: np.ndarray
    branch_power_vector_1: np.ndarray
    branch_power_vector_2: np.ndarray
    loss: np.complex


class PowerFlowSolutionFixedPoint(PowerFlowSolution):
    """Fixed point power flow solution object."""

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):
        """Instantiate fixed point power flow solution object for given `scenario_name`
        assuming nominal power conditions.
        """

        # Obtain `electric_grid_model`.
        electric_grid_model = fledge.electric_grid_models.ElectricGridModel(scenario_name)

        self.__init__(electric_grid_model)

    @multimethod
    def __init__(
            self,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModel
    ):
        """Instantiate fixed point power flow solution object for given `electric_grid_model`
        assuming nominal power conditions.
        """

        # Obtain `der_power_vector` assuming nominal power conditions.
        der_power_vector = electric_grid_model.der_power_vector_nominal

        self.__init__(
            electric_grid_model,
            der_power_vector
        )

    @multimethod
    def __init__(
            self,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModel,
            der_power_vector: np.ndarray
    ):
        """Instantiate fixed point power flow solution object for given `electric_grid_model` and `der_power_vector`.
        """

        # Store DER power vector.
        self.der_power_vector = der_power_vector

        # Obtain node power vectors.
        self.node_power_vector_wye = (
            np.transpose([
                electric_grid_model.der_incidence_wye_matrix
                @ self.der_power_vector
            ])
        )
        self.node_power_vector_delta = (
            np.transpose([
                electric_grid_model.der_incidence_delta_matrix
                @ self.der_power_vector
            ])
        )

        # Obtain voltage solution.
        self.node_voltage_vector = (
            fledge.power_flow_solvers.get_voltage_fixed_point(
                electric_grid_model,
                self.node_power_vector_wye,
                self.node_power_vector_delta
            )
        )

        # Obtain branch flow solution.
        (
            self.branch_power_vector_1,
            self.branch_power_vector_2
        ) = (
            fledge.power_flow_solvers.get_branch_power_fixed_point(
                electric_grid_model,
                self.node_voltage_vector
            )
        )

        # Obtain loss solution.
        self.loss = (
            fledge.power_flow_solvers.get_loss_fixed_point(
                electric_grid_model,
                self.node_voltage_vector
            )
        )
