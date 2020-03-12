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
import fledge.utils

logger = fledge.config.get_logger(__name__)


class PowerFlowSolution(object):
    """Power flow solution object."""

    der_power_vector: np.ndarray
    node_voltage_vector: np.ndarray
    branch_power_vector_1: np.ndarray
    branch_power_vector_2: np.ndarray
    loss: np.complex

    @multimethod
    def __init__(
            self,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModel,
            **kwargs
    ):

        # Obtain `der_power_vector`, assuming nominal power conditions.
        der_power_vector = electric_grid_model.der_power_vector_nominal

        self.__init__(
            electric_grid_model,
            der_power_vector,
            **kwargs
        )


class PowerFlowSolutionFixedPoint(PowerFlowSolution):
    """Fixed point power flow solution object."""

    node_power_vector_wye: np.ndarray
    node_power_vector_delta: np.ndarray

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            **kwargs
    ):

        # Obtain `electric_grid_model`.
        electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

        super().__init__(
            electric_grid_model,
            **kwargs
        )

    @multimethod
    def __init__(
            self,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            der_power_vector: np.ndarray,
            **kwargs
    ):

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
            self.get_voltage(
                electric_grid_model,
                self.der_power_vector,
                **kwargs
            )
        )

        # Obtain branch flow solution.
        (
            self.branch_power_vector_1,
            self.branch_power_vector_2
        ) = (
            self.get_branch_power(
                electric_grid_model,
                self.node_voltage_vector
            )
        )

        # Obtain loss solution.
        self.loss = (
            self.get_loss(
                electric_grid_model,
                self.node_voltage_vector
            )
        )

    @staticmethod
    def get_voltage(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        der_power_vector: np.ndarray,
        voltage_iteration_limit=100,
        voltage_tolerance=1e-2
    ):
        """Get nodal voltage vector by solving with the fixed point algorithm.

        - Initial DER power vector / node voltage vector must be a valid
          solution to te fixed-point equation, e.g., a previous solution from a past
          operation point.
        - Fixed point equation according to: <https://arxiv.org/pdf/1702.03310.pdf>
        """

        # Obtain no-source variables for fixed point equation.
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
        der_incidence_wye_matrix_no_source = (
            electric_grid_model.der_incidence_wye_matrix[
                np.ix_(
                    fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
                    range(len(electric_grid_model.ders))
                )
            ]
        )
        der_incidence_delta_matrix_no_source = (
            electric_grid_model.der_incidence_delta_matrix[
                np.ix_(
                    fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
                    range(len(electric_grid_model.ders))
                )
            ]
        )

        node_power_vector_wye_no_source = (
            der_incidence_wye_matrix_no_source
            @ np.transpose([der_power_vector.ravel()])
        )
        node_power_vector_delta_no_source = (
            der_incidence_delta_matrix_no_source
            @ np.transpose([der_power_vector.ravel()])
        )
        node_voltage_vector_no_load_no_source = (
            electric_grid_model.node_voltage_vector_no_load[
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
            ]
        )

        # Obtain initial nodal power and voltage vectors, assuming no power conditions.
        node_power_vector_wye_initial_no_source = np.zeros(node_power_vector_wye_no_source.shape, dtype=complex)
        node_power_vector_delta_initial_no_source = np.zeros(node_power_vector_delta_no_source.shape, dtype=complex)
        node_voltage_vector_initial_no_source = node_voltage_vector_no_load_no_source

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

        if voltage_iteration >= voltage_iteration_limit:
            # Reaching the iteration limit is considered undesired and therefore triggers a warning.
            logger.warning(
                f"Fixed point voltage solution algorithm reached maximum limit of {voltage_iteration_limit} iterations."
            )

        # Get full voltage vector by concatenating source and calculated voltage.
        node_voltage_vector = (
            np.vstack([
                electric_grid_model.node_voltage_vector_no_load[
                    fledge.utils.get_index(electric_grid_model.nodes, node_type='source')
                ],
                node_voltage_vector_initial_no_source  # Takes value of `node_voltage_vector_solution_no_source`.
            ])
        )
        return node_voltage_vector

    @staticmethod
    def get_branch_power(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        node_voltage_vector: np.ndarray
    ):
        """Get branch power vectors by calculating power flow with given nodal voltage.

        - Returns two branch power vectors, where `branch_power_vector_1` represents the
          "from"-direction and `branch_power_vector_2` represents the "to"-direction.
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

    @staticmethod
    def get_loss(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        node_voltage_vector: np.ndarray
    ):
        """Get total electric losses with given nodal voltage."""

        # Calculate total losses.
        # TODO: Validate loss solution.
        loss = (
            np.conj(
                np.transpose(node_voltage_vector)
                @ (
                    electric_grid_model.node_admittance_matrix
                    @ node_voltage_vector
                )
            )
        )

        return loss


class PowerFlowSolutionOpenDSS(PowerFlowSolution):
    """OpenDSS power flow solution object."""

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            **kwargs
    ):

        # Obtain `electric_grid_model`.
        electric_grid_model = fledge.electric_grid_models.ElectricGridModelOpenDSS(scenario_name)

        super().__init__(
            electric_grid_model,
            **kwargs
        )

    @multimethod
    def __init__(
            self,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelOpenDSS,
            der_power_vector: np.ndarray,
            **kwargs
    ):

        # Store DER power vector.
        self.der_power_vector = der_power_vector

        # Obtain voltage solution.
        self.node_voltage_vector = (
            self.get_voltage()
        )

        # Obtain branch flow solution.
        (
            self.branch_power_vector_1,
            self.branch_power_vector_2
        ) = (
            self.get_branch_power()
        )

        # Obtain loss solution.
        self.loss = (
            self.get_loss()
        )

    @staticmethod
    def get_voltage():
        """Get nodal voltage vector by solving OpenDSS model.

        - OpenDSS model must be readily set up, with the desired power being set for all DERs.
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

    @staticmethod
    def get_branch_power():
        """Get branch power vectors by solving OpenDSS model.

        - OpenDSS model must be readily set up, with the desired power being set for all DERs.
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

    @staticmethod
    def get_loss():
        """Get total loss by solving OpenDSS model.

        - OpenDSS model must be readily set up, with the desired power being set for all DERs.
        """

        # Solve OpenDSS model.
        opendssdirect.run_command("solve")

        # Obtain loss.
        loss = opendssdirect.Circuit.Losses()[0] + 1.0j * opendssdirect.Circuit.Losses()[1]

        return loss
