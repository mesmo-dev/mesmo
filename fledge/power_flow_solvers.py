"""Power flow solvers."""

from multimethod import multimethod
import numpy as np
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

    - Obtains nodal power vectors assuming nominal loading conditions from an
      `electric_grid_model` object.
    """

    # Obtain nodal power vectors assuming nominal loading conditions.
    node_power_vector_wye = (
        np.transpose([
            electric_grid_model.load_incidence_wye_matrix
            @ electric_grid_model.load_power_vector_nominal
        ])
    )
    node_power_vector_delta = (
        np.transpose([
            electric_grid_model.load_incidence_delta_matrix
            @ electric_grid_model.load_power_vector_nominal
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
      inital nodal wye-power, delta-power and voltage vectors as well as
      nodal no-load voltage vector without source nodes from an
      `electric_grid_model` object.
    - Assumes no-load conditions for initial nodal power and voltage vectors.
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

    # Define initial nodal power and voltage vectors as no-load conditions.
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


class PowerFlowSolution(object):
    """Power flow solution object."""

    node_voltage_vector: np.ndarray
    branch_power_vector_1: np.ndarray
    branch_power_vector_2: np.ndarray
    loss: np.complex


class FixedPointPowerFlowSolution(PowerFlowSolution):
    """Fixed point power flow solution object."""

    def __init__(
            self,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModel = None,
            load_power_vector: np.ndarray = None,
            scenario_name: str = None
    ):
        """Instantiate fixed point power flow solution object for given `electric_grid_model` and `load_power_vector`
        or for given `scenario_name` assuming nominal loading conditions.
        """

        # Obtain `electric_grid_model`, if none.
        if electric_grid_model is None:
            electric_grid_model = fledge.electric_grid_models.ElectricGridModel(scenario_name=scenario_name)

        # Obtain `load_power_vector`, if none, assuming nominal loading conditions.
        if load_power_vector is None:
            load_power_vector = electric_grid_model.load_power_vector_nominal
