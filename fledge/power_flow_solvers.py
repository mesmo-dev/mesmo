"""Power flow solvers."""

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg

import fledge.config
import fledge.database_interface
import fledge.electric_grid_models

logger = fledge.config.get_logger(__name__)


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
