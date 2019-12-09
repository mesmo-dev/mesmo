"Example script for setting up and solving an optimal power flow problem."

import FLEDGE

import GLPK
import GR
import JuMP
import Logging
import Plots
import TimeSeries

# Settings.
scenario_name = "singapore_6node"
Plots.gr()  # Select plotting backend.

# Get model.
electric_grid_data = (
    FLEDGE.DatabaseInterface.ElectricGridData(scenario_name)
)
electric_grid_index = (
    FLEDGE.ElectricGridModels.ElectricGridIndex(scenario_name)
)
electric_grid_model = (
    FLEDGE.ElectricGridModels.ElectricGridModel(scenario_name)
)
linear_electric_grid_model = (
    FLEDGE.ElectricGridModels.LinearElectricGridModel(scenario_name)
)
Logging.@info("", linear_electric_grid_model)

# Instantiate optimization problem.
optimization_problem = (
    JuMP.Model(JuMP.with_optimizer(GLPK.Optimizer, msg_lev=GLPK.MSG_ON))
)

# Define variables.
JuMP.@variable(
    optimization_problem,
    load_active_power_vector[electric_grid_index.load_names]
)
JuMP.@variable(
    optimization_problem,
    load_reactive_power_vector[electric_grid_index.load_names]
)
JuMP.@variable(
    optimization_problem,
    voltage_magnitude_vector[electric_grid_index.nodes_phases]
)
JuMP.@variable(
    optimization_problem,
    voltage_magnitude_per_unit_deviation_vector[electric_grid_index.nodes_phases]
)
