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
electric_grid_model = (
    FLEDGE.API.get_electric_grid_model(scenario_name)
)
linear_electric_grid_model = (
    FLEDGE.API.get_linear_electric_grid_model(scenario_name)
)
Logging.@info("", linear_electric_grid_model)

# Instantiate optimization problem.
optimization_problem = (
    JuMP.Model(JuMP.with_optimizer(GLPK.Optimizer, msg_lev=GLPK.MSG_ON))
)

# Define variables.
JuMP.@variable(
    optimization_problem,
    load_active_power_vector
)
JuMP.@variable(
    optimization_problem,
    load_reactive_power_vector
)
JuMP.@variable(
    optimization_problem,
    voltage_magnitude_vector
)
JuMP.@variable(
    optimization_problem,
    voltage_magnitude_per_unit_deviation_vector
)
