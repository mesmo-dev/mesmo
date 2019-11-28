"Example script for setting up and solving a flexible load operation problem."

import FLEDGE
include("../src/config.jl")

import GLPK
import JuMP

# Settings.
scenario_name = "singapore_6node"

# Get electric grid data.
electric_grid_data = FLEDGE.get_electric_grid_data(scenario_name)

# Instantiate optimization problem.
optimization_problem = (
    JuMP.Model(JuMP.with_optimizer(GLPK.Optimizer, msg_lev=GLPK.MSG_ON))
)

# Derive sets.
load_set = electric_grid_data.electric_grid_loads[:load_name]
display(load_set)

# Define variables.
JuMP.@variable(
    optimization_problem,
    1 <= load_active_power[load_set]
)
JuMP.@variable(
    optimization_problem,
    1 <= load_reactive_power[load_set]
)

# Define constraints.
JuMP.@constraint(
    optimization_problem,
    test_constraint[load = load_set],
    load_active_power[load] >= 2 * load_reactive_power[load]
)

# Define objective.
JuMP.@objective(
    optimization_problem,
    Min,
    sum(load_active_power) + sum(load_reactive_power)
)

# Solve problem.
JuMP.optimize!(optimization_problem)
display(JuMP.termination_status(optimization_problem))
display(JuMP.value.(load_active_power))
display(JuMP.dual.(test_constraint))

