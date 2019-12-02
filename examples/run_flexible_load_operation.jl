"Example script for setting up and solving a flexible load operation problem."

import FLEDGE

import GLPK
import JuMP
import Logging

# Settings.
scenario_name = "singapore_6node"

# Get data.
timestep_data = FLEDGE.get_timestep_data(scenario_name)
flexible_load_data = FLEDGE.get_flexible_load_data(scenario_name)
price_data = FLEDGE.get_price_data(scenario_name)

# Get model.
load_name = flexible_load_data.flexible_loads[1, :load_name] # Take first load.
flexible_load_model = (
    FLEDGE.DERModels.GenericFlexibleLoadModel(
        flexible_load_data,
        load_name
    )
)

# Instantiate optimization problem.
optimization_problem = (
    JuMP.Model(JuMP.with_optimizer(GLPK.Optimizer, msg_lev=GLPK.MSG_ON))
)

# Define variables.
JuMP.@variable(
    optimization_problem,
    state_vector[
        flexible_load_model.state_names,
        timestep_data.timesteps
    ]
)
JuMP.@variable(
    optimization_problem,
    control_vector[
        flexible_load_model.control_names,
        timestep_data.timesteps
    ]
)
JuMP.@variable(
    optimization_problem,
    output_vector[
        flexible_load_model.output_names,
        timestep_data.timesteps
    ]
)

# Define constraints.
JuMP.@constraint(
    optimization_problem,
    state_equation[timestep = timestep_data.timesteps[1:end-1]],
    state_vector[:, timestep + timestep_data.timestep_interval_seconds].data .== (
        flexible_load_model.state_matrix
        * state_vector[:, timestep].data
        + flexible_load_model.control_matrix
        * control_vector[:, timestep].data
        + flexible_load_model.disturbance_matrix
        * transpose(values(flexible_load_model.disturbance_timeseries[timestep]))
    )
)
JuMP.@constraint(
    optimization_problem,
    output_equation[timestep = timestep_data.timesteps],
    output_vector[:, timestep].data .== (
        flexible_load_model.state_output_matrix
        * state_vector[:, timestep].data
        + flexible_load_model.control_output_matrix
        * control_vector[:, timestep].data
        + flexible_load_model.disturbance_output_matrix
        * transpose(values(flexible_load_model.disturbance_timeseries[timestep]))
    )
)
JuMP.@constraint(
    optimization_problem,
    output_minimum[timestep = timestep_data.timesteps],
    output_vector[:, timestep].data .>= (
        transpose(values(flexible_load_model.output_minimum_timeseries[timestep]))
    )
)
JuMP.@constraint(
    optimization_problem,
    output_maximum[timestep = timestep_data.timesteps],
    output_vector[:, timestep].data .<= (
        transpose(values(flexible_load_model.output_maximum_timeseries[timestep]))
    )
)

# Define objective.
price_name = "energy"
JuMP.@objective(
    optimization_problem,
    Min,
    sum(sum([
        (
            -1.0
            * values(price_data.price_timeseries_dict[price_name][:price_value][timestep])
            .* output_vector[:, timestep].data
        )
        for timestep = timestep_data.timesteps
    ]))
)

# Solve problem.
Logging.@info("Solving optimization problem.", optimization_problem)
JuMP.optimize!(optimization_problem)

# Get results.
optimization_termination_status = JuMP.termination_status(optimization_problem)
Logging.@info("", optimization_termination_status)
