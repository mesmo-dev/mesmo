"Example script for setting up and solving a flexible load operation problem."

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

# Get data.
timestep_data = FLEDGE.DatabaseInterface.TimestepData(scenario_name)
flexible_load_data = FLEDGE.DatabaseInterface.FlexibleLoadData(scenario_name)
price_data = FLEDGE.DatabaseInterface.PriceData(scenario_name)

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
# TODO: Define initial state in model.
JuMP.@constraint(
    optimization_problem,
    state_initial,
    state_vector[:, timestep_data.timesteps[1]].data .== 0.0
)
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
            .* output_vector[output_name, timestep]
        )
        for timestep = timestep_data.timesteps
        for output_name = ["active_power", "reactive_power"]
    ]))
)

# Solve problem.
Logging.@info("Solving optimization problem.", optimization_problem)
JuMP.optimize!(optimization_problem)

# Get results.
optimization_termination_status = JuMP.termination_status(optimization_problem)
Logging.@info("", optimization_termination_status)
output_vector_timeseries = (
    TimeSeries.TimeArray(
        timestep_data.timesteps,
        transpose(JuMP.value.(output_vector.data)),
        Symbol.(flexible_load_model.output_names)
    )
)

# Plot results.
Plots.plot(
    TimeSeries.rename(
        flexible_load_model.output_maximum_timeseries[[:accumulated_energy]],
        :accumulated_energy_maximum
    ),
    line = :steppost,
)
Plots.plot!(
    TimeSeries.rename(
        flexible_load_model.output_minimum_timeseries[[:accumulated_energy]],
        :accumulated_energy_minimum
    ),
    line = :steppost,
)
Plots.plot!(
    output_vector_timeseries[[:accumulated_energy]],
    line = :steppost
)
display(Plots.plot!(legend = :outertop))
Plots.plot(
    TimeSeries.rename(
        flexible_load_model.output_maximum_timeseries[[:active_power]],
        :active_power_maximum
    ),
    line = :steppost,
)
Plots.plot!(
    TimeSeries.rename(
        flexible_load_model.output_minimum_timeseries[[:active_power]],
        :active_power_minimum
    ),
    line = :steppost,
)
Plots.plot!(
    output_vector_timeseries[[:active_power]],
    line = :steppost
)
display(Plots.plot!(legend = :outertop))
Plots.plot(
    price_data.price_timeseries_dict[price_name][:price_value],
    line = :steppost
)
display(Plots.plot!(legend = :outertop))
