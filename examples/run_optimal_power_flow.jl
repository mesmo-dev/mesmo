"Example script for setting up and solving an optimal power flow problem."

import FLEDGE

import GLPK
import GR
import JuMP
import Logging
import Plots
import Statistics
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

# Define derivative model parameters.
load_active_power_vector_nominal = (
    real.(electric_grid_model.load_power_vector_nominal)
)
load_reactive_power_vector_nominal = (
    imag.(electric_grid_model.load_power_vector_nominal)
)

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

# Define constraints.
JuMP.@constraint(
    optimization_problem,
    load_active_minimum_maximum,
    (
        0.5 .* load_active_power_vector_nominal
        .<=
        load_active_power_vector.data
        .<=
        1.5 .* load_active_power_vector_nominal
    )
)
JuMP.@constraint(
    optimization_problem,
    load_reactive_minimum_maximum,
    (
        0.5 .* load_reactive_power_vector_nominal
        .<=
        load_reactive_power_vector.data
        .<=
        1.5 .* load_reactive_power_vector_nominal
    )
)
JuMP.@constraint(
    optimization_problem,
    voltage_magnitude_equation,
    (
        voltage_magnitude_vector.data
        .==
        (
            abs.(electric_grid_model.nodal_voltage_vector_no_load)
            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_wye_active
            * electric_grid_model.load_incidence_wye_matrix
            * (-1.0 .* (load_active_power_vector.data - load_active_power_vector_nominal))
            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_wye_reactive
            * electric_grid_model.load_incidence_wye_matrix
            * (-1.0 .* (load_reactive_power_vector.data - load_reactive_power_vector_nominal))
            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_delta_active
            * electric_grid_model.load_incidence_delta_matrix
            * (-1.0 .* (load_active_power_vector.data - load_active_power_vector_nominal))
            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_delta_reactive
            * electric_grid_model.load_incidence_delta_matrix
            * (-1.0 .* (load_reactive_power_vector.data - load_reactive_power_vector_nominal))
        )
    )
)
JuMP.@constraint(
    optimization_problem,
    voltage_magnitude_equation_0,
    (
        voltage_magnitude_per_unit_deviation_vector.data
        .>=
        0.0
    )
)
JuMP.@constraint(
    optimization_problem,
    voltage_magnitude_equation_1,
    (
        voltage_magnitude_per_unit_deviation_vector.data
        .>=
        (
            voltage_magnitude_vector.data
            ./ abs.(electric_grid_model.nodal_voltage_vector_no_load)
            .- 1.0
        )
    )
)
JuMP.@constraint(
    optimization_problem,
    voltage_magnitude_equation_2,
    (
        voltage_magnitude_per_unit_deviation_vector.data
        .>=
        -1.0
        .* (
            voltage_magnitude_vector.data
            ./ abs.(electric_grid_model.nodal_voltage_vector_no_load)
            .- 1.0
        )
    )
)

# Define objective.
JuMP.@objective(
    optimization_problem,
    Min,
    sum(voltage_magnitude_per_unit_deviation_vector.data)
)

# Solve optimization problem.
Logging.@info("", optimization_problem)
JuMP.optimize!(optimization_problem)

# Get results.
optimization_termination_status = JuMP.termination_status(optimization_problem)
Logging.@info("", optimization_termination_status)

voltage_magnitude_vector_per_unit_result = (
    JuMP.value.(voltage_magnitude_vector.data)
    ./ abs.(electric_grid_model.nodal_voltage_vector_no_load)
)
Logging.@info("", Statistics.mean(voltage_magnitude_vector_per_unit_result))
load_active_power_vector_per_unit_result = (
    JuMP.value.(load_active_power_vector.data)
    ./ real.(electric_grid_model.load_power_vector_nominal)
)
Logging.@info("", Statistics.mean(load_active_power_vector_per_unit_result))
