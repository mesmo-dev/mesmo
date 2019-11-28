"Distributed energy resource (DER) models."
module DERModels

include("../config.jl")
import ..FLEDGE

import DataFrames
import LinearAlgebra
import TimeSeries

"DER model type."
abstract type DERModel end

"Fixed load model object."
struct FixedLoadModel <: DERModel
    active_power_nominal_timeseries::TimeSeries.TimeArray
    reactive_power_nominal_timeseries::TimeSeries.TimeArray
end

"Construct fixed load model object by fixed load data and load name."
function FixedLoadModel(
    fixed_load_data::FLEDGE.DatabaseInterface.FixedLoadData,
    load_name::String
)
    # Get load index by `load_name`.
    load_index = (
        load_name .== fixed_load_data.fixed_loads[:load_name]
    )

    # Construct active and reactive power timeseries.
    active_power_nominal_timeseries = (
        fixed_load_data.fixed_load_timeseries_dict[
            fixed_load_data.fixed_loads[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* fixed_load_data.fixed_loads[:scaling_factor][load_index][1]
        .* fixed_load_data.fixed_loads[:active_power][load_index][1]
        .* -1.0 # Load / demand is negative.
    )
    reactive_power_nominal_timeseries = (
        fixed_load_data.fixed_load_timeseries_dict[
            fixed_load_data.fixed_loads[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* fixed_load_data.fixed_loads[:scaling_factor][load_index][1]
        .* fixed_load_data.fixed_loads[:reactive_power][load_index][1]
        .* -1.0 # Load / demand is negative.
    )

    FixedLoadModel(
        active_power_nominal_timeseries,
        reactive_power_nominal_timeseries
    )
end

"EV charger model object."
struct EVChargerModel <: DERModel
    active_power_nominal_timeseries::TimeSeries.TimeArray
    reactive_power_nominal_timeseries::TimeSeries.TimeArray
end

"Construct EV charger model object by EV charger load data and load name."
function EVChargerModel(
    ev_charger_data::FLEDGE.DatabaseInterface.EVChargerData,
    load_name::String
)
    # Get load index by `load_name`.
    load_index = (
        load_name .== ev_charger_data.ev_chargers[:load_name]
    )

    # Construct active and reactive power timeseries.
    active_power_nominal_timeseries = (
        ev_charger_data.ev_charger_timeseries_dict[
            ev_charger_data.ev_chargers[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* ev_charger_data.ev_chargers[:scaling_factor][load_index][1]
        .* ev_charger_data.ev_chargers[:active_power][load_index][1]
        .* -1.0 # Load / demand is negative.
    )
    reactive_power_nominal_timeseries = (
        ev_charger_data.ev_charger_timeseries_dict[
            ev_charger_data.ev_chargers[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* ev_charger_data.ev_chargers[:scaling_factor][load_index][1]
        .* ev_charger_data.ev_chargers[:reactive_power][load_index][1]
        .* -1.0 # Load / demand is negative.
    )

    EVChargerModel(
        active_power_nominal_timeseries,
        reactive_power_nominal_timeseries
    )
end

"Flexible load model type."
abstract type FlexibleLoadModel <: DERModel end

"Generic flexible load model object."
struct GenericFlexibleLoadModel <: FlexibleLoadModel
    active_power_nominal_timeseries::TimeSeries.TimeArray
    reactive_power_nominal_timeseries::TimeSeries.TimeArray
    index_states::Vector{String}
    index_controls::Vector{String}
    index_disturbances::Vector{String}
    index_outputs::Vector{String}
    state_matrix::Matrix{Float64}
    control_matrix::Matrix{Float64}
    disturbance_matrix::Matrix{Float64}
    state_output_matrix::Matrix{Float64}
    control_output_matrix::Matrix{Float64}
    disturbance_output_matrix::Matrix{Float64}
    disturbance_timeseries::TimeSeries.TimeArray
    output_maximum_timeseries::TimeSeries.TimeArray
    output_minimum_timeseries::TimeSeries.TimeArray
end

"Construct flexible load model object by flexible load data and load name."
function GenericFlexibleLoadModel(
    flexible_load_data::FLEDGE.DatabaseInterface.FlexibleLoadData,
    load_name::String
)
    # Get load index by `load_name`.
    load_index = (
        load_name .== flexible_load_data.flexible_loads[:load_name]
    )

    # Construct nominal active and reactive power timeseries.
    active_power_nominal_timeseries = (
        flexible_load_data.flexible_load_timeseries_dict[
            flexible_load_data.flexible_loads[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* flexible_load_data.flexible_loads[:scaling_factor][load_index][1]
        .* flexible_load_data.flexible_loads[:active_power][load_index][1]
        .* -1.0 # Load / demand is negative.
    )
    reactive_power_nominal_timeseries = (
        flexible_load_data.flexible_load_timeseries_dict[
            flexible_load_data.flexible_loads[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* flexible_load_data.flexible_loads[:scaling_factor][load_index][1]
        .* flexible_load_data.flexible_loads[:reactive_power][load_index][1]
        .* -1.0 # Load / demand is negative.
    )
    # Update column names for use when merging multiple timeseries.
    TimeSeries.rename!(
        active_power_nominal_timeseries,
        :active_power
    )
    TimeSeries.rename!(
        reactive_power_nominal_timeseries,
        :reactive_power
    )

    # Instantiate index vectors.
    index_states = Vector{String}()
    index_controls = ["active_power", "reactive_power"]
    index_disturbances = Vector{String}()
    index_outputs = ["active_power", "reactive_power"]

    # Instantiate state space matrices.
    state_matrix = (
        Matrix{Float64}(
            undef,
            length(index_states),
            length(index_states)
        )
    )
    control_matrix = (
        Matrix{Float64}(
            undef,
            length(index_states),
            length(index_controls)
        )
    )
    disturbance_matrix = (
        Matrix{Float64}(
            undef,
            length(index_states),
            length(index_disturbances)
        )
    )
    state_output_matrix = (
        Matrix{Float64}(
            undef,
            length(index_outputs),
            length(index_states)
        )
    )
    control_output_matrix = (
        Matrix{Float64}(
            LinearAlgebra.I,
            length(index_outputs),
            length(index_controls)
        )
    )
    disturbance_output_matrix = (
        Matrix{Float64}(
            undef,
            length(index_outputs),
            length(index_disturbances)
        )
    )

    # Instantiate disturbance timeseries.
    disturbance_timeseries = 0.0 .* active_power_nominal_timeseries

    # Construct output constraint timeseries.
    output_maximum_timeseries = (
        (
            1.0
            - flexible_load_data.flexible_loads[:power_decrease_percentage_maximum][1]
        )
        .* TimeSeries.merge(
            active_power_nominal_timeseries,
            reactive_power_nominal_timeseries
        )
    )
    output_minimum_timeseries = (
        (
            1.0
            + flexible_load_data.flexible_loads[:power_increase_percentage_maximum][1]
        )
        .* TimeSeries.merge(
            active_power_nominal_timeseries,
            reactive_power_nominal_timeseries
        )
    )

    GenericFlexibleLoadModel(
        active_power_nominal_timeseries,
        reactive_power_nominal_timeseries,
        index_states,
        index_controls,
        index_disturbances,
        index_outputs,
        state_matrix,
        control_matrix,
        disturbance_matrix,
        state_output_matrix,
        control_output_matrix,
        disturbance_output_matrix,
        disturbance_timeseries,
        output_maximum_timeseries,
        output_minimum_timeseries
    )
end

end