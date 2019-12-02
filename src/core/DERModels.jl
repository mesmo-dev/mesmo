"Distributed energy resource (DER) models."
module DERModels

# TODO: Fix apparent power to active/reactive power ratio.

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
    # Get fixed load data by `load_name`.
    fixed_load = (
        fixed_load_data.fixed_loads[
            load_name .== fixed_load_data.fixed_loads[:load_name]
        , :]
    )

    # Construct active and reactive power timeseries.
    active_power_nominal_timeseries = (
        fixed_load_data.fixed_load_timeseries_dict[
            fixed_load[1, :timeseries_name]
        ][:apparent_power_per_unit]
        .* fixed_load[1, :scaling_factor]
        .* fixed_load[1, :active_power]
        .* -1.0 # Load / demand is negative.
    )
    reactive_power_nominal_timeseries = (
        fixed_load_data.fixed_load_timeseries_dict[
            fixed_load[1, :timeseries_name]
        ][:apparent_power_per_unit]
        .* fixed_load[1, :scaling_factor]
        .* fixed_load[1, :reactive_power]
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
    # Get EV charger data by `load_name`.
    ev_charger = (
        ev_charger_data.ev_chargers[
            load_name .== ev_charger_data.ev_chargers[:load_name]
        , :]
    )

    # Construct active and reactive power timeseries.
    active_power_nominal_timeseries = (
        ev_charger_data.ev_charger_timeseries_dict[
            ev_charger[1, :timeseries_name]
        ][:apparent_power_per_unit]
        .* ev_charger[1, :scaling_factor]
        .* ev_charger[1, :active_power]
        .* -1.0 # Load / demand is negative.
    )
    reactive_power_nominal_timeseries = (
        ev_charger_data.ev_charger_timeseries_dict[
            ev_charger[1, :timeseries_name]
        ][:apparent_power_per_unit]
        .* ev_charger[1, :scaling_factor]
        .* ev_charger[1, :reactive_power]
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
    state_names::Vector{String}
    control_names::Vector{String}
    disturbance_names::Vector{String}
    output_names::Vector{String}
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
    # Get flexible load data by `load_name`.
    flexible_load = (
        flexible_load_data.flexible_loads[
            load_name .== flexible_load_data.flexible_loads[:load_name]
        , :]
    )

    # Construct nominal active and reactive power timeseries.
    active_power_nominal_timeseries = (
        flexible_load_data.flexible_load_timeseries_dict[
            flexible_load[1, :timeseries_name]
        ][:, :apparent_power_per_unit]
        .* flexible_load[1, :scaling_factor]
        .* flexible_load[1, :active_power]
        .* -1.0 # Load / demand is negative.
    )
    reactive_power_nominal_timeseries = (
        flexible_load_data.flexible_load_timeseries_dict[
            flexible_load[1, :timeseries_name]
        ][:, :apparent_power_per_unit]
        .* flexible_load[1, :scaling_factor]
        .* flexible_load[1, :reactive_power]
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

    # Calculate nominal accumulated energy timeseries.
    accumulated_energy_nominal_timeseries = (
        TimeSeries.rename(
            TimeSeries.cumsum(active_power_nominal_timeseries, dims=1),
            :accumulated_energy
        )
    )

    # Instantiate index vectors.
    state_names = ["accumulated_energy"]
    control_names = ["active_power", "reactive_power"]
    disturbance_names = Vector{String}()
    output_names = ["accumulated_energy", "active_power", "reactive_power"]

    # Instantiate state space matrices.
    state_matrix = (
        Matrix{Float64}(
            undef,
            length(state_names),
            length(state_names)
        )
    )
    state_matrix[1, 1] = 1.0
    control_matrix = (
        Matrix{Float64}(
            undef,
            length(state_names),
            length(control_names)
        )
    )
    control_matrix[1, 1] = 1.0
    disturbance_matrix = (
        Matrix{Float64}(
            undef,
            length(state_names),
            length(disturbance_names)
        )
    )
    state_output_matrix = (
        Matrix{Float64}(
            undef,
            length(output_names),
            length(state_names)
        )
    )
    state_output_matrix[1, 1] = 1.0
    control_output_matrix = (
        Matrix{Float64}(
            undef,
            length(output_names),
            length(control_names)
        )
    )
    control_output_matrix[2, 1] = 1.0
    control_output_matrix[3, 2] = 1.0
    disturbance_output_matrix = (
        Matrix{Float64}(
            undef,
            length(output_names),
            length(disturbance_names)
        )
    )

    # Instantiate disturbance timeseries.
    disturbance_timeseries = (
        TimeSeries.TimeArray(
            TimeSeries.timestamp(active_power_nominal_timeseries),
            Matrix{Float64}(
                undef,
                length(TimeSeries.timestamp(active_power_nominal_timeseries)),
                length(disturbance_names)
            ),
            Symbol.(disturbance_names)
        )
    )

    # Construct output constraint timeseries
    # TODO: Fix issue with accumulated energy constraint.
    output_maximum_timeseries = (
        TimeSeries.merge(
            (
                accumulated_energy_nominal_timeseries .* 0.0
            ),
            (
                (1.0 - flexible_load[1, :power_decrease_percentage_maximum])
                .* active_power_nominal_timeseries
            ),
            (
                (1.0 - flexible_load[1, :power_decrease_percentage_maximum])
                .* reactive_power_nominal_timeseries
            )
        )
    )
    output_minimum_timeseries = (
        TimeSeries.merge(
            (
                (accumulated_energy_nominal_timeseries .* 0.0)
                .+ values(accumulated_energy_nominal_timeseries[:accumulated_energy][end])
            ),
            (
                (1.0 + flexible_load[1, :power_increase_percentage_maximum])
                .* active_power_nominal_timeseries
            ),
            (
                (1.0 + flexible_load[1, :power_increase_percentage_maximum])
                .* reactive_power_nominal_timeseries
            )
        )
    )

    GenericFlexibleLoadModel(
        active_power_nominal_timeseries,
        reactive_power_nominal_timeseries,
        state_names,
        control_names,
        disturbance_names,
        output_names,
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
