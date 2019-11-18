"Distributed energy resource (DER) models."
module DERModels

include("../config.jl")
import ..FLEDGE

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
    )
    reactive_power_nominal_timeseries = (
        fixed_load_data.fixed_load_timeseries_dict[
            fixed_load_data.fixed_loads[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* fixed_load_data.fixed_loads[:scaling_factor][load_index][1]
        .* fixed_load_data.fixed_loads[:reactive_power][load_index][1]
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
    )
    reactive_power_nominal_timeseries = (
        ev_charger_data.ev_charger_timeseries_dict[
            ev_charger_data.ev_chargers[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* ev_charger_data.ev_chargers[:scaling_factor][load_index][1]
        .* ev_charger_data.ev_chargers[:reactive_power][load_index][1]
    )

    EVChargerModel(
        active_power_nominal_timeseries,
        reactive_power_nominal_timeseries
    )
end

"Flexible load model object."
struct FlexibleLoadModel <: DERModel
    active_power_nominal_timeseries::TimeSeries.TimeArray
    reactive_power_nominal_timeseries::TimeSeries.TimeArray
end

"Construct flexible load model object by flexible load data and load name."
function FlexibleLoadModel(
    flexible_load_data::FLEDGE.DatabaseInterface.FlexibleLoadData,
    load_name::String
)
    # Get load index by `load_name`.
    load_index = (
        load_name .== flexible_load_data.flexible_loads[:load_name]
    )

    # Construct active and reactive power timeseries.
    active_power_nominal_timeseries = (
        flexible_load_data.flexible_load_timeseries_dict[
            flexible_load_data.flexible_loads[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* flexible_load_data.flexible_loads[:scaling_factor][load_index][1]
        .* flexible_load_data.flexible_loads[:active_power][load_index][1]
    )
    reactive_power_nominal_timeseries = (
        flexible_load_data.flexible_load_timeseries_dict[
            flexible_load_data.flexible_loads[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* flexible_load_data.flexible_loads[:scaling_factor][load_index][1]
        .* flexible_load_data.flexible_loads[:reactive_power][load_index][1]
    )

    FlexibleLoadModel(
        active_power_nominal_timeseries,
        reactive_power_nominal_timeseries
    )
end

end