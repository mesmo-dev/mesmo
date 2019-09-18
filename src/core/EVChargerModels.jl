"EV charger models."
module EVChargerModels

include("../config.jl")
import ..FLEDGE

import TimeSeries

"EV charger model object."
struct EVChargerModel
    load_active_power_timeseries::TimeSeries.TimeArray
    load_reactive_power_timeseries::TimeSeries.TimeArray
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
    load_active_power_timeseries = (
        ev_charger_data.ev_charger_timeseries_dict[
            ev_charger_data.ev_chargers[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* ev_charger_data.ev_chargers[:scaling_factor][load_index][1]
        .* ev_charger_data.ev_chargers[:active_power][load_index][1]
    )
    load_reactive_power_timeseries = (
        ev_charger_data.ev_charger_timeseries_dict[
            ev_charger_data.ev_chargers[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* ev_charger_data.ev_chargers[:scaling_factor][load_index][1]
        .* ev_charger_data.ev_chargers[:reactive_power][load_index][1]
    )

    EVChargerModel(
        load_active_power_timeseries,
        load_reactive_power_timeseries
    )
end

end
