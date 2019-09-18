"Flexible load models."
module FlexibleLoadModels

include("../config.jl")
import ..FLEDGE

import TimeSeries

"Flexible load model object."
struct FlexibleLoadModel
    load_active_power_timeseries::TimeSeries.TimeArray
    load_reactive_power_timeseries::TimeSeries.TimeArray
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
    load_active_power_timeseries = (
        flexible_load_data.flexible_load_timeseries_dict[
            flexible_load_data.flexible_loads[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* flexible_load_data.flexible_loads[:scaling_factor][load_index][1]
        .* flexible_load_data.flexible_loads[:active_power][load_index][1]
    )
    load_reactive_power_timeseries = (
        flexible_load_data.flexible_load_timeseries_dict[
            flexible_load_data.flexible_loads[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* flexible_load_data.flexible_loads[:scaling_factor][load_index][1]
        .* flexible_load_data.flexible_loads[:reactive_power][load_index][1]
    )

    FlexibleLoadModel(
        load_active_power_timeseries,
        load_reactive_power_timeseries
    )
end

end
