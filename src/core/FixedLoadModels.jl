"Fixed load models."
module FixedLoadModels

include("../config.jl")
import ..FLEDGE

import TimeSeries

"Fixed load model object."
struct FixedLoadModel
    load_active_power_timeseries::TimeSeries.TimeArray
    load_reactive_power_timeseries::TimeSeries.TimeArray
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
    load_active_power_timeseries = (
        fixed_load_data.fixed_load_timeseries_dict[
            fixed_load_data.fixed_loads[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* fixed_load_data.fixed_loads[:scaling_factor][load_index][1]
        .* fixed_load_data.fixed_loads[:active_power][load_index][1]
    )
    load_reactive_power_timeseries = (
        fixed_load_data.fixed_load_timeseries_dict[
            fixed_load_data.fixed_loads[:timeseries_name][load_index][1]
        ][:apparent_power_per_unit]
        .* fixed_load_data.fixed_loads[:scaling_factor][load_index][1]
        .* fixed_load_data.fixed_loads[:reactive_power][load_index][1]
    )

    FixedLoadModel(
        load_active_power_timeseries,
        load_reactive_power_timeseries
    )
end

end
