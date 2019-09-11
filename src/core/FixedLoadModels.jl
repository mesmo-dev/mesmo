"Fixed load models."
module FixedLoadModels

include("../config.jl")
import ..FLEDGE

"Fixed load model object."
struct FixedLoadModel
    load_active_power_timeseries::Array{Float64,1}
    load_reactive_power_timeseries::Array{Float64,1}
end

"Construct fixed load model object by electric grid data and load name."
function FixedLoadModel(
    electric_grid_data::FLEDGE.DatabaseInterface.ElectricGridData,
    load_name::String
)
    load_active_power_timeseries = zeros(Float64, 10)
    load_reactive_power_timeseries = zeros(Float64, 10)

    FixedLoadModel(
        load_active_power_timeseries,
        load_reactive_power_timeseries
    )
end
end
