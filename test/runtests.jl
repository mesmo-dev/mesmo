# FLEDGE tests.

# Enable debug message logging for FLEDGE.
ENV["JULIA_DEBUG"] = "FLEDGE"

import FLEDGE

import CSV
import DataFrames
import OpenDSSDirect
import PlotlyJS
import Plots
import SQLite
import SparseArrays
import Test

# Settings.
test_data_path = joinpath(@__DIR__, "data")
test_scenario_name = "singapore_6node"
Plots.plotlyjs()  # Select plotting backend.
Plots.default(size=(750, 500))
test_plots = false # If true, tests may produce plots.

# Load utility functions for tests.
include("utils.jl")

Test.@testset "FLEDGE tests" begin
    include("test_template.jl")
    include("test_database_interface.jl")
    include("test_electric_grid_models.jl")
    include("test_der_models.jl")
    include("test_power_flow_solvers.jl")
    include("test_api.jl")
end
