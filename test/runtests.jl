# FLEDGE tests.

# Enable debug message logging for FLEDGE.
ENV["JULIA_DEBUG"] = "FLEDGE"

import FLEDGE

import CSV
import DataFrames
import GR
import OpenDSSDirect
import Plots
import SQLite
import SparseArrays
import Test

# Settings.
test_data_path = joinpath(@__DIR__, "data")
scenario_name = "singapore_6node"
Plots.gr()  # Select plotting backend.
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
