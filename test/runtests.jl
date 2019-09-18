# FLEDGE tests.

import FLEDGE

import CSV
import DataFrames
import Memento
import OpenDSSDirect
import Plots
import Plotly
import SQLite
import SparseArrays
import Test

# Test data directory path and default test scenario.
test_data_path = joinpath(@__DIR__, "data")
test_scenario_name = "singapore_6node"

# Create test logger and set logging level.
test_logger = Memento.getlogger("FLEDGE tests")
Memento.setlevel!(test_logger, "debug")

# Define plotting backend and default settings.
Plots.plotly()
Plots.default(size=(750, 500))
test_plots = false # If true, tests may produce plots.

# Utility functions for tests.
include("utils.jl")

Test.@testset "FLEDGE tests" begin
    include("test_template.jl")
    include("test_database_interface.jl")
    include("test_electric_grid_models.jl")
    include("test_fixed_load_models.jl")
    include("test_ev_charger_models.jl")
    include("test_flexible_load_models.jl")
    include("test_power_flow_solvers.jl")
    include("test_api.jl")
end
