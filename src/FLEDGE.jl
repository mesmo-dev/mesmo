"Flexible distribution grid demonstrator."
module FLEDGE
# TODO: Move TODOs out of docstrings.

# Configuration setup and initialization routines.
include("config.jl")

# Include / define all FLEDGE submodules.
include("Utils.jl")
include("DatabaseInterface.jl")
include("core/DERModels.jl")
include("core/ElectricGridModels.jl")
include("core/PowerFlowSolvers.jl")
include("core/OptimizationSolvers.jl")
include("core/OperationProblems.jl")
include("API.jl")

# Everything in the API module is made available and exported
# the in FLEDGE main module.
using .API
include("export.jl")

end
