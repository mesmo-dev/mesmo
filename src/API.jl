"Application programming interface."
module API

include("config.jl")
import ..FLEDGE

"""
Run operation problem.

- Instantiates and solves operation problem for the given `scenario_name`.
"""
function run_operation_problem(scenario_name::String)
    electric_grid_data = (
        FLEDGE.DatabaseInterface.ElectricGridData(scenario_name)
    )
    operation_problem = (
        FLEDGE.OperationProblems.OperationProblem(electric_grid_data)
    )

    success = true
    return success
end

# Everything in the API module is made available and exported
# the in FLEDGE main module.
include("export.jl")

end
