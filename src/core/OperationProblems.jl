"Operation problems."
module OperationProblems

include("../config.jl")
import ..FLEDGE

import JuMP
import GLPK

"Operation problem."
struct OperationProblem
    electric_grid_data::FLEDGE.DatabaseInterface.ElectricGridData
end

end
