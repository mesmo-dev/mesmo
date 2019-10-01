"Operation problems."
module OperationProblems

include("../config.jl")
import ..FLEDGE

import JuMP
import GLPK

"Operation problem."
struct OperationProblem
    electric_grid_model::FLEDGE.ElectricGridModels.ElectricGridModel
    linear_electric_grid_model::FLEDGE.ElectricGridModels.LinearElectricGridModel
    optimization_problem::JuMP.Model
end

"Instantiate operation problem for given `electric_grid_data`."
function OperationProblem(
    electric_grid_data::FLEDGE.DatabaseInterface.ElectricGridData
)
    # Obtain electric grid model.
    electric_grid_model = (
        FLEDGE.ElectricGridModels.ElectricGridModel(electric_grid_data)
    )

    # Obtain power flow solution for nominal loading conditions.
    nodal_voltage_vector = (
        FLEDGE.PowerFlowSolvers.get_voltage_fixed_point(electric_grid_model)
    )
    (
        branch_power_vector_1,
        branch_power_vector_2
    ) = (
        FLEDGE.PowerFlowSolvers.get_branch_power_fixed_point(
            electric_grid_model,
            nodal_voltage_vector
        )
    )

    # Obtain linear electric grid model.
    linear_electric_grid_model = (
        FLEDGE.ElectricGridModels.LinearElectricGridModel(
            electric_grid_model::FLEDGE.ElectricGridModels.ElectricGridModel,
            nodal_voltage_vector::Array{ComplexF64,1},
            branch_power_vector_1::Array{ComplexF64,1},
            branch_power_vector_2::Array{ComplexF64,1}
        )
    )

    # Instantiate optimization problem.
    optimization_problem = (
        JuMP.Model(JuMP.with_optimizer(GLPK.Optimizer))
    )
end

end
