"Application programming interface."
module API

include("config.jl")
import ..FLEDGE

"Get timestep data for given `scenario_name`"
function get_timestep_data(scenario_name::String)
    scenario_data = (
        FLEDGE.DatabaseInterface.TimestepData(scenario_name)
    )
end

"""
Get electric grid data.

- Instantiates and returns electric grid data object for
  given `scenario_name`.
"""
function get_electric_grid_data(scenario_name::String)
    electric_grid_data = (
        FLEDGE.DatabaseInterface.ElectricGridData(scenario_name)
    )
    return electric_grid_data
end

"Get fixed load data for given `scenario_name`."
function get_fixed_load_data(scenario_name::String)
    fixed_load_data = (
        FLEDGE.DatabaseInterface.FixedLoadData(scenario_name)
    )

    return fixed_load_data
end

"""
Get electric grid model.

- Instantiates and returns electric grid model object for
  given `scenario_name`.
"""
function get_electric_grid_model(scenario_name::String)
    electric_grid_data = get_electric_grid_data(scenario_name)
    electric_grid_model = (
        FLEDGE.ElectricGridModels.ElectricGridModel(electric_grid_data)
    )

    return electric_grid_model
end

"""
Get linear electric grid model.

- Instantiates and returns linear electric grid model object for
  given `scenario_name`.
"""
function get_linear_electric_grid_model(scenario_name::String)
    # Obtain electric grid model.
    electric_grid_model = get_electric_grid_model(scenario_name)

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

    # Instantiate linear electric grid model.
    linear_electric_grid_model = (
        FLEDGE.ElectricGridModels.LinearElectricGridModel(
            electric_grid_model,
            nodal_voltage_vector,
            branch_power_vector_1,
            branch_power_vector_2
        )
    )

    return linear_electric_grid_model
end

"""
Initialize OpenDSS model.

- Instantiates OpenDSS model.
- No object is returned because the OpenDSS model lives in memory and
  can be accessed with the API of the `OpenDSS.jl` package.
"""
function initialize_open_dss_model(scenario_name::String)
    electric_grid_data = get_electric_grid_data(scenario_name)
    success = (
        FLEDGE.ElectricGridModels.initialize_open_dss_model(electric_grid_data)
    )

    return success
end

"""
Run operation problem.

- Instantiates and solves operation problem for the given `scenario_name`.
"""
function run_operation_problem(scenario_name::String)
    electric_grid_data = get_electric_grid_data(scenario_name)
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
