"Power flow solvers."
module PowerFlowSolvers
# TODO: Check development status of https://github.com/JuliaComputing/MKL.jl and
#       move from OpenBLAS to MKL in future.
# TODO: Consider pre-factorization of admittance if more performance is needed.

include("../config.jl")
import ..FLEDGE

import LinearAlgebra
import OpenDSSDirect
import SparseArrays

"""
Check conditions for fixed-point solution existence, uniqueness
and non-singularity for given power vector candidate and initial point.

- Conditions are formulated according to: <https://arxiv.org/pdf/1702.03310.pdf>
- Note the performance issues of this condition check algorithm due to the
  requirement for matrix inversions / solving of linear equations.
"""
function check_solution_conditions(
    nodal_admittance_matrix_no_source::SparseArrays.SparseMatrixCSC{ComplexF64,Int},
    nodal_transformation_matrix_no_source::SparseArrays.SparseMatrixCSC{Int,Int},
    nodal_power_vector_wye_initial_no_source::Array{ComplexF64,1},
    nodal_power_vector_delta_initial_no_source::Array{ComplexF64,1},
    nodal_power_vector_wye_candidate_no_source::Array{ComplexF64,1},
    nodal_power_vector_delta_candidate_no_source::Array{ComplexF64,1},
    nodal_voltage_vector_no_load_no_source::Array{ComplexF64,1},
    nodal_voltage_vector_initial_no_source::Array{ComplexF64,1},
)
    # Calculate norm of the initial nodal power vector.
    xi_initial = (
        maximum(sum(
            abs.(
                inv.(nodal_voltage_vector_no_load_no_source)
                .* (
                    nodal_admittance_matrix_no_source
                    \ (
                        (
                            inv.(nodal_voltage_vector_no_load_no_source)
                            .* nodal_power_vector_wye_initial_no_source
                        )
                    )
                )
            ),
            dims=2
        ))
        + maximum(sum(
            abs.(
                inv.(nodal_voltage_vector_no_load_no_source)
                .* (
                    nodal_admittance_matrix_no_source
                    \ Matrix(
                        (
                            transpose(nodal_transformation_matrix_no_source)
                            .* transpose(inv.(
                                abs.(nodal_transformation_matrix_no_source)
                                * abs.(nodal_voltage_vector_no_load_no_source)
                            ))
                        )
                        .* transpose(nodal_power_vector_delta_initial_no_source)
                    )
                )
            ),
            dims=2
        ))
    )

    # Calculate norm of the candidate nodal power vector.
    xi_candidate = (
        maximum(sum(
            abs.(
                inv.(nodal_voltage_vector_no_load_no_source)
                .* (
                    nodal_admittance_matrix_no_source
                    \ (
                        LinearAlgebra.Diagonal(
                            inv.(nodal_voltage_vector_no_load_no_source)
                            .* (
                                nodal_power_vector_wye_candidate_no_source
                                - nodal_power_vector_wye_initial_no_source
                            )
                        )
                    )
                )
            ),
            dims=2
        ))
        + maximum(sum(
            abs.(
                inv.(nodal_voltage_vector_no_load_no_source)
                .* (
                    nodal_admittance_matrix_no_source
                    \ Matrix(
                        (
                            transpose(nodal_transformation_matrix_no_source)
                            .* transpose(inv.(
                                abs.(nodal_transformation_matrix_no_source)
                                * abs.(nodal_voltage_vector_no_load_no_source)
                            ))
                        ) .* transpose(
                            nodal_power_vector_delta_candidate_no_source
                            - nodal_power_vector_delta_initial_no_source
                        )
                    )
                )
            ),
            dims=2
        ))
    )

    # Calculate norm of the initial nodal voltage vector. 
    gamma = (
        minimum([
            minimum(
                abs.(nodal_voltage_vector_initial_no_source)
                ./ abs.(nodal_voltage_vector_no_load_no_source)
            ),
            minimum(
                abs.(
                    nodal_transformation_matrix_no_source
                    * nodal_voltage_vector_initial_no_source
                )
                ./ (
                    abs.(nodal_transformation_matrix_no_source)
                    * abs.(nodal_voltage_vector_no_load_no_source)
                )
            )
        ])
    )

    # Obtain conditions for solution existence, uniqueness and non-singularity.
    condition_initial = (
        xi_initial
        < gamma ^ 2
    )
    condition_candidate = (
        xi_candidate
        < 0.25 * (((gamma ^ 2) - xi_initial) / gamma) ^ 2
    )
    is_valid = (
        condition_initial
        && condition_candidate
    )

    # If `condition_initial` is violated, the given initial nodal voltage vector
    # and power vectors are not valid. This suggests an error in the
    # problem setup and hence triggers a warning.
    if !condition_initial
        Memento.warn(
            _logger,
            "Fixed point solution condition is not satisfied for the"
            * " provided initial point."
        )
    end

    return is_valid
end

"""
Get nodal voltage vector by solving with the fixed point algorithm.

- Takes the nodal admittance, transformation matrices and
  nodal wye-power, delta-power, voltage vectors without source nodes, i.e.,
  source nodes must be removed from the arrays before passing to this function.
- Initial nodal wye-power, delta-power, voltage vectors must be a valid
  solution to te fixed-point equation, e.g., a previous solution from a past
  operation point.
- Fixed point equation according to: <https://arxiv.org/pdf/1702.03310.pdf>
"""
function get_voltage_fixed_point(
    nodal_admittance_matrix_no_source::SparseArrays.SparseMatrixCSC{ComplexF64,Int},
    nodal_transformation_matrix_no_source::SparseArrays.SparseMatrixCSC{Int,Int},
    nodal_power_vector_wye_no_source::Array{ComplexF64,1},
    nodal_power_vector_delta_no_source::Array{ComplexF64,1},
    nodal_power_vector_wye_initial_no_source::Array{ComplexF64,1},
    nodal_power_vector_delta_initial_no_source::Array{ComplexF64,1},
    nodal_voltage_vector_no_load_no_source::Array{ComplexF64,1},
    nodal_voltage_vector_initial_no_source::Array{ComplexF64,1};
    outer_iteration_limit = 100,
    power_candidate_iteration_limit = 100,
    power_candidate_reduction_factor = 0.5,
    voltage_iteration_limit=100,
    voltage_tolerance=1e-2
)
    # Copy initial nodal voltage vector to avoid changing outer scope variables.
    nodal_voltage_vector_initial_no_source = copy(
        nodal_voltage_vector_initial_no_source
    )
    nodal_voltage_vector_solution_no_source = copy(
        nodal_voltage_vector_initial_no_source
    )

    # Instantiate outer iteration variables.
    is_final = false
    outer_iteration = 1

    # Iterate between power vector candidate selection and fixed point
    # voltage solution algorithm until a final solution is found.
    while (
        !is_final
        && (outer_iteration < outer_iteration_limit)
    )
        # Define nodal power vector candidate to the desired nodal power vector.
        nodal_power_vector_wye_candidate_no_source = (
            nodal_power_vector_wye_no_source
        )
        nodal_power_vector_delta_candidate_no_source = (
            nodal_power_vector_delta_no_source
        )

        # Check solution conditions for nodal power vector candidate.
        is_final = check_solution_conditions(
            nodal_admittance_matrix_no_source,
            nodal_transformation_matrix_no_source,
            nodal_power_vector_wye_initial_no_source,
            nodal_power_vector_delta_initial_no_source,
            nodal_power_vector_wye_candidate_no_source,
            nodal_power_vector_delta_candidate_no_source,
            nodal_voltage_vector_no_load_no_source,
            nodal_voltage_vector_initial_no_source,
        )

        # Instantiate power candidate iteration variable.
        power_candidate_iteration = 1
        is_valid = copy(is_final)

        # If solution conditions are violated, iteratively reduce power
        # to find a power vector candidate which satisfies the solution
        # conditions.
        while (
            !is_valid
            && (power_candidate_iteration < power_candidate_iteration_limit)
        )
            nodal_power_vector_wye_candidate_no_source -= (
                power_candidate_reduction_factor
                * (
                    nodal_power_vector_wye_candidate_no_source
                    - nodal_power_vector_wye_initial_no_source
                )
            )
            nodal_power_vector_delta_candidate_no_source -= (
                power_candidate_reduction_factor
                * (
                    nodal_power_vector_delta_candidate_no_source
                    - nodal_power_vector_delta_initial_no_source
                )
            )

            is_valid = check_solution_conditions(
                nodal_admittance_matrix_no_source,
                nodal_transformation_matrix_no_source,
                nodal_power_vector_wye_initial_no_source,
                nodal_power_vector_delta_initial_no_source,
                nodal_power_vector_wye_candidate_no_source,
                nodal_power_vector_delta_candidate_no_source,
                nodal_voltage_vector_no_load_no_source,
                nodal_voltage_vector_initial_no_source,
            )
            power_candidate_iteration += 1
        end

        # Reaching the iteration limit is considered undesired and therefore
        # triggers a warning.
        if power_candidate_iteration == power_candidate_iteration_limit
            Memento.warn(
                _logger,
                "Power vector candidate selection algorithm reached maximum"
                * " limit of $power_candidate_iteration_limit iterations."
            )
        end

        # Store current candidate power vectors as initial power vectors 
        # for next round of computation of solution conditions. 
        nodal_power_vector_wye_initial_no_source = copy(
            nodal_power_vector_wye_candidate_no_source
        )
        nodal_power_vector_delta_initial_no_source = copy(
            nodal_power_vector_delta_candidate_no_source
        )

        # Instantiate fixed point iteration variables.
        voltage_iteration = 1
        voltage_change = Inf

        # Iterate fixed point equation until the solution quality is sufficient.
        while (
            (voltage_iteration < voltage_iteration_limit)
            && (voltage_change > voltage_tolerance)
        )
            # Calculate fixed point equation.
            nodal_voltage_vector_solution_no_source = copy(
                nodal_voltage_vector_initial_no_source
            )
            nodal_voltage_vector_solution_no_source = (
                nodal_voltage_vector_no_load_no_source
                + (
                    nodal_admittance_matrix_no_source
                    \ (
                        (
                            inv.(conj.(nodal_voltage_vector_solution_no_source))
                            .* conj.(nodal_power_vector_wye_candidate_no_source)
                        )
                        + (
                            transpose(nodal_transformation_matrix_no_source)
                            * (
                                inv.(
                                    nodal_transformation_matrix_no_source
                                    * conj.(
                                        nodal_voltage_vector_solution_no_source
                                    )
                                )
                                .* conj.(
                                    nodal_power_vector_delta_candidate_no_source
                                )
                            )
                        )
                    )
                )
            )

            # Calculate voltage change from previous iteration.
            voltage_change = (
                maximum(abs.(
                    nodal_voltage_vector_solution_no_source
                    - nodal_voltage_vector_initial_no_source
                ))
            )

            # Set voltage solution as initial voltage for next iteration. 
            nodal_voltage_vector_initial_no_source = copy(
                nodal_voltage_vector_solution_no_source
            )

            # Increment voltage iteration counter.
            voltage_iteration += 1
        end

        # Increment outer iteration counter.
        outer_iteration += 1

        # Reaching the iteration limit is considered undesired and therefore
        # triggers a warning.
        if voltage_iteration == voltage_iteration_limit
            Memento.warn(
                _logger,
                "Fixed point voltage solution algorithm reached maximum"
                * " limit of $voltage_iteration_limit iterations."
            )
        end
    end

    # Reaching the iteration limit is considered undesired and therefore
    # triggers a warning.
    if outer_iteration == outer_iteration_limit
        Memento.warn(
            _logger,
            "'Get voltage vector' solution algorithm reached maximum"
            * " limit of $outer_iteration_limit iterations."
        )
    end

    return nodal_voltage_vector_solution_no_source
end

"""
Get nodal voltage vector by solving with the fixed point algorithm.

- Takes nodal wye-power, delta-power vectors as inputs.
- Obtains the nodal admittance, transformation matrices and
  inital nodal wye-power, delta-power and voltage vectors as well as
  nodal no-load voltage vector without source nodes from an
  `electric_grid_model` object.
- Assumes no-load conditions for initial nodal power and voltage vectors.
"""
function get_voltage_fixed_point(
    electric_grid_model::FLEDGE.ElectricGridModels.ElectricGridModel,
    nodal_power_vector_wye::Array{ComplexF64,1},
    nodal_power_vector_delta::Array{ComplexF64,1};
    options...
)
    # Obtain no-source variables for fixed point equation.
    nodal_admittance_matrix_no_source = (
        electric_grid_model.nodal_admittance_matrix[
            electric_grid_model.index.node_by_node_type["no_source"],
            electric_grid_model.index.node_by_node_type["no_source"]
        ]
    )
    nodal_transformation_matrix_no_source = (
        electric_grid_model.nodal_transformation_matrix[
            electric_grid_model.index.node_by_node_type["no_source"],
            electric_grid_model.index.node_by_node_type["no_source"]
        ]
    )
    nodal_power_vector_wye_no_source = (
        (
            nodal_power_vector_wye
        )[
            electric_grid_model.index.node_by_node_type["no_source"]
        ]
    )
    nodal_power_vector_delta_no_source = (
        (
            nodal_power_vector_delta
        )[
            electric_grid_model.index.node_by_node_type["no_source"]
        ]
    )
    nodal_voltage_vector_no_load_no_source = (
        electric_grid_model.nodal_voltage_vector_no_load[
            electric_grid_model.index.node_by_node_type["no_source"]
        ]
    )

    # Define initial nodal power and voltage vectors as no-load conditions.
    nodal_power_vector_wye_initial_no_source = (
        zeros(ComplexF64, size(nodal_power_vector_wye_no_source))
    )
    nodal_power_vector_delta_initial_no_source = (
        zeros(ComplexF64, size(nodal_power_vector_delta_no_source))
    )
    nodal_voltage_vector_initial_no_source = (
        nodal_voltage_vector_no_load_no_source
    )

    # Get fixed point solution.
    nodal_voltage_vector_solution = get_voltage_fixed_point(
        nodal_admittance_matrix_no_source,
        nodal_transformation_matrix_no_source,
        nodal_power_vector_wye_no_source,
        nodal_power_vector_delta_no_source,
        nodal_power_vector_wye_initial_no_source,
        nodal_power_vector_delta_initial_no_source,
        nodal_voltage_vector_no_load_no_source,
        nodal_voltage_vector_initial_no_source;
        options...
    )

    # Get full voltage vector by concatenating source and calculated voltage.
    nodal_voltage_vector_solution = [
        electric_grid_model.nodal_voltage_vector_no_load[
            electric_grid_model.index.node_by_node_type["source"]
        ];
        nodal_voltage_vector_solution
    ]
    return nodal_voltage_vector_solution
end

"""
Get nodal voltage vector by solving with the fixed point algorithm.

- Obtains nodal power vectors assuming nominal loading conditions from an
  `electric_grid_model` object.
"""
function get_voltage_fixed_point(
    electric_grid_model::FLEDGE.ElectricGridModels.ElectricGridModel;
    options...
)
    # Obtain nodal power vectors assuming nominal loading conditions.
    nodal_power_vector_wye = (
        electric_grid_model.load_incidence_wye_matrix
        * electric_grid_model.load_power_vector_nominal
    )
    nodal_power_vector_delta = (
        electric_grid_model.load_incidence_delta_matrix
        * electric_grid_model.load_power_vector_nominal
    )

    # Get fixed point solution.
    nodal_voltage_vector_solution = get_voltage_fixed_point(
        electric_grid_model,
        nodal_power_vector_wye,
        nodal_power_vector_delta;
        options...
    )
    return nodal_voltage_vector_solution
end

"""
Get branch power vectors by calculating power flow with given nodal voltage.

- Returns two branch power vectors, where `branch_power_vector_1` represents the
  "from"-direction and `branch_power_vector_2` represents the "to"-direction.
- Nodal voltage vector is assumed to be obtained from fixed-point solution,
  therefore this function is associated with the fixed-point solver.
- This function directly takes branch admittance and incidence matrices as
  inputs, which can be obtained from an `electric_grid_model` object.
"""
function get_branch_power_fixed_point(
    branch_admittance_1_matrix::SparseArrays.SparseMatrixCSC{ComplexF64,Int},
    branch_admittance_2_matrix::SparseArrays.SparseMatrixCSC{ComplexF64,Int},
    branch_incidence_1_matrix::SparseArrays.SparseMatrixCSC{Int,Int},
    branch_incidence_2_matrix::SparseArrays.SparseMatrixCSC{Int,Int},
    nodal_voltage_vector::Array{ComplexF64,1}
)
    # Calculate branch power vectors.
    branch_power_vector_1 = (
        (
            branch_incidence_1_matrix
            * nodal_voltage_vector
        )
        .* conj.(
            branch_admittance_1_matrix
            * nodal_voltage_vector
        )
    )
    branch_power_vector_2 = (
        (
            branch_incidence_2_matrix
            * nodal_voltage_vector
        )
        .* conj.(
            branch_admittance_2_matrix
            * nodal_voltage_vector
        )
    )

    return (
        branch_power_vector_1,
        branch_power_vector_2
    )
end

"""
Get branch power vectors by calculating power flow with given nodal voltage.

- Obtains the needed matrices from an `electric_grid_model` object.
"""
function get_branch_power_fixed_point(
    electric_grid_model::FLEDGE.ElectricGridModels.ElectricGridModel,
    nodal_voltage_vector::Array{ComplexF64,1}
)
    # Obtain branch admittance and incidence matrices.
    branch_admittance_1_matrix = (
        electric_grid_model.branch_admittance_1_matrix
    )
    branch_admittance_2_matrix = (
        electric_grid_model.branch_admittance_2_matrix
    )
    branch_incidence_1_matrix = (
        electric_grid_model.branch_incidence_1_matrix
    )
    branch_incidence_2_matrix = (
        electric_grid_model.branch_incidence_2_matrix
    )

    # Calculate branch power vectors.
    (
        branch_power_vector_1,
        branch_power_vector_2
    ) = get_branch_power_fixed_point(
        branch_admittance_1_matrix,
        branch_admittance_2_matrix,
        branch_incidence_1_matrix,
        branch_incidence_2_matrix,
        nodal_voltage_vector
    )
end


"""
Get total electric losses with given nodal voltage.

- Nodal voltage vector is assumed to be obtained from fixed-point solution,
  therefore this function is associated with the fixed-point solver.
- This function directly takes the nodal admittance matrix as
  input, which can be obtained from an `electric_grid_model` object.
"""
function get_loss_fixed_point(
    nodal_admittance_matrix::SparseArrays.SparseMatrixCSC{ComplexF64,Int},
    nodal_voltage_vector::Array{ComplexF64,1}
)
    # TODO: Validate loss solution.
    # Calculate total losses.
    total_loss = (
        conj(
            transpose(nodal_voltage_vector)
            * (
                nodal_admittance_matrix
                * nodal_voltage_vector
            )
        )
    )

    return total_loss
end


"""
Get branch power vectors by calculating power flow with given nodal voltage.

- Obtains the nodal admittance matrix from an `electric_grid_model` object.
"""
function get_loss_fixed_point(
    electric_grid_model::FLEDGE.ElectricGridModels.ElectricGridModel,
    nodal_voltage_vector::Array{ComplexF64,1}
)
    # Obtain total losses with admittance matrix from electric grid model.
    total_loss = (
        get_loss_fixed_point(
            electric_grid_model.nodal_admittance_matrix,
            nodal_voltage_vector
        )
    )

    return total_loss
end

"""
Get nodal voltage vector by solving OpenDSS model.

- This expects an OpenDSS model to be readily set up with the desired
  power being set for all loads.
"""
function get_voltage_open_dss()
    # Solve OpenDSS model.
    OpenDSSDirect.dss("solve")

    # Extract nodal voltage vector.
    # - Voltages are sorted by node names in the fashion as nodes are sorted in
    #   nodes_phases in FLEDGE.ElectricGridModels.ElectricGridModelIndex().
    nodal_voltage_vector_solution = (
        OpenDSSDirect.Circuit.AllBusVolts()[
            sortperm(
                OpenDSSDirect.Circuit.AllNodeNames(),
                lt=FLEDGE.Utils.natural_less_than
            )
        ]
    )

    return nodal_voltage_vector_solution
end

end
