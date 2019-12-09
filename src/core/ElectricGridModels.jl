"Electric grid models."
module ElectricGridModels

include("../config.jl")
import ..FLEDGE

import DataFrames
import LinearAlgebra
import OpenDSSDirect
import Printf
import SparseArrays

"Utility function for taking the real valued conjugate of complex arrays."
function real_valued_conjugate(complex_array)
    conjugate_array = (
        vcat(
            hcat(real.(complex_array), imag.(complex_array)),
            hcat(imag.(complex_array), -real.(complex_array))
        )
    )
    return conjugate_array
end

"Electric grid index object."
struct ElectricGridIndex
    node_dimension::Int
    branch_dimension::Int
    load_dimension::Int
    phases::Vector{String}
    node_names::Vector{String}
    node_types::Vector{String}
    nodes_phases::Vector{Tuple{String,String}}
    line_names::Vector{String}
    transformer_names::Vector{String}
    branch_names::Vector{String}
    branch_types::Vector{String}
    branches_phases::Vector{Tuple{String,String}}
    load_names::Vector{String}
    node_by_node_name::Dict{String,Array{Int,1}}
    node_by_phase::Dict{String,Array{Int,1}}
    node_by_node_type::Dict{String,Array{Int,1}}
    node_by_load_name::Dict{String,Array{Int,1}}
    branch_by_line_name::Dict{String,Array{Int,1}}
    branch_by_transformer_name::Dict{String,Array{Int,1}}
    branch_by_phase::Dict{String,Array{Int,1}}
    load_by_load_name::Dict{String,Array{Int,1}}
    # TODO: Check if `load_by_load_name` is needed at all.
end

"Instantiate electric grid index object for given `electric_grid_data`"
function ElectricGridIndex(
    electric_grid_data::FLEDGE.DatabaseInterface.ElectricGridData
)
    # Define node dimension, i.e., number of phases of all nodes, which
    # will be the dimension of the nodal admittance matrix.
    # - The admittance matrix has one entry for each phase of each node in
    #   both dimensions.
    # - There cannot be "empty" dimensions for missing phases of nodes,
    #   because the matrix would become singular.
    # - Therefore the admittance matrix must have the exact number of existing
    #   phases of all nodes.
    node_dimension = (
        sum(Matrix(
            electric_grid_data.electric_grid_nodes[
            :,
                [
                    :is_phase_1_connected,
                    :is_phase_2_connected,
                    :is_phase_3_connected
                ]
            ]
        ))
    )

    # Define branch dimension, i.e., number of phases of all branches, which
    # will be the first dimension of the branch admittance matrices.
    # - Branches consider all power delivery elements, i.e., lines as well as
    #   transformers.
    # - The second dimension of the branch admittance matrices is the number of
    #   phases of all nodes.
    # - TODO: Add switches.
    branch_dimension = Int(
        sum(Matrix(
            electric_grid_data.electric_grid_lines[
                :,
                [
                    :is_phase_1_connected,
                    :is_phase_2_connected,
                    :is_phase_3_connected
                ]
            ]
        ))
        + sum(Matrix(
            electric_grid_data.electric_grid_transformers[
                :,
                [
                    :is_phase_1_connected,
                    :is_phase_2_connected,
                    :is_phase_3_connected
                ]
            ]
        ))
        / 2
    )

    # Define load dimension, i.e., number of all loads, which
    # will be the second dimension of the load incidence matrix.
    load_dimension = (
        size(electric_grid_data.electric_grid_loads, 1)
    )

    # Create `nodes` data frame, i.e., collection of all phases of all nodes
    # for generating indexing functions for the admittance matrix.
    nodes = DataFrames.DataFrame(
        [String, String, String],
        [:node_name, :phase, :node_type]
    )
    for node in eachrow(electric_grid_data.electric_grid_nodes)
        if (
            node[:node_name]
            ==
            electric_grid_data.electric_grids[1, :source_node_name]
        )
            node_type = "source"
        else
            node_type = "no_source"
        end
        if node[:is_phase_1_connected] == 1
            push!(
                nodes,
                [node[:node_name], "1", node_type]
            )
        end
        if node[:is_phase_2_connected] == 1
            push!(
                nodes,
                [node[:node_name], "2", node_type]
            )
        end
        if node[:is_phase_3_connected] == 1
            push!(
                nodes,
                [node[:node_name], "3", node_type]
            )
        end
    end
    # Sort `nodes` for `node_name` and `phase`.
    # - This is to ensure compatibility when comparing voltage solution with
    #   OpenDSS solution from FLEDGE.PowerFlowSolvers.get_voltage_open_dss().
    nodes = (
        nodes[
            sortperm(
                nodes[!, :node_name] .* nodes[!, :phase],
                lt=FLEDGE.Utils.natural_less_than
            ),
            1:end
        ]
    )

    # Create `branches` data frame, i.e., collection of phases of all branches
    # for generating indexing functions for the branch admittance matrices.
    # - Transformers must have same number of phases per winding and exactly
    #   two windings.
    branches = DataFrames.DataFrame(
        [String, String, String],
        [:branch_name, :phase, :branch_type]
    )
    for data_branches in [
        electric_grid_data.electric_grid_lines,
        electric_grid_data.electric_grid_transformers
    ]
        if :transformer_name in names(data_branches)
            branch_type = "transformer"
        else
            branch_type = "line"
        end
        for branch in eachrow(data_branches)
            if branch_type == "transformer"
                if branch[:winding] == 2
                    # Avoid duplicate branch definition for transformers by
                    # only taking primary side.
                    continue
                end
            end
            if branch[:is_phase_1_connected] == 1
                push!(
                    branches,
                    [branch[Symbol(branch_type * "_name")], "1", branch_type]
                )
            end
            if branch[:is_phase_2_connected] == 1
                push!(
                    branches,
                    [branch[Symbol(branch_type * "_name")], "2", branch_type]
                )
            end
            if branch[:is_phase_3_connected] == 1
                push!(
                    branches,
                    [branch[Symbol(branch_type * "_name")], "3", branch_type]
                )
            end
        end
    end

    # Create `loads` data frame for generating indexing
    # functions for the load incidence matrix.
    loads = (
        electric_grid_data.electric_grid_loads[!, :load_name]
    )

    # Define index vectors for various element types
    # for easier index definitions, e.g., in the optimization problem.
    phases = ["1", "2", "3"]
    node_names = (
        Vector(electric_grid_data.electric_grid_nodes[!, :node_name])
    )
    Logging.@info("", node_names)
    node_types = ["source", "no_source"]
    nodes_phases = (
        [
            (node[:node_name], node[:phase])
            for node in eachrow(nodes)
        ]
    )
    Logging.@info("", nodes_phases)
    line_names = (
        Vector(electric_grid_data.electric_grid_lines[!, :line_name])
    )
    transformer_names = (
        Vector(electric_grid_data.electric_grid_transformers[!, :transformer_name])
    )
    branch_names = (
        vcat(
            line_names,
            transformer_names
        )
    )
    branch_types = ["line", "transformer"]
    branches_phases = (
        [
            (branch[:branch_name], branch[:phase])
            for branch in eachrow(branches)
        ]
    )
    load_names = (
        Vector(electric_grid_data.electric_grid_loads[!, :load_name])
    )

    # Generate indexing dictionaries for the nodal admittance matrix,
    # i.e., for all phases of all nodes.
    # - This is a workaround to avoid low-performance string search operations
    #   for each indexing access.
    # - Instead, the appropriate boolean index vectors are pre-generated here
    #   and stored into data frames.

    # Index by node name.
    node_by_node_name = (
        Dict(
            key => Vector{Int}()
            for key in (
                node_names
            )
        )
    )
    for node_name in node_names
        node_by_node_name[node_name] = findall(
            nodes[!, :node_name] .== node_name
        )
    end
    # Index by phase.
    node_by_phase = (
        Dict(
            key => Vector{Int}()
            for key in (
                phases
            )
        )
    )
    for phase in phases
        node_by_phase[phase] = findall(
            nodes[!, :phase] .== phase
        )
    end
    # Index by node type.
    node_by_node_type = (
        Dict(
            key => Vector{Int}()
            for key in (
                node_types
            )
        )
    )
    for node_type in node_types
        node_by_node_type[node_type] = findall(
            nodes[!, :node_type] .== node_type
        )
    end
    # Index by load name.
    node_by_load_name = (
        Dict(
            key => Vector{Int}()
            for key in (
                load_names
            )
        )
    )
    for load in eachrow(electric_grid_data.electric_grid_loads)
        load_index = repeat([false], node_dimension)
        if load[:is_phase_1_connected] == 1
            load_index .|= (
                (nodes[!, :node_name] .== load[:node_name])
                .& (nodes[!, :phase] .== "1")
            )
        end
        if load[:is_phase_2_connected] == 1
            load_index .|= (
                (nodes[!, :node_name] .== load[:node_name])
                .& (nodes[!, :phase] .== "2")
            )
        end
        if load[:is_phase_3_connected] == 1
            load_index .|= (
                (nodes[!, :node_name] .== load[:node_name])
                .& (nodes[!, :phase] .== "3")
            )
        end
        node_by_load_name[load[:load_name]] = findall(
            load_index
        )
    end

    # Generate indexing dictionaries for the branch admittance matrices,
    # i.e., for all phases of all branches.
    # - This is a workaround to avoid low-performance string search operations
    #   for each indexing access.
    # - Instead, the appropriate boolean index vectors are pre-generated here
    #   and stored into data frames.

    # Index by line name.
    branch_by_line_name = (
        Dict(
            key => Vector{Int}()
            for key in (
                line_names
            )
        )
    )
    for line_name in line_names
        branch_by_line_name[line_name] = findall(
            (branches[!, :branch_name] .== line_name)
            .& (branches[!, :branch_type] .== "line")
        )
    end
    # Index by transformer name.
    branch_by_transformer_name = (
        Dict(
            key => Vector{Int}()
            for key in (
                transformer_names
            )
        )
    )
    for transformer_name in transformer_names
        branch_by_transformer_name[transformer_name] = findall(
            (branches[!, :branch_name] .== transformer_name)
            .& (branches[!, :branch_type] .== "transformer")
        )
    end
    # Index by phase.
    branch_by_phase = (
        Dict(
            key => Vector{Int}()
            for key in (
                phases
            )
        )
    )
    for phase in phases
        branch_by_phase[phase] = findall(
            branches[!, :phase] .== phase
        )
    end

    # Generate indexing dictionary for the load incidence matrix.

    # Index by load name.
    load_by_load_name = (
        Dict(
            key => Int[value]
            for (key, value) in zip(
                load_names,
                1:load_dimension
            )
        )
    )

    ElectricGridIndex(
        node_dimension,
        branch_dimension,
        load_dimension,
        phases,
        node_names,
        node_types,
        nodes_phases,
        line_names,
        transformer_names,
        branch_names,
        branch_types,
        branches_phases,
        load_names,
        node_by_node_name,
        node_by_phase,
        node_by_node_type,
        node_by_load_name,
        branch_by_line_name,
        branch_by_transformer_name,
        branch_by_phase,
        load_by_load_name
    )
end

"Instantiate electric grid index object for given `scenario_name`."
function ElectricGridIndex(scenario_name::String)
    # Obtain electric grid data.
    electric_grid_data = (
        FLEDGE.DatabaseInterface.ElectricGridData(scenario_name)
    )

    ElectricGridIndex(electric_grid_data)
end

"Electric grid model object."
struct ElectricGridModel
    electric_grid_data::FLEDGE.DatabaseInterface.ElectricGridData
    index::FLEDGE.ElectricGridModels.ElectricGridIndex
    nodal_admittance_matrix::SparseArrays.SparseMatrixCSC{ComplexF64,Int}
    nodal_transformation_matrix::SparseArrays.SparseMatrixCSC{Int,Int}
    branch_admittance_1_matrix::SparseArrays.SparseMatrixCSC{ComplexF64,Int}
    branch_admittance_2_matrix::SparseArrays.SparseMatrixCSC{ComplexF64,Int}
    branch_incidence_1_matrix::SparseArrays.SparseMatrixCSC{Int,Int}
    branch_incidence_2_matrix::SparseArrays.SparseMatrixCSC{Int,Int}
    load_incidence_wye_matrix::SparseArrays.SparseMatrixCSC{Float64,Int}
    load_incidence_delta_matrix::SparseArrays.SparseMatrixCSC{Int,Int}
    nodal_voltage_vector_no_load::Array{ComplexF64,1}
    load_power_vector_nominal::Array{ComplexF64,1}
end

"""
Instantiate electric grid model object for given `electric_grid_data`.

- The nodal no-load voltage vector can be constructed by
  1) `voltage_no_load_method="by_definition"`, i.e., the nodal voltage
  definition in the database is taken, or by
  2) `voltage_no_load_method="by_calculation"`, i.e., the no-load voltage is
  calculated from the source node voltage and the nodal admittance matrix.
"""
function ElectricGridModel(
    electric_grid_data::FLEDGE.DatabaseInterface.ElectricGridData;
    voltage_no_load_method="by_calculation"
)
    # TODO: Debug issues when changing / reordering node_name to random string.
    # TODO: Debug issues with delta-wye transformers or document limitations.

    # Obtain electric grid index object.
    index = ElectricGridIndex(electric_grid_data)

    # Define dictionaries for nodal admittance, nodal transformation,
    # branch admittance, branch incidence and load matrix matrix entries.
    # - For efficient construction, all matrix entries are first collected into
    #   dictionaries of row indexes, column indexes and values.
    # - The full matrices are constructed as sparse matrices later on.
    nodal_admittance_dictionary = Dict(
        :col_indexes => Vector{Int}(),
        :row_indexes => Vector{Int}(),
        :values => Vector{ComplexF64}()
    )
    nodal_transformation_dictionary = Dict(
        :col_indexes => Vector{Int}(),
        :row_indexes => Vector{Int}(),
        :values => Vector{Int}()
    )
    branch_admittance_1_dictionary = Dict(
        :col_indexes => Vector{Int}(),
        :row_indexes => Vector{Int}(),
        :values => Vector{ComplexF64}()
    )
    branch_admittance_2_dictionary = Dict(
        :col_indexes => Vector{Int}(),
        :row_indexes => Vector{Int}(),
        :values => Vector{ComplexF64}()
    )
    branch_incidence_1_dictionary = Dict(
        :col_indexes => Vector{Int}(),
        :row_indexes => Vector{Int}(),
        :values => Vector{Int}()
    )
    branch_incidence_2_dictionary = Dict(
        :col_indexes => Vector{Int}(),
        :row_indexes => Vector{Int}(),
        :values => Vector{Int}()
    )
    load_incidence_wye_dictionary = Dict(
        :col_indexes => Vector{Int}(),
        :row_indexes => Vector{Int}(),
        :values => Vector{Float64}()
    )
    load_incidence_delta_dictionary = Dict(
        :col_indexes => Vector{Int}(),
        :row_indexes => Vector{Int}(),
        :values => Vector{Int}()
    )
    # Define utility function to insert sub matrix into the matrix dictionaries
    # at given row/column indexes.
    function insert_sub_matrix!(
        matrix_dictionary,
        sub_matrix,
        row_indexes,
        col_indexes
    )
        for (col_index, value_col) in zip(col_indexes, eachcol(sub_matrix))
            for (row_index, value) in zip(row_indexes, value_col)
                if value != 0
                    push!(matrix_dictionary[:col_indexes], col_index)
                    push!(matrix_dictionary[:row_indexes], row_index)
                    push!(matrix_dictionary[:values], value)
                end
            end
        end
    end

    # Add lines to admittance, transformation and incidence matrices.
    for line in eachrow(electric_grid_data.electric_grid_lines)
        # Obtain line resistance and reactance matrix entries for the line.
        rxc_matrix_entries_index = (
            electric_grid_data.electric_grid_line_types_matrices[!, :line_type]
            .== line[:line_type]
        )
        r_matrix_entries = (
            electric_grid_data.electric_grid_line_types_matrices[
                rxc_matrix_entries_index,
                :r
            ]
        )
        x_matrix_entries = (
            electric_grid_data.electric_grid_line_types_matrices[
                rxc_matrix_entries_index,
                :x
            ]
        )
        c_matrix_entries = (
            electric_grid_data.electric_grid_line_types_matrices[
                rxc_matrix_entries_index,
                :c
            ]
        )

        # Obtain the full line resistance and reactance matrices.
        # Data only contains upper half entries.
        rxc_matrix_full_index = [
            1 2 4;
            2 3 5;
            4 5 6
        ]
        # TODO: Remove usage of n_phases.
        rxc_matrix_full_index = (
            rxc_matrix_full_index[1:line[:n_phases], 1:line[:n_phases]]
        )
        r_matrix = r_matrix_entries[rxc_matrix_full_index]
        x_matrix = x_matrix_entries[rxc_matrix_full_index]
        c_matrix = c_matrix_entries[rxc_matrix_full_index]

        # Construct line series admittance matrix.
        series_admittance_matrix = inv(
            (r_matrix + 1im * x_matrix)
            * line[:length]
        )

        # Construct line shunt admittance.
        # Note: nF to Ω with X = 1 / (2π * f * C)
        # TODO: Check line shunt admittance.
        base_frequency = 60  # TODO: Define base frequency in the database
        shunt_admittance_matrix = (
            c_matrix
            * 2π * base_frequency * 10 ^ -9
            * 0.5im
            * line[:length]
        )

        # Construct line element admittance matrices according to:
        # https://doi.org/10.1109/TPWRS.2017.2728618
        admittance_matrix_11 = (
            series_admittance_matrix
            + shunt_admittance_matrix
        )
        admittance_matrix_12 = (
            - series_admittance_matrix
        )
        admittance_matrix_21 = (
            - series_admittance_matrix
        )
        admittance_matrix_22 = (
            series_admittance_matrix
            + shunt_admittance_matrix
        )

        # Obtain indexes for positioning the line element matrices
        # in the full admittance matrices.
        node_index_1 = (
            index.node_by_node_name[line[:node_1_name]]
        )
        node_index_2 = (
            index.node_by_node_name[line[:node_2_name]]
        )
        branch_index = (
            index.branch_by_line_name[line[:line_name]]
        )

        # Add line element matrices to the nodal admittance matrix.
        insert_sub_matrix!(
            nodal_admittance_dictionary,
            admittance_matrix_11,
            node_index_1,
            node_index_1
        )
        insert_sub_matrix!(
            nodal_admittance_dictionary,
            admittance_matrix_12,
            node_index_1,
            node_index_2
        )
        insert_sub_matrix!(
            nodal_admittance_dictionary,
            admittance_matrix_21,
            node_index_2,
            node_index_1
        )
        insert_sub_matrix!(
            nodal_admittance_dictionary,
            admittance_matrix_22,
            node_index_2,
            node_index_2
        )

        # Add line element matrices to the branch admittance matrices.
        insert_sub_matrix!(
            branch_admittance_1_dictionary,
            admittance_matrix_11,
            branch_index,
            node_index_1
        )
        insert_sub_matrix!(
            branch_admittance_1_dictionary,
            admittance_matrix_12,
            branch_index,
            node_index_2
        )
        insert_sub_matrix!(
            branch_admittance_2_dictionary,
            admittance_matrix_21,
            branch_index,
            node_index_1
        )
        insert_sub_matrix!(
            branch_admittance_2_dictionary,
            admittance_matrix_22,
            branch_index,
            node_index_2
        )

        # Add line element matrices to the branch incidence matrices.
        insert_sub_matrix!(
            branch_incidence_1_dictionary,
            LinearAlgebra.Diagonal(ones(Int, length(branch_index))),
            branch_index,
            node_index_1
        )
        insert_sub_matrix!(
            branch_incidence_2_dictionary,
            LinearAlgebra.Diagonal(ones(Int, length(branch_index))),
            branch_index,
            node_index_2
        )
    end

    # Add transformers to admittance, transformation and incidence matrices.
    # - Note: This setup only works for transformers with exactly two windings
    #   and identical number of phases at each winding / side.

    # Define transformer factor matrices according to:
    # https://doi.org/10.1109/TPWRS.2017.2728618
    transformer_factors_1 = [
        1 0 0;
        0 1 0;
        0 0 1
    ]
    transformer_factors_2 = (
        1 / 3
        * [
            2 -1 -1;
            -1 2 -1;
            -1 -1 2
        ]
    )
    transformer_factors_3 = (
        1 / sqrt(3)
        * [
            -1 1 0;
            0 -1 1;
            1 0 -1
        ]
    )

    # Add transformers to admittance matrix.
    for transformer in eachrow(
        electric_grid_data.electric_grid_transformers[
            electric_grid_data.electric_grid_transformers[!, :winding] .== 1,
            1:end
        ]
    )
        # Obtain transformer windings.
        windings = (
            electric_grid_data.electric_grid_transformers[
                (
                    electric_grid_data.
                    electric_grid_transformers[!, :transformer_name]
                    .== transformer[:transformer_name]
                ),
                1:end
            ]
        )

        # Obtain primary and secondary voltage.
        voltage_1 = (
            electric_grid_data.electric_grid_nodes[
                (
                    electric_grid_data.electric_grid_nodes[!, :node_name]
                    .== windings[1, :node_name]
                ),
                :voltage
            ]
        )[1]
        voltage_2 = (
            electric_grid_data.electric_grid_nodes[
                (
                    electric_grid_data.electric_grid_nodes[!, :node_name]
                    .== windings[2, :node_name]
                ),
                :voltage
            ]
        )[1]

        # Obtain transformer type.
        type = (
            windings[1, :connection]
            * "-"
            * windings[2, :connection]
        )

        # Obtain transformer resistance and reactance.
        resistance_percentage = (
            transformer[:resistance_percentage]
        )
        reactance_percentage = (
            electric_grid_data.electric_grid_transformer_reactances[
                (
                    electric_grid_data.
                    electric_grid_transformer_reactances[!, :transformer_name]
                    .== transformer[:transformer_name]
                ),
                :reactance_percentage
            ]
        )[1]

        # Calculate transformer admittance.
        admittance = (
            inv(
                (
                    2 * resistance_percentage / 100
                    + 1im * reactance_percentage / 100
                )
                * (
                    voltage_2 ^ 2
                    / windings[1, :power]
                )
            )
        )

        # Calculate turn ratio.
        turn_ratio = (
            (
                1.0 # TODO: Replace `1.0` with actual tap position.
                *voltage_1
            )
            / (
                1.0 # TODO: Replace `1.0` with actual tap position.
                *voltage_2
            )
        )

        # Construct transformer element admittance matrices according to:
        # https://doi.org/10.1109/TPWRS.2017.2728618
        # - TODO: Add warning if wye-transformer is not grounded
        if type == "wye-wye"
            admittance_matrix_11 = (
                admittance
                * transformer_factors_1
                / turn_ratio ^ 2
            )
            admittance_matrix_12 = (
                - 1 * admittance
                * transformer_factors_1
                / turn_ratio
            )
            admittance_matrix_21 = (
                - 1 * admittance
                * transformer_factors_1
                / turn_ratio
            )
            admittance_matrix_22 = (
                admittance
                * transformer_factors_1
            )
        elseif type == "delta-wye"
            admittance_matrix_11 = (
                admittance
                * transformer_factors_2
                / turn_ratio ^ 2
            )
            admittance_matrix_12 = (
                - 1 * admittance
                * - 1 * transpose(transformer_factors_3)
                / turn_ratio
            )
            admittance_matrix_21 = (
                - 1 * admittance
                * - 1 * transformer_factors_3
                / turn_ratio
            )
            admittance_matrix_22 = (
                admittance
                * transformer_factors_1
            )
        elseif type == "wye-delta"
            admittance_matrix_11 = (
                admittance
                * transformer_factors_1
                / turn_ratio ^ 2
            )
            admittance_matrix_12 = (
                - 1 * admittance
                * - 1 * transformer_factors_3
                / turn_ratio
            )
            admittance_matrix_21 = (
                - 1 * admittance
                * - 1 * transpose(transformer_factors_3)
                / turn_ratio
            )
            admittance_matrix_22 = (
                admittance
                * transformer_factors_2
            )
        elseif type == "delta-delta"
            admittance_matrix_11 = (
                admittance
                * transformer_factors_2
                / turn_ratio ^ 2
            )
            admittance_matrix_12 = (
                - 1 * admittance
                * transformer_factors_2
                / turn_ratio
            )
            admittance_matrix_21 = (
                - 1 * admittance
                * transformer_factors_2
                / turn_ratio
            )
            admittance_matrix_22 = (
                admittance
                * transformer_factors_2
            )
        else
            Logging.@error(
                "Unknown transformer type: " * "$type"
            )
        end

        # Obtain indexes for positioning the transformer element
        # matrices in the full matrices.
        node_index_1 = (
            index.node_by_node_name[windings[1, :node_name]]
        )
        node_index_2 = (
            index.node_by_node_name[windings[2, :node_name]]
        )
        branch_index = (
            index.branch_by_transformer_name[transformer[:transformer_name]]
        )

        # Add transformer element matrices to the nodal admittance matrix.
        insert_sub_matrix!(
            nodal_admittance_dictionary,
            admittance_matrix_11,
            node_index_1,
            node_index_1
        )
        insert_sub_matrix!(
            nodal_admittance_dictionary,
            admittance_matrix_12,
            node_index_1,
            node_index_2
        )
        insert_sub_matrix!(
            nodal_admittance_dictionary,
            admittance_matrix_21,
            node_index_2,
            node_index_1
        )
        insert_sub_matrix!(
            nodal_admittance_dictionary,
            admittance_matrix_22,
            node_index_2,
            node_index_2
        )

        # Add transformer element matrices to the branch admittance matrices.
        insert_sub_matrix!(
            branch_admittance_1_dictionary,
            admittance_matrix_11,
            branch_index,
            node_index_1
        )
        insert_sub_matrix!(
            branch_admittance_1_dictionary,
            admittance_matrix_12,
            branch_index,
            node_index_2
        )
        insert_sub_matrix!(
            branch_admittance_2_dictionary,
            admittance_matrix_21,
            branch_index,
            node_index_1
        )
        insert_sub_matrix!(
            branch_admittance_2_dictionary,
            admittance_matrix_22,
            branch_index,
            node_index_2
        )

        # Add transformer element matrices to the branch incidence matrices.
        insert_sub_matrix!(
            branch_incidence_1_dictionary,
            LinearAlgebra.Diagonal(ones(Int, length(branch_index))),
            branch_index,
            node_index_1
        )
        insert_sub_matrix!(
            branch_incidence_2_dictionary,
            LinearAlgebra.Diagonal(ones(Int, length(branch_index))),
            branch_index,
            node_index_2
        )
    end

    # Define transformation matrix according to:
    # https://doi.org/10.1109/TPWRS.2018.2823277
    transformation_entries = [
        1 -1 0;
        0 1 -1;
        -1 0 1
    ]
    for node in eachrow(electric_grid_data.electric_grid_nodes)
        # Obtain node transformation matrix index.
        transformation_index = (
            findall(
                [
                    node[:is_phase_1_connected],
                    node[:is_phase_2_connected],
                    node[:is_phase_3_connected]
                ]
                .== 1
            )
        )

        # Construct node transformation matrix.
        transformation_matrix = (
            transformation_entries[transformation_index, transformation_index]
        )

        # Obtain index for positioning node transformation matrix in full
        # transformation matrix.
        node_index = (
            index.node_by_node_name[node[:node_name]]
        )

        # Add node transformation matrix to full transformation matrix.
        insert_sub_matrix!(
            nodal_transformation_dictionary,
            transformation_matrix,
            node_index,
            node_index
        )
    end

    # Add loads to load incidence matrix.
    for load in eachrow(electric_grid_data.electric_grid_loads)
        # Obtain load connection type.
        connection = load[:connection]

        # Obtain indexes for positioning load in incidence matrix.
        node_index = index.node_by_load_name[load[:load_name]]
        load_index = index.load_by_load_name[load[:load_name]]

        if connection == "wye"
            # Define incidence matrix entries.
            # - Wye loads are represented as balanced loads across all
            #   their connected phases.
            incidence_matrix = (
                - ones(Float64, length(node_index))
                ./ length(node_index)
            )
            insert_sub_matrix!(
                load_incidence_wye_dictionary,
                incidence_matrix,
                node_index,
                load_index
            )
        elseif connection == "delta"
            # Obtain phases of the delta load.
            phases = (
                findall(
                    [
                        load[:is_phase_1_connected],
                        load[:is_phase_2_connected],
                        load[:is_phase_3_connected]
                    ]
                    .== 1
                )
            )

            # Select connection node based on phase arrangement of delta load.
            # - Delta loads must be single-phase.
            if phases in ([1, 2], [2, 3])
                node_index = [node_index[1]]
            elseif phases == [1, 3]
                node_index = [node_index[2]]
            else
                Logging.@error(
                    "Unknown delta load phase arrangement: " * "$phases"
                )
            end

            # Define incidence matrix entry.
            # - Delta loads are assumed to be single-phase.
            incidence_matrix = [- 1]
            insert_sub_matrix!(
                load_incidence_wye_dictionary,
                incidence_matrix,
                node_index,
                load_index
            )
        else
            Logging.@error(
                "Unknown load connection type: " * "$connection"
            )
        end
    end

    # Construct sparse matrices for nodal admittance, nodal transformation,
    # branch admittance, branch incidence and load incidence matrices
    # from the respective dictionaries.
    nodal_admittance_matrix = SparseArrays.sparse(
        nodal_admittance_dictionary[:row_indexes],
        nodal_admittance_dictionary[:col_indexes],
        nodal_admittance_dictionary[:values],
        index.node_dimension,
        index.node_dimension
    )
    nodal_transformation_matrix = SparseArrays.sparse(
        nodal_transformation_dictionary[:row_indexes],
        nodal_transformation_dictionary[:col_indexes],
        nodal_transformation_dictionary[:values],
        index.node_dimension,
        index.node_dimension
    )
    branch_admittance_1_matrix = SparseArrays.sparse(
        branch_admittance_1_dictionary[:row_indexes],
        branch_admittance_1_dictionary[:col_indexes],
        branch_admittance_1_dictionary[:values],
        index.branch_dimension,
        index.node_dimension
    )
    branch_admittance_2_matrix = SparseArrays.sparse(
        branch_admittance_2_dictionary[:row_indexes],
        branch_admittance_2_dictionary[:col_indexes],
        branch_admittance_2_dictionary[:values],
        index.branch_dimension,
        index.node_dimension
    )
    branch_incidence_1_matrix = SparseArrays.sparse(
        branch_incidence_1_dictionary[:row_indexes],
        branch_incidence_1_dictionary[:col_indexes],
        branch_incidence_1_dictionary[:values],
        index.branch_dimension,
        index.node_dimension
    )
    branch_incidence_2_matrix = SparseArrays.sparse(
        branch_incidence_2_dictionary[:row_indexes],
        branch_incidence_2_dictionary[:col_indexes],
        branch_incidence_2_dictionary[:values],
        index.branch_dimension,
        index.node_dimension
    )
    load_incidence_wye_matrix = SparseArrays.sparse(
        load_incidence_wye_dictionary[:row_indexes],
        load_incidence_wye_dictionary[:col_indexes],
        load_incidence_wye_dictionary[:values],
        index.node_dimension,
        index.load_dimension
    )
    load_incidence_delta_matrix = SparseArrays.sparse(
        load_incidence_delta_dictionary[:row_indexes],
        load_incidence_delta_dictionary[:col_indexes],
        load_incidence_delta_dictionary[:values],
        index.node_dimension,
        index.load_dimension
    )

    # Construct no-load voltage vector for the grid.
    # - The nodal no-load voltage vector can be constructed by
    #   1) `voltage_no_load_method="by_definition"`, i.e., the nodal voltage
    #   definition in the database is taken, or by
    #   2) `voltage_no_load_method="by_calculation"`, i.e., the no-load voltage is
    #   calculated from the source node voltage and the nodal admittance matrix.
    # - TODO: Check if no-load voltage divide by sqrt(3) correct.
    nodal_voltage_vector_no_load = (
        zeros(ComplexF64, index.node_dimension)
    )
    # Define phase orientations.
    voltage_phase_factors = [
        exp(0 * 1im), # Phase 1.
        exp(- 2π / 3 * 1im), # Phase 2.
        exp(2π / 3 * 1im), # Phase 3.
    ]
    if voltage_no_load_method == "by_definition"
        for node in eachrow(electric_grid_data.electric_grid_nodes)
            # Obtain phases for node.
            phases = (
                findall(
                    [
                        node[:is_phase_1_connected],
                        node[:is_phase_2_connected],
                        node[:is_phase_3_connected]
                    ]
                    .== 1
                )
            )

            # Obtain node voltage level.
            voltage = node[:voltage]

            # Insert voltage into voltage vector.
            nodal_voltage_vector_no_load[
                index.node_by_node_name[node[:node_name]]
            ] = (
                voltage
                .* voltage_phase_factors[phases]
                ./ sqrt(3)
            )
        end
    elseif voltage_no_load_method == "by_calculation"
        # Obtain source node.
        node = first(
            electric_grid_data.electric_grid_nodes[
                (
                    electric_grid_data.electric_grid_nodes[!, :node_name]
                    .== electric_grid_data.electric_grids[1, :source_node_name]
                ),
                1:end
            ]
        )

        # Obtain phases for source node.
        phases = (
            findall(
                [
                    node[:is_phase_1_connected],
                    node[:is_phase_2_connected],
                    node[:is_phase_3_connected]
                ]
                .== 1
            )
        )

        # Obtain source node voltage level.
        voltage = node[:voltage]

        # Insert source node voltage into voltage vector.
        nodal_voltage_vector_no_load[index.node_by_node_type["source"]] = (
            voltage
            .* voltage_phase_factors[phases]
            ./ sqrt(3)
        )

        # Calculate all remaining no-load node voltages.
        nodal_voltage_vector_no_load[index.node_by_node_type["no_source"]] = (
            - nodal_admittance_matrix[
                index.node_by_node_type["no_source"],
                index.node_by_node_type["no_source"]
            ]
            \ (
                nodal_admittance_matrix[
                    index.node_by_node_type["no_source"],
                    index.node_by_node_type["source"]
                ]
                * nodal_voltage_vector_no_load[index.node_by_node_type["source"]]
            )
        )
    end

    # Construct nominal load power vector.
    load_power_vector_nominal = (
        electric_grid_data.electric_grids[1, :load_multiplier]
        .* Vector(
            electric_grid_data.electric_grid_loads[!, :active_power]
            + 1im .* electric_grid_data.electric_grid_loads[!, :reactive_power]
        )
    )

    ElectricGridModel(
        electric_grid_data,
        index,
        nodal_admittance_matrix,
        nodal_transformation_matrix,
        branch_admittance_1_matrix,
        branch_admittance_2_matrix,
        branch_incidence_1_matrix,
        branch_incidence_2_matrix,
        load_incidence_wye_matrix,
        load_incidence_delta_matrix,
        nodal_voltage_vector_no_load,
        load_power_vector_nominal
    )
end

"Instantiate electric grid model object for given `scenario_name`."
function ElectricGridModel(
    scenario_name::String;
    kwargs...
)
    # Obtain electric grid data.
    electric_grid_data = (
        FLEDGE.DatabaseInterface.ElectricGridData(scenario_name)
    )

    ElectricGridModel(electric_grid_data; kwargs...)
end

"Linear electric grid model object."
struct LinearElectricGridModel
    sensitivity_voltage_by_power_wye_active
    sensitivity_voltage_by_power_wye_reactive
    sensitivity_voltage_by_power_delta_active
    sensitivity_voltage_by_power_delta_reactive
    sensitivity_voltage_magnitude_by_power_wye_active
    sensitivity_voltage_magnitude_by_power_wye_reactive
    sensitivity_voltage_magnitude_by_power_delta_active
    sensitivity_voltage_magnitude_by_power_delta_reactive
    sensitivity_power_branch_from_by_power_wye_active
    sensitivity_power_branch_from_by_power_wye_reactive
    sensitivity_power_branch_from_by_power_delta_active
    sensitivity_power_branch_from_by_power_delta_reactive
    sensitivity_power_branch_to_by_power_wye_active
    sensitivity_power_branch_to_by_power_wye_reactive
    sensitivity_power_branch_to_by_power_delta_active
    sensitivity_power_branch_to_by_power_delta_reactive
    sensitivity_loss_active_by_power_wye_active
    sensitivity_loss_active_by_power_wye_reactive
    sensitivity_loss_active_by_power_delta_active
    sensitivity_loss_active_by_power_delta_reactive
    sensitivity_loss_reactive_by_power_wye_active
    sensitivity_loss_reactive_by_power_wye_reactive
    sensitivity_loss_reactive_by_power_delta_active
    sensitivity_loss_reactive_by_power_delta_reactive
    sensitivity_loss_active_by_noload_voltage
    sensitivity_loss_reactive_by_noload_voltage
    constant_loss_active_no_load
    constant_loss_reactive_no_load
end

"""
Construct linear electric grid model by electric grid model,
nodal voltage vector and branch power vectors.

- Expects valid nodal voltage vector solution with corresponding
  branch power vectors as input.
"""
function LinearElectricGridModel(
    electric_grid_model::FLEDGE.ElectricGridModels.ElectricGridModel,
    nodal_voltage_vector::Array{ComplexF64,1},
    branch_power_vector_1::Array{ComplexF64,1},
    branch_power_vector_2::Array{ComplexF64,1}
)
    # TODO: Rename sensitivity matrices to match global nomenclature.

    # Obtain no_source matrices and vectors.
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
    nodal_voltage_no_source = (
        nodal_voltage_vector[
            electric_grid_model.index.node_by_node_type["no_source"]
        ]
    )

    # Voltage
    sensitivity_voltage_by_power_wye_active = (
        SparseArrays.spzeros(
            ComplexF64,
            electric_grid_model.index.node_dimension,
            electric_grid_model.index.node_dimension,
        )
    )
    sensitivity_voltage_by_power_wye_active[
        electric_grid_model.index.node_by_node_type["no_source"],
        electric_grid_model.index.node_by_node_type["no_source"]
    ] = (
        nodal_admittance_matrix_no_source
        \ LinearAlgebra.Diagonal(conj.(inv.(nodal_voltage_no_source)))
    )

    sensitivity_voltage_by_power_wye_reactive = (
        SparseArrays.spzeros(
            ComplexF64,
            electric_grid_model.index.node_dimension,
            electric_grid_model.index.node_dimension,
        )
    )
    sensitivity_voltage_by_power_wye_reactive[
        electric_grid_model.index.node_by_node_type["no_source"],
        electric_grid_model.index.node_by_node_type["no_source"]
    ] = (
        - 1im * nodal_admittance_matrix_no_source
        \ LinearAlgebra.Diagonal(conj.(inv.(nodal_voltage_no_source)))
    )

    sensitivity_voltage_by_power_delta_active = (
        SparseArrays.spzeros(
            ComplexF64,
            electric_grid_model.index.node_dimension,
            electric_grid_model.index.node_dimension,
        )
    )
    # TODO: Currently requiring conversion to dense matrix for `ldiv!` to work.
    #       Check for alternatives or open issue.
    # TODO: Consider pre-factorization of admittance if performance is needed.
    sensitivity_voltage_by_power_delta_active[
        electric_grid_model.index.node_by_node_type["no_source"],
        electric_grid_model.index.node_by_node_type["no_source"]
    ] = (
        (
            nodal_admittance_matrix_no_source
            \ Matrix(transpose(nodal_transformation_matrix_no_source))
        )
        .* transpose(
            inv.(
                nodal_transformation_matrix_no_source
                * conj.(nodal_voltage_no_source)
            )
        )
    )

    sensitivity_voltage_by_power_delta_reactive = (
        SparseArrays.spzeros(
            ComplexF64,
            electric_grid_model.index.node_dimension,
            electric_grid_model.index.node_dimension,
        )
    )
    sensitivity_voltage_by_power_delta_reactive[
        electric_grid_model.index.node_by_node_type["no_source"],
        electric_grid_model.index.node_by_node_type["no_source"]
    ] = (
        (
            - 1im * nodal_admittance_matrix_no_source
            \ Matrix(transpose(nodal_transformation_matrix_no_source))
        )
        .* transpose(
            inv.(
                nodal_transformation_matrix_no_source
                * conj.(nodal_voltage_no_source)
            )
        )
    )

    sensitivity_voltage_magnitude_by_power_wye_active = (
        LinearAlgebra.Diagonal(abs.(inv.(nodal_voltage_vector)))
        * real.(
            LinearAlgebra.Diagonal(conj.(nodal_voltage_vector))
            * sensitivity_voltage_by_power_wye_active
        )
    )

    sensitivity_voltage_magnitude_by_power_wye_reactive = (
        LinearAlgebra.Diagonal(abs.(inv.(nodal_voltage_vector)))
        * real.(
            LinearAlgebra.Diagonal(conj.(nodal_voltage_vector))
            * sensitivity_voltage_by_power_wye_reactive
        )
    )

    sensitivity_voltage_magnitude_by_power_delta_active = (
        LinearAlgebra.Diagonal(abs.(inv.(nodal_voltage_vector)))
        * real.(
            LinearAlgebra.Diagonal(conj.(nodal_voltage_vector))
            * sensitivity_voltage_by_power_delta_active
        )
    )

    sensitivity_voltage_magnitude_by_power_delta_reactive = (
        LinearAlgebra.Diagonal(abs.(inv.(nodal_voltage_vector)))
        * real.(
            LinearAlgebra.Diagonal(conj.(nodal_voltage_vector))
            * sensitivity_voltage_by_power_delta_reactive
        )
    )

    # Power flows
    sensitivity_power_branch_from_by_voltage = (
        LinearAlgebra.Diagonal(conj.(
            electric_grid_model.branch_admittance_1_matrix
            * nodal_voltage_vector
        ))
        * electric_grid_model.branch_incidence_1_matrix
        + LinearAlgebra.Diagonal(
            electric_grid_model.branch_incidence_1_matrix
            * nodal_voltage_vector
        )
        * conj.(electric_grid_model.branch_admittance_1_matrix)
    )
    sensitivity_power_branch_to_by_voltage = (
        LinearAlgebra.Diagonal(conj.(
            electric_grid_model.branch_admittance_2_matrix
            * nodal_voltage_vector
        ))
        * electric_grid_model.branch_incidence_2_matrix
        + LinearAlgebra.Diagonal(
            electric_grid_model.branch_incidence_2_matrix
            * nodal_voltage_vector
        )
        * conj.(electric_grid_model.branch_admittance_2_matrix)
    )

    sensitivity_power_branch_from_by_power_wye_active = (
        2 .* hcat(
            LinearAlgebra.Diagonal(real.(branch_power_vector_1)),
            LinearAlgebra.Diagonal(imag.(branch_power_vector_1))
        )
        * vcat(
            real.(
                sensitivity_power_branch_from_by_voltage
                * conj.(sensitivity_voltage_by_power_wye_active)
            ),
            imag.(
                sensitivity_power_branch_from_by_voltage
                * conj.(sensitivity_voltage_by_power_wye_active)
            )
        )
    )
    sensitivity_power_branch_from_by_power_wye_reactive = (
        2 * hcat(
            LinearAlgebra.Diagonal(real.(branch_power_vector_1)),
            LinearAlgebra.Diagonal(imag.(branch_power_vector_1))
        )
        * vcat(
            real.(
                sensitivity_power_branch_from_by_voltage
                * conj(sensitivity_voltage_by_power_wye_reactive)
            ),
            imag.(
                sensitivity_power_branch_from_by_voltage
                * conj(sensitivity_voltage_by_power_wye_reactive)
            )
        )
    )

    sensitivity_power_branch_from_by_power_delta_active = (
        2 * hcat(
            LinearAlgebra.Diagonal(real.(branch_power_vector_1)),
            LinearAlgebra.Diagonal(imag.(branch_power_vector_1))
        )
        * vcat(
            real.(
                sensitivity_power_branch_from_by_voltage
                * conj.(sensitivity_voltage_by_power_delta_active)
            ),
            imag.(
                sensitivity_power_branch_from_by_voltage
                * conj.(sensitivity_voltage_by_power_delta_active)
            )
        )
    )
    sensitivity_power_branch_from_by_power_delta_reactive = (
        2 * hcat(
            LinearAlgebra.Diagonal(real.(branch_power_vector_1)),
            LinearAlgebra.Diagonal(imag.(branch_power_vector_1))
        )
        * vcat(
            real.(
                sensitivity_power_branch_from_by_voltage
                * conj.(sensitivity_voltage_by_power_delta_reactive)
            ),
            imag.(
                sensitivity_power_branch_from_by_voltage
                * conj.(sensitivity_voltage_by_power_delta_reactive)
            )
        )
    )

    sensitivity_power_branch_to_by_power_wye_active = (
        2 * hcat(
            LinearAlgebra.Diagonal(real.(branch_power_vector_2)),
            LinearAlgebra.Diagonal(imag.(branch_power_vector_2))
        )
        * vcat(
            real.(
                sensitivity_power_branch_to_by_voltage
                * conj.(sensitivity_voltage_by_power_wye_active)
            ),
            imag.(
                sensitivity_power_branch_to_by_voltage
                * conj.(sensitivity_voltage_by_power_wye_active)
            )
        )
    )
    sensitivity_power_branch_to_by_power_wye_reactive = (
        2 * hcat(
            LinearAlgebra.Diagonal(real.(branch_power_vector_2)),
            LinearAlgebra.Diagonal(imag.(branch_power_vector_2))
        )
        * vcat(
            real.(
                sensitivity_power_branch_to_by_voltage
                * conj.(sensitivity_voltage_by_power_wye_reactive)
            ),
            imag.(
                sensitivity_power_branch_to_by_voltage
                * conj.(sensitivity_voltage_by_power_wye_reactive)
            )
        )
    )

    sensitivity_power_branch_to_by_power_delta_active = (
        2 * hcat(
            LinearAlgebra.Diagonal(real.(branch_power_vector_2)),
            LinearAlgebra.Diagonal(imag.(branch_power_vector_2))
        )
        * vcat(
            real.(
                sensitivity_power_branch_to_by_voltage
                * conj.(sensitivity_voltage_by_power_delta_active)
            ),
            imag.(
                sensitivity_power_branch_to_by_voltage
                * conj.(sensitivity_voltage_by_power_delta_active)
            )
        )
    )
    sensitivity_power_branch_to_by_power_delta_reactive = (
        2 * hcat(
            LinearAlgebra.Diagonal(real.(branch_power_vector_2)),
            LinearAlgebra.Diagonal(imag.(branch_power_vector_2))
        )
        * vcat(
            real.(
                sensitivity_power_branch_to_by_voltage
                * conj.(sensitivity_voltage_by_power_delta_reactive)
            ),
            imag.(
                sensitivity_power_branch_to_by_voltage
                * conj.(sensitivity_voltage_by_power_delta_reactive)
            )
        )
    )

    # Losses
    # Second rows are the imaginary part, sum of which can be proven to be zero!
    sensitivity_loss_active_by_voltage = (
        real_valued_conjugate(
            conj.(
                transpose(nodal_voltage_vector)
                * real.(electric_grid_model.nodal_admittance_matrix)
            )
            + transpose(conj.(
                real.(electric_grid_model.nodal_admittance_matrix)
                * nodal_voltage_vector
            ))
        )
    )[1, :]
    sensitivity_loss_reactive_by_voltage = (
        real_valued_conjugate(
            conj.(
                transpose(nodal_voltage_vector)
                * imag.(- electric_grid_model.nodal_admittance_matrix)
            )
            + transpose(conj.(
                imag.(- electric_grid_model.nodal_admittance_matrix)
                * nodal_voltage_vector
            ))
        )
    )[1, :]

    sensitivity_loss_active_by_power_wye_active = (
        transpose(sensitivity_loss_active_by_voltage)
        * vcat(
            real.(sensitivity_voltage_by_power_wye_active),
            imag.(sensitivity_voltage_by_power_wye_active)
        )
    )
    sensitivity_loss_active_by_power_wye_reactive = (
        transpose(sensitivity_loss_active_by_voltage)
        * vcat(
            real.(sensitivity_voltage_by_power_wye_reactive),
            imag.(sensitivity_voltage_by_power_wye_reactive)
        )
    )

    sensitivity_loss_active_by_power_delta_active = (
        transpose(sensitivity_loss_active_by_voltage)
        * vcat(
            real.(sensitivity_voltage_by_power_delta_active),
            imag.(sensitivity_voltage_by_power_delta_active)
        )
    )
    sensitivity_loss_active_by_power_delta_reactive = (
        transpose(sensitivity_loss_active_by_voltage)
        * vcat(
            real.(sensitivity_voltage_by_power_delta_reactive),
            imag.(sensitivity_voltage_by_power_delta_reactive)
        )
    )

    sensitivity_loss_reactive_by_power_wye_active = (
        transpose(sensitivity_loss_reactive_by_voltage)
        * vcat(
            real.(sensitivity_voltage_by_power_wye_active),
            imag.(sensitivity_voltage_by_power_wye_active)
        )
    )
    sensitivity_loss_reactive_by_power_wye_reactive = (
        transpose(sensitivity_loss_reactive_by_voltage)
        * vcat(
            real.(sensitivity_voltage_by_power_wye_reactive),
            imag.(sensitivity_voltage_by_power_wye_reactive)
        )
    )

    sensitivity_loss_reactive_by_power_delta_active = (
        transpose(sensitivity_loss_reactive_by_voltage)
        * vcat(
            real.(sensitivity_voltage_by_power_delta_active),
            imag.(sensitivity_voltage_by_power_delta_active)
        )
    )
    sensitivity_loss_reactive_by_power_delta_reactive = (
        transpose(sensitivity_loss_reactive_by_voltage)
        * vcat(
            real.(sensitivity_voltage_by_power_delta_reactive),
            imag.(sensitivity_voltage_by_power_delta_reactive)
        )
    )

    # No-load losses
    sensitivity_loss_active_by_noload_voltage = (
        real_valued_conjugate(
            conj.(
                transpose(electric_grid_model.nodal_voltage_vector_no_load)
                * real.(electric_grid_model.nodal_admittance_matrix)
            )
            + transpose(conj.(
                real.(electric_grid_model.nodal_admittance_matrix)
                * electric_grid_model.nodal_voltage_vector_no_load
            ))
        )
    )
    sensitivity_loss_reactive_by_noload_voltage = (
        real_valued_conjugate(
            conj.(
                transpose(electric_grid_model.nodal_voltage_vector_no_load)
                * imag.(- electric_grid_model.nodal_admittance_matrix)
            )
            + transpose(conj.(
                imag.(- electric_grid_model.nodal_admittance_matrix)
                * electric_grid_model.nodal_voltage_vector_no_load
            ))
        )
    )

    constant_loss_active_no_load = (
        sensitivity_loss_active_by_noload_voltage
        * vcat(
            real.(electric_grid_model.nodal_voltage_vector_no_load),
            imag.(electric_grid_model.nodal_voltage_vector_no_load)
        )
    )
    constant_loss_reactive_no_load = (
        sensitivity_loss_reactive_by_noload_voltage
        * vcat(
            real.(electric_grid_model.nodal_voltage_vector_no_load),
            imag.(electric_grid_model.nodal_voltage_vector_no_load)
        )
    )

    LinearElectricGridModel(
        sensitivity_voltage_by_power_wye_active,
        sensitivity_voltage_by_power_wye_reactive,
        sensitivity_voltage_by_power_delta_active,
        sensitivity_voltage_by_power_delta_reactive,
        sensitivity_voltage_magnitude_by_power_wye_active,
        sensitivity_voltage_magnitude_by_power_wye_reactive,
        sensitivity_voltage_magnitude_by_power_delta_active,
        sensitivity_voltage_magnitude_by_power_delta_reactive,
        sensitivity_power_branch_from_by_power_wye_active,
        sensitivity_power_branch_from_by_power_wye_reactive,
        sensitivity_power_branch_from_by_power_delta_active,
        sensitivity_power_branch_from_by_power_delta_reactive,
        sensitivity_power_branch_to_by_power_wye_active,
        sensitivity_power_branch_to_by_power_wye_reactive,
        sensitivity_power_branch_to_by_power_delta_active,
        sensitivity_power_branch_to_by_power_delta_reactive,
        sensitivity_loss_active_by_power_wye_active,
        sensitivity_loss_active_by_power_wye_reactive,
        sensitivity_loss_active_by_power_delta_active,
        sensitivity_loss_active_by_power_delta_reactive,
        sensitivity_loss_reactive_by_power_wye_active,
        sensitivity_loss_reactive_by_power_wye_reactive,
        sensitivity_loss_reactive_by_power_delta_active,
        sensitivity_loss_reactive_by_power_delta_reactive,
        sensitivity_loss_active_by_noload_voltage,
        sensitivity_loss_reactive_by_noload_voltage,
        constant_loss_active_no_load,
        constant_loss_reactive_no_load
    )
end

"Instantiate linear electric grid model object for given `scenario_name`."
function LinearElectricGridModel(scenario_name::String)
    # Obtain electric grid model.
    electric_grid_model = ElectricGridModel(scenario_name)

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

    LinearElectricGridModel(
        electric_grid_model,
        nodal_voltage_vector,
        branch_power_vector_1,
        branch_power_vector_2
    )
end

"Utility function for creating the node phases string for OpenDSS."
function get_node_phases_string(element)
    node_phases_string = ""
    if element[:is_phase_0_connected] == 1
        node_phases_string *= ".0"
    end
    if element[:is_phase_1_connected] == 1
        node_phases_string *= ".1"
    end
    if element[:is_phase_2_connected] == 1
        node_phases_string *= ".2"
    end
    if element[:is_phase_3_connected] == 1
        node_phases_string *= ".3"
    end

    return node_phases_string
end

"""
Initialize OpenDSS circuit model for given `electric_grid_data`.

- Instantiates OpenDSS model.
- No object is returned because the OpenDSS model lives in memory and
  can be accessed with the API of the `OpenDSS.jl` package.
"""
function initialize_open_dss_model(
    electric_grid_data::FLEDGE.DatabaseInterface.ElectricGridData
)
    # Clear OpenDSS.
    opendss_command_string = "clear"
    Logging.@debug("", opendss_command_string)
    OpenDSSDirect.dss(opendss_command_string)

    # Obtain extra definitions string.
    if ismissing(electric_grid_data.electric_grids[:extra_definitions_string][1])
        extra_definitions_string = ""
    else
        extra_definitions_string = (
            electric_grid_data.electric_grids[:extra_definitions_string][1]
        )
    end

    # Add circuit info to OpenDSS command string.
    opendss_command_string = (
        "new circuit.$(electric_grid_data.electric_grids[:electric_grid_name][1])"
        * " phases=$(electric_grid_data.electric_grids[:n_phases][1])"
        * " bus1=$(electric_grid_data.electric_grids[:source_node_name][1])"
        * " basekv=$(electric_grid_data.electric_grids[:source_voltage][1] / 1e3)"
        * " $extra_definitions_string"
    )

    # Create circuit in OpenDSS.
    Logging.@debug("", opendss_command_string)
    OpenDSSDirect.dss(opendss_command_string)

    # Define line codes.
    for line_type in eachrow(electric_grid_data.electric_grid_line_types)
        # Obtain line resistance and reactance matrix entries for the line.
        matrices = (
            electric_grid_data.electric_grid_line_types_matrices[
                (
                    electric_grid_data.electric_grid_line_types_matrices[
                        :line_type
                    ]
                    .== line_type[:line_type]
                ),
                [:r, :x, :c]
            ]
        )

        # Add line type name and number of phases to OpenDSS command string.
        opendss_command_string = (
            "new linecode.$(line_type[:line_type])"
            * " nphases=$(line_type[:n_phases])"
        )

        # Add resistance and reactance matrix entries to OpenDSS command string,
        # with formatting depending on number of phases.
        if line_type[:n_phases] == 1
            opendss_command_string *= (
                " rmatrix = "
                * (Printf.@sprintf("[%.8f]", matrices[:r]...))
                * " xmatrix = "
                * (Printf.@sprintf("[%.8f]", matrices[:x]...))
                * " cmatrix = "
                * (Printf.@sprintf("[%.8f]", matrices[:c]...))
            )
        elseif line_type[:n_phases] == 2
            opendss_command_string *= (
                " rmatrix = "
                * (Printf.@sprintf("[%.8f | %.8f %.8f]", matrices[:r]...))
                * " xmatrix = "
                * (Printf.@sprintf("[%.8f | %.8f %.8f]", matrices[:x]...))
                * " cmatrix = "
                * (Printf.@sprintf("[%.8f | %.8f %.8f]", matrices[:c]...))
            )
        elseif line_type[:n_phases] == 3
            opendss_command_string *= (
                " rmatrix = "
                * (Printf.@sprintf(
                    "[%.8f | %.8f %.8f | %.8f %.8f %.8f]", matrices[:r]...
                ))
                * " xmatrix = "
                * (Printf.@sprintf(
                    "[%.8f | %.8f %.8f | %.8f %.8f %.8f]", matrices[:x]...
                ))
                * " cmatrix = "
                * (Printf.@sprintf(
                    "[%.8f | %.8f %.8f | %.8f %.8f %.8f]", matrices[:c]...
                ))
            )
        end

        # Create line code in OpenDSS.
        Logging.@debug("", opendss_command_string)
        OpenDSSDirect.dss(opendss_command_string)
    end

    # Define lines.
    for line in eachrow(electric_grid_data.electric_grid_lines)
        # Obtain number of phases for the line.
        n_phases = Int(
            sum([
                line[:is_phase_1_connected],
                line[:is_phase_2_connected],
                line[:is_phase_3_connected]
            ])
        )

        # Add line name, phases, node connections, line type and length
        # to OpenDSS command string.
        opendss_command_string = (
            "new line.$(line[:line_name])"
            * " phases=$(line[:n_phases])"
            * " bus1=$(line[:node_1_name])$(get_node_phases_string(line))"
            * " bus2=$(line[:node_2_name])$(get_node_phases_string(line))"
            * " linecode=$(line[:line_type])"
            * " length=$(line[:length])"
        )

        # Create line in OpenDSS.
        Logging.@debug("", opendss_command_string)
        OpenDSSDirect.dss(opendss_command_string)
    end

    # Define transformers.
    # - Note: This setup only works for transformers with
    #   identical number of phases at each winding / side.
    for transformer in eachrow(
        electric_grid_data.electric_grid_transformers[
            electric_grid_data.electric_grid_transformers[:winding] .== 1,
            1:end
        ]
    )
        # Obtain number of phases for the transformer.
        # This assumes identical number of phases at all windings.
        n_phases = Int(
            sum([
                transformer[:is_phase_1_connected],
                transformer[:is_phase_2_connected],
                transformer[:is_phase_3_connected]
            ])
        )

        # Obtain windings for the transformer.
        windings = (
            electric_grid_data.electric_grid_transformers[
                (
                    electric_grid_data.
                    electric_grid_transformers[:transformer_name]
                    .== transformer[:transformer_name]
                ),
                1:end
            ]
        )

        # Obtain reactances for the transformer.
        reactances = (
            electric_grid_data.electric_grid_transformer_reactances[
                (
                    electric_grid_data.
                    electric_grid_transformer_reactances[:transformer_name]
                    .== transformer[:transformer_name]
                ),
                1:end
            ]
        )

        # Obtain taps for the transformer.
        taps = (
            electric_grid_data.electric_grid_transformer_taps[
                (
                    electric_grid_data.
                    electric_grid_transformer_taps[:transformer_name]
                    .== transformer[:transformer_name]
                ),
                1:end
            ]
        )

        # Add transformer name, number of phases / windings and reactances
        # to OpenDSS command string.
        opendss_command_string = (
            "new transformer.$(transformer[:transformer_name])"
            * " phases=$n_phases"
            * " windings=$(length(windings[:winding]))"
            * " xscarray=$([x for x in reactances[:reactance_percentage]])"
        )
        for winding = eachrow(windings)
            # Obtain nominal voltage level for each winding.
            voltage = (
                electric_grid_data.electric_grid_nodes[
                    (
                        electric_grid_data.electric_grid_nodes[:node_name]
                        .== winding[:node_name]
                    ),
                    :voltage
                ]
            )[1]

            # Obtain node phases connection string for each winding.
            if winding[:connection] == "wye"
                if winding[:is_phase_0_connected] == 0
                    # Enforce wye-open connection according to:
                    # OpenDSS Manual April 2018, page 136, "rneut".
                    node_phases_string = (
                        get_node_phases_string(winding)
                        * ".4"
                    )
                elseif winding[:is_phase_0_connected] == 1
                    # Enforce wye-grounded connection.
                    node_phases_string = (
                        get_node_phases_string(winding)
                        * ".0"
                    )
                    # Remove leading ".0".
                    node_phases_string = node_phases_string[3:end]
                end
            elseif winding[:connection] == "delta"
                if winding[:is_phase_0_connected] == 0
                    node_phases_string = (
                        get_node_phases_string(winding)
                    )
                elseif winding[:is_phase_0_connected] == 1
                    node_phases_string = (
                        get_node_phases_string(winding)
                    )
                    # Remove leading ".0"
                    node_phases_string = node_phases_string[3:end]
                    Logging.@warn(
                        "No ground connection possible for delta-connected"
                        * " transformer $(transformer[:transformer_name])."
                    )
                end
            end

            # Add node connection, nominal voltage / power and resistance
            # to OpenDSS command string for each winding.
            opendss_command_string *= (
                " wdg=$(winding[:winding])"
                * " bus=$(winding[:node_name])" * node_phases_string
                * " conn=$(winding[:connection])"
                * " kv=$(voltage / 1000)"
                * " kva=$(winding[:power] / 1000)"
                * " %r=$(winding[:resistance_percentage])"
            )

            # Add maximum / minimum level
            # to OpenDSS command string for each winding.
            for winding_index in findall(taps[:winding] .== winding[:winding])
                opendss_command_string *= (
                    " maxtap="
                    * "$(taps[:tap_maximum_voltage_per_unit][winding_index])"
                    * " mintap="
                    * "$(taps[:tap_minimum_voltage_per_unit][winding_index])"
                )
            end
        end

        # Create transformer in OpenDSS.
        Logging.@debug("", opendss_command_string)
        OpenDSSDirect.dss(opendss_command_string)
    end

    # Define loads.
    for load in eachrow(electric_grid_data.electric_grid_loads)
        # Obtain number of phases for the load.
        n_phases = Int(
            sum([
                load[:is_phase_1_connected],
                load[:is_phase_2_connected],
                load[:is_phase_3_connected]
            ])
        )

        # Obtain nominal voltage level for the load.
        voltage = (
            electric_grid_data.electric_grid_nodes[
                (
                    electric_grid_data.electric_grid_nodes[:node_name]
                    .== load[:node_name]
                ),
                :voltage
            ]
        )[1]
        # Convert to line-to-neutral voltage for single-phase loads, according to:
        # https://sourceforge.net/p/electricdss/discussion/861976/thread/9c9e0efb/
        if n_phases == 1
            voltage /= sqrt(3)
        end

        # Add node connection, model type, voltage, nominal power
        # to OpenDSS command string.
        opendss_command_string = (
            "new load.$(load[:load_name])"
            # TODO: Check if any difference without ".0" for wye-connected loads.
            * " bus1=$(load[:node_name])$(get_node_phases_string(load))"
            * " phases=$n_phases"
            * " conn=$(load[:connection])"
            # All loads are modelled as constant P/Q according to:
            # OpenDSS Manual April 2018, page 150, "Model"
            * " model=1"
            * " kv=$(voltage / 1000)"
            * " kw=$(load[:active_power] / 1000)"
            * " kvar=$(load[:reactive_power] / 1000)"
            # Set low V_min to avoid switching to impedance model according to:
            # OpenDSS Manual April 2018, page 150, "Vminpu"
            * " vminpu=0.6"
            # Set high V_max to avoid switching to impedance model according to:
            # OpenDSS Manual April 2018, page 150, "Vmaxpu"
            * " vmaxpu=1.4"
        )

        # Create load in OpenDSS.
        Logging.@debug("", opendss_command_string)
        OpenDSSDirect.dss(opendss_command_string)
    end

    # TODO: Add switches.

    # Set control mode and voltage bases.
    opendss_command_string = (
        "set voltagebases="
        * "$(electric_grid_data.electric_grids[:voltage_bases_string][1])"
        * "\nset controlmode="
        * "$(electric_grid_data.electric_grids[:control_mode_string][1])"
        * "\nset loadmult="
        * "$(electric_grid_data.electric_grids[:load_multiplier][1])"
        * "\ncalcvoltagebases"
    )
    Logging.@debug("", opendss_command_string)
    OpenDSSDirect.dss(opendss_command_string)

    # Set solution mode to "single snapshot power flow" according to:
    # OpenDSSComDoc, November 2016, page 1
    opendss_command_string = "set mode=0"
    Logging.@debug("", opendss_command_string)
    OpenDSSDirect.dss(opendss_command_string)
end

"Initialize OpenDSS model for given `scenario_name`."
function initialize_open_dss_model(scenario_name::String)
    # Obtain electric grid data.
    electric_grid_data = (
        FLEDGE.DatabaseInterface.ElectricGridData(scenario_name)
    )

    initialize_open_dss_model(electric_grid_data)
end

end
