# Electric grid model tests.

Test.@testset "Electric grid model tests" begin
    Test.@testset "Electric grid model test" begin
        # Define expected result.
        # - TODO: Replace type test with proper result check.
        expected = FLEDGE.ElectricGridModels.ElectricGridModel

        # Get actual result.
        @time_log "Electric grid model test" actual = (
            typeof(FLEDGE.ElectricGridModels.ElectricGridModel(scenario_name))
        )

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Simple linear electric grid model test" begin
        # Define expected result.
        expected = FLEDGE.ElectricGridModels.LinearElectricGridModel

        # Get actual result.
        @time_log "Simple linear electric grid model test" actual = (
            typeof(
                FLEDGE.ElectricGridModels.LinearElectricGridModel(scenario_name)
            )
        )

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Detailed linear electric grid model test" begin
        # Obtain electric grid model.
        electric_grid_model = (
            FLEDGE.ElectricGridModels.ElectricGridModel(scenario_name)
        )

        # Obtain power flow solution for nominal loading conditions.
        node_voltage_vector = (
            FLEDGE.PowerFlowSolvers.get_voltage_fixed_point(electric_grid_model)
        )
        (
            branch_power_vector_1,
            branch_power_vector_2
        ) = (
            FLEDGE.PowerFlowSolvers.get_branch_power_fixed_point(
                electric_grid_model,
                node_voltage_vector
            )
        )

        # Define expected result.
        expected = FLEDGE.ElectricGridModels.LinearElectricGridModel

        # Get actual result.
        @time_log "Detailed linear electric grid model test" linear_electric_grid_model = (
            FLEDGE.ElectricGridModels.LinearElectricGridModel(
                electric_grid_model,
                node_voltage_vector,
                branch_power_vector_1,
                branch_power_vector_2
            )
        )
        actual = typeof(linear_electric_grid_model)

        # Define power vector multipliers for testing of linear model at
        # different loading conditions.
        power_multipliers = 0:0.25:1.25

        # Obtain nodal power vectors assuming nominal loading conditions.
        node_power_vector_wye = (
            electric_grid_model.load_incidence_wye_matrix
            * electric_grid_model.load_power_vector_nominal
        )
        node_power_vector_delta = (
            electric_grid_model.load_incidence_delta_matrix
            * electric_grid_model.load_power_vector_nominal
        )
        node_power_vector_wye_active = (
            real.(node_power_vector_wye)
        )
        node_power_vector_wye_reactive = (
            imag.(node_power_vector_wye)
        )
        node_power_vector_delta_active = (
            real.(node_power_vector_delta)
        )
        node_power_vector_delta_reactive = (
            imag.(node_power_vector_delta)
        )

        # Obtain initial and no-load nodal voltage vectors.
        node_voltage_vector_initial = (
            FLEDGE.PowerFlowSolvers.get_voltage_fixed_point(
                electric_grid_model,
                node_power_vector_wye,
                node_power_vector_delta
            )
        )
        (
            branch_power_vector_1_initial,
            branch_power_vector_2_initial
        ) = (
            FLEDGE.PowerFlowSolvers.get_branch_power_fixed_point(
                electric_grid_model,
                node_voltage_vector_initial
            )
        )
        loss_initial = (
            FLEDGE.PowerFlowSolvers.get_loss_fixed_point(
                electric_grid_model,
                node_voltage_vector_initial
            )
        )

        node_voltage_vector_no_load = (
            electric_grid_model.node_voltage_vector_no_load
        )

        # Pre-allocate testing arrays.
        node_voltage_vector_magnitude_fixed_point = (
            zeros(
                Float64,
                electric_grid_model.index.node_dimension,
                length(power_multipliers)
            )
        )
        node_voltage_vector_magnitude_linear_model = (
            zeros(
                Float64,
                electric_grid_model.index.node_dimension,
                length(power_multipliers)
            )
        )
        node_voltage_vector_fixed_point = (
            zeros(
                ComplexF64,
                electric_grid_model.index.node_dimension,
                length(power_multipliers)
            )
        )
        node_voltage_vector_linear_model = (
            zeros(
                ComplexF64,
                electric_grid_model.index.node_dimension,
                length(power_multipliers)
            )
        )
        branch_power_vector_1_squared_fixed_point = (
            zeros(
                Float64,
                electric_grid_model.index.branch_dimension,
                length(power_multipliers)
            )
        )
        branch_power_vector_1_squared_linear_model = (
            zeros(
                Float64,
                electric_grid_model.index.branch_dimension,
                length(power_multipliers)
            )
        )
        branch_power_vector_2_squared_fixed_point = (
            zeros(
                Float64,
                electric_grid_model.index.branch_dimension,
                length(power_multipliers)
            )
        )
        branch_power_vector_2_squared_linear_model = (
            zeros(
                Float64,
                electric_grid_model.index.branch_dimension,
                length(power_multipliers)
            )
        )
        loss_active_fixed_point = (
            zeros(Float64, length(power_multipliers))
        )
        loss_active_linear_model = (
            zeros(Float64, length(power_multipliers))
        )
        loss_reactive_fixed_point = (
            zeros(Float64, length(power_multipliers))
        )
        loss_reactive_linear_model = (
            zeros(Float64, length(power_multipliers))
        )
        node_voltage_vector_error = (
            zeros(Float64, length(power_multipliers))
        )
        node_voltage_vector_magnitude_error = (
            zeros(Float64, length(power_multipliers))
        )
        branch_power_vector_1_squared_error = (
            zeros(Float64, length(power_multipliers))
        )
        branch_power_vector_2_squared_error = (
            zeros(Float64, length(power_multipliers))
        )
        loss_active_error = (
            zeros(Float64, length(power_multipliers))
        )
        loss_reactive_error = (
            zeros(Float64, length(power_multipliers))
        )

        # Define error calculation utility function.
        get_error(actual, approximate) = (
            100
            * maximum(abs.(
                (approximate - actual)
                ./ actual
            ))
        )

        # Evaluate linear model errors for each power multiplier.
        for (multiplier_index, power_multiplier) in enumerate(power_multipliers)
            # Obtain nodal power vectors depending on multiplier.
            node_power_vector_wye_candidate = (
                power_multiplier
                * node_power_vector_wye
            )
            node_power_vector_delta_candidate = (
                power_multiplier
                * node_power_vector_delta
            )
            node_power_vector_wye_candidate_active = (
                real.(node_power_vector_wye_candidate)
            )
            node_power_vector_wye_candidate_reactive = (
                imag.(node_power_vector_wye_candidate)
            )
            node_power_vector_delta_candidate_active = (
                real.(node_power_vector_delta_candidate)
            )
            node_power_vector_delta_candidate_reactive = (
                imag.(node_power_vector_delta_candidate)
            )

            # Obtain nodal voltage vector with fixed-point solution.
            node_voltage_vector_fixed_point[:, multiplier_index] = (
                FLEDGE.PowerFlowSolvers.get_voltage_fixed_point(
                    electric_grid_model,
                    node_power_vector_wye_candidate,
                    node_power_vector_delta_candidate
                )
            )
            node_voltage_vector_magnitude_fixed_point[:, multiplier_index] = (
                abs.(node_voltage_vector_fixed_point[:, multiplier_index])
            )

            # Obtain branch power vectors based on fixed-point solution.
            (
                branch_power_vector_1_fixed_point,
                branch_power_vector_2_fixed_point
            ) = (
                FLEDGE.PowerFlowSolvers.get_branch_power_fixed_point(
                    electric_grid_model,
                    node_voltage_vector_fixed_point[:, multiplier_index]
                )
            )
            branch_power_vector_1_squared_fixed_point[:, multiplier_index] = (
                abs.(branch_power_vector_1_fixed_point) .^ 2
            )
            branch_power_vector_2_squared_fixed_point[:, multiplier_index] = (
                abs.(branch_power_vector_2_fixed_point) .^ 2
            )

            # Obtain total losses based on fixed-point solution.
            loss_fixed_point = (
                FLEDGE.PowerFlowSolvers.get_loss_fixed_point(
                    electric_grid_model,
                    node_voltage_vector_fixed_point[:, multiplier_index]
                )
            )
            loss_active_fixed_point[multiplier_index] = (
                real(loss_fixed_point)
            )
            loss_reactive_fixed_point[multiplier_index] = (
                imag(loss_fixed_point)
            )

            # Calculate nodal power difference / change.
            node_power_vector_wye_active_change = (
                node_power_vector_wye_candidate_active
                - node_power_vector_wye_active
            )
            node_power_vector_wye_reactive_change = (
                node_power_vector_wye_candidate_reactive
                - node_power_vector_wye_reactive
            )
            node_power_vector_delta_active_change = (
                node_power_vector_delta_candidate_active
                - node_power_vector_delta_active
            )
            node_power_vector_delta_reactive_change = (
                node_power_vector_delta_candidate_reactive
                - node_power_vector_delta_reactive
            )

            # Calculate approximate voltage, power vectors and total losses.
            node_voltage_vector_linear_model[:, multiplier_index] = (
                node_voltage_vector_initial
                + linear_electric_grid_model.sensitivity_voltage_by_power_wye_active
                * node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_voltage_by_power_wye_reactive
                * node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_voltage_by_power_delta_active
                * node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_voltage_by_power_delta_reactive
                * node_power_vector_delta_reactive_change
            )
            node_voltage_vector_magnitude_linear_model[:, multiplier_index] = (
                abs.(node_voltage_vector_initial)
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_wye_active
                * node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_wye_reactive
                * node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_delta_active
                * node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_delta_reactive
                * node_power_vector_delta_reactive_change
            )
            branch_power_vector_1_squared_linear_model[:, multiplier_index] = (
                (abs.(branch_power_vector_1_initial) .^ 2)
                + linear_electric_grid_model.sensitivity_branch_power_1_by_power_wye_active
                * node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_branch_power_1_by_power_wye_reactive
                * node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_branch_power_1_by_power_delta_active
                * node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_branch_power_1_by_power_delta_reactive
                * node_power_vector_delta_reactive_change
            )
            branch_power_vector_2_squared_linear_model[:, multiplier_index] = (
                (abs.(branch_power_vector_2_initial) .^ 2)
                + linear_electric_grid_model.sensitivity_branch_power_2_by_power_wye_active
                * node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_branch_power_2_by_power_wye_reactive
                * node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_branch_power_2_by_power_delta_active
                * node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_branch_power_2_by_power_delta_reactive
                * node_power_vector_delta_reactive_change
            )
            # TODO: Check usage of constant loss term from linear model.
            loss_active_linear_model[multiplier_index, :] = (
                [real(loss_initial)]
                + linear_electric_grid_model.sensitivity_loss_active_by_power_wye_active
                * node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_loss_active_by_power_wye_reactive
                * node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_loss_active_by_power_delta_active
                * node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_loss_active_by_power_delta_reactive
                * node_power_vector_delta_reactive_change
            )
            loss_reactive_linear_model[multiplier_index, :] = (
                [imag(loss_initial)]
                + linear_electric_grid_model.sensitivity_loss_reactive_by_power_wye_active
                * node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_loss_reactive_by_power_wye_reactive
                * node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_loss_reactive_by_power_delta_active
                * node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_loss_reactive_by_power_delta_reactive
                * node_power_vector_delta_reactive_change
            )

            # Calculate errors for voltage, power vectors and total losses.
            node_voltage_vector_error[multiplier_index] = (
                get_error(
                    node_voltage_vector_fixed_point[:, multiplier_index],
                    node_voltage_vector_linear_model[:, multiplier_index]
                )
            )
            node_voltage_vector_magnitude_error[multiplier_index] = (
                get_error(
                    node_voltage_vector_magnitude_fixed_point[:, multiplier_index],
                    node_voltage_vector_magnitude_linear_model[:, multiplier_index]
                )
            )
            branch_power_vector_1_squared_error[multiplier_index] = (
                get_error(
                    branch_power_vector_1_squared_fixed_point[:, multiplier_index],
                    branch_power_vector_1_squared_linear_model[:, multiplier_index]
                )
            )
            branch_power_vector_2_squared_error[multiplier_index] = (
                get_error(
                    branch_power_vector_2_squared_fixed_point[:, multiplier_index],
                    branch_power_vector_2_squared_linear_model[:, multiplier_index]
                )
            )
            loss_active_error[multiplier_index] = (
                get_error(
                    loss_active_fixed_point[multiplier_index],
                    loss_active_linear_model[multiplier_index]
                )
            )
            loss_reactive_error[multiplier_index] = (
                get_error(
                    loss_reactive_fixed_point[multiplier_index],
                    loss_reactive_linear_model[multiplier_index]
                )
            )
        end

        linear_electric_grid_model_error = DataFrames.DataFrame(
            [
                power_multipliers,
                node_voltage_vector_error,
                node_voltage_vector_magnitude_error,
                branch_power_vector_1_squared_error,
                branch_power_vector_2_squared_error,
                loss_active_error,
                loss_reactive_error
            ],
            [
                :power_multipliers,
                :node_voltage_vector_error,
                :node_voltage_vector_magnitude_error,
                :branch_power_vector_1_squared_error,
                :branch_power_vector_2_squared_error,
                :loss_active_error,
                :loss_reactive_error
            ]
        )
        Logging.@info("", linear_electric_grid_model_error)
        display(linear_electric_grid_model_error)

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Initialize OpenDSS model test" begin
        # Define expected result.
        electric_grid_data = (
            FLEDGE.DatabaseInterface.ElectricGridData(scenario_name)
        )
        expected = electric_grid_data.electric_grids[1, :electric_grid_name]

        # Get actual result.
        @time_log "Initialize OpenDSS model test" (
            FLEDGE.ElectricGridModels.initialize_open_dss_model(scenario_name)
        )
        actual = OpenDSSDirect.Circuit.Name()

        # Evaluate test.
        Test.@test actual == expected
    end
end
