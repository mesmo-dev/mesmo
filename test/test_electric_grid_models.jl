# Electric grid model tests.

Test.@testset "Electric grid model tests" begin
    Test.@testset "Linear electric grid model test" begin
        # Obtain electric grid model.
        electric_grid_model = (
            FLEDGE.API.get_electric_grid_model(test_scenario_name)
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

        # Define expected result.
        expected = FLEDGE.ElectricGridModels.LinearElectricGridModel

        # Get actual result.
        @time_log "Linear electric grid model test" linear_electric_grid_model = (
            FLEDGE.ElectricGridModels.LinearElectricGridModel(
                electric_grid_model,
                nodal_voltage_vector,
                branch_power_vector_1,
                branch_power_vector_2
            )
        )
        actual = typeof(linear_electric_grid_model)

        # Define power vector multipliers for testing of linear model at
        # different loading conditions.
        power_multipliers = 0:0.2:1.2

        # Obtain nodal power vectors assuming nominal loading conditions.
        nodal_power_vector_wye = (
            electric_grid_model.load_incidence_wye_matrix
            * electric_grid_model.load_power_vector_nominal
        )
        nodal_power_vector_delta = (
            electric_grid_model.load_incidence_delta_matrix
            * electric_grid_model.load_power_vector_nominal
        )
        nodal_power_vector_wye_active = (
            real.(nodal_power_vector_wye)
        )
        nodal_power_vector_wye_reactive = (
            imag.(nodal_power_vector_wye)
        )
        nodal_power_vector_delta_active = (
            real.(nodal_power_vector_delta)
        )
        nodal_power_vector_delta_reactive = (
            imag.(nodal_power_vector_delta)
        )

        # Obtain initial and no-load nodal voltage vectors.
        nodal_voltage_vector_initial = (
            FLEDGE.PowerFlowSolvers.get_voltage_fixed_point(
                electric_grid_model,
                nodal_power_vector_wye,
                nodal_power_vector_delta
            )
        )
        nodal_voltage_vector_no_load = (
            electric_grid_model.nodal_voltage_vector_no_load
        )

        # Pre-allocate testing arrays.
        nodal_voltage_vector_magnitude_fixed_point = (
            zeros(
                Float64,
                electric_grid_model.index.node_dimension,
                length(power_multipliers)
            )
        )
        nodal_voltage_vector_magnitude_linear_model = (
            zeros(
                Float64,
                electric_grid_model.index.node_dimension,
                length(power_multipliers)
            )
        )
        nodal_voltage_vector_fixed_point = (
            zeros(
                ComplexF64,
                electric_grid_model.index.node_dimension,
                length(power_multipliers)
            )
        )
        nodal_voltage_vector_linear_model = (
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
        total_loss_active_fixed_point = (
            zeros(Float64, length(power_multipliers))
        )
        total_loss_active_linear_model = (
            zeros(Float64, length(power_multipliers))
        )
        total_loss_reactive_fixed_point = (
            zeros(Float64, length(power_multipliers))
        )
        total_loss_reactive_linear_model = (
            zeros(Float64, length(power_multipliers))
        )
        nodal_voltage_vector_error = (
            zeros(Float64, length(power_multipliers))
        )
        nodal_voltage_vector_magnitude_error = (
            zeros(Float64, length(power_multipliers))
        )
        branch_power_vector_1_squared_error = (
            zeros(Float64, length(power_multipliers))
        )
        branch_power_vector_2_squared_error = (
            zeros(Float64, length(power_multipliers))
        )
        total_loss_active_error = (
            zeros(Float64, length(power_multipliers))
        )
        total_loss_reactive_error = (
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
            nodal_power_vector_wye_candidate = (
                power_multiplier
                * nodal_power_vector_wye
            )
            nodal_power_vector_delta_candidate = (
                power_multiplier
                * nodal_power_vector_delta
            )
            nodal_power_vector_wye_candidate_active = (
                real.(nodal_power_vector_wye_candidate)
            )
            nodal_power_vector_wye_candidate_reactive = (
                imag.(nodal_power_vector_wye_candidate)
            )
            nodal_power_vector_delta_candidate_active = (
                real.(nodal_power_vector_delta_candidate)
            )
            nodal_power_vector_delta_candidate_reactive = (
                imag.(nodal_power_vector_delta_candidate)
            )

            # Obtain nodal voltage vector with fixed-point solution.
            nodal_voltage_vector_fixed_point[:, multiplier_index] = (
                FLEDGE.PowerFlowSolvers.get_voltage_fixed_point(
                    electric_grid_model,
                    nodal_power_vector_wye_candidate,
                    nodal_power_vector_delta_candidate
                )
            )
            nodal_voltage_vector_magnitude_fixed_point[:, multiplier_index] = (
                abs.(nodal_voltage_vector_fixed_point[:, multiplier_index])
            )

            # Obtain branch power vectors based on fixed-point solution.
            (
                branch_power_vector_1_fixed_point,
                branch_power_vector_2_fixed_point
            ) = (
                FLEDGE.PowerFlowSolvers.get_branch_power_fixed_point(
                    electric_grid_model,
                    nodal_voltage_vector_fixed_point[:, multiplier_index]
                )
            )
            branch_power_vector_1_squared_fixed_point[:, multiplier_index] = (
                abs.(branch_power_vector_1_fixed_point) .^ 2
            )
            branch_power_vector_2_squared_fixed_point[:, multiplier_index] = (
                abs.(branch_power_vector_2_fixed_point) .^ 2
            )

            # Obtain total losses based on fixed-point solution.
            total_loss_fixed_point = (
                FLEDGE.PowerFlowSolvers.get_loss_fixed_point(
                    electric_grid_model,
                    nodal_voltage_vector_fixed_point[:, multiplier_index]
                )
            )
            total_loss_active_fixed_point[multiplier_index] = (
                real(total_loss_fixed_point)
            )
            total_loss_reactive_fixed_point[multiplier_index] = (
                imag(total_loss_fixed_point)
            )

            # Calculate nodal power difference / change.
            nodal_power_vector_wye_active_difference = (
                nodal_power_vector_wye_candidate_active
                - nodal_power_vector_wye_active
            )
            nodal_power_vector_wye_reactive_difference = (
                nodal_power_vector_wye_candidate_reactive
                - nodal_power_vector_wye_reactive
            )
            nodal_power_vector_delta_active_difference = (
                nodal_power_vector_delta_candidate_active
                - nodal_power_vector_delta_active
            )
            nodal_power_vector_delta_reactive_difference = (
                nodal_power_vector_delta_candidate_reactive
                - nodal_power_vector_delta_reactive
            )

            # Calculate approximate voltage, power vectors and total losses.
            nodal_voltage_vector_linear_model[:, multiplier_index] = (
                nodal_voltage_vector_initial
                + linear_electric_grid_model.
                sensitivity_voltage_by_power_wye_active
                * nodal_power_vector_wye_active_difference
                + linear_electric_grid_model.
                sensitivity_voltage_by_power_wye_reactive
                * nodal_power_vector_wye_reactive_difference
                + linear_electric_grid_model.
                sensitivity_voltage_by_power_delta_active
                * nodal_power_vector_delta_active_difference
                + linear_electric_grid_model.
                sensitivity_voltage_by_power_delta_reactive
                * nodal_power_vector_delta_reactive_difference
            )
            # nodal_voltage_vector_linear_model[:, multiplier_index] = (
            #     nodal_voltage_vector_no_load
            #     + linear_electric_grid_model.
            #     sensitivity_voltage_by_power_wye_active
            #     * nodal_power_vector_wye_candidate_active
            #     + linear_electric_grid_model.
            #     sensitivity_voltage_by_power_wye_reactive
            #     * nodal_power_vector_wye_candidate_reactive
            #     + linear_electric_grid_model.
            #     sensitivity_voltage_by_power_delta_active
            #     * nodal_power_vector_delta_candidate_active
            #     + linear_electric_grid_model.
            #     sensitivity_voltage_by_power_delta_reactive
            #     * nodal_power_vector_delta_candidate_reactive
            # )
            nodal_voltage_vector_magnitude_linear_model[:, multiplier_index] = (
                abs.(nodal_voltage_vector_initial)
                + linear_electric_grid_model.
                sensitivity_voltage_magnitude_by_power_wye_active
                * nodal_power_vector_wye_active_difference
                + linear_electric_grid_model.
                sensitivity_voltage_magnitude_by_power_wye_reactive
                * nodal_power_vector_wye_reactive_difference
                + linear_electric_grid_model.
                sensitivity_voltage_magnitude_by_power_delta_active
                * nodal_power_vector_delta_active_difference
                + linear_electric_grid_model.
                sensitivity_voltage_magnitude_by_power_delta_reactive
                * nodal_power_vector_delta_reactive_difference
            )
            # nodal_voltage_vector_magnitude_linear_model[:, multiplier_index] = (
            #     abs.(nodal_voltage_vector_no_load)
            #     + linear_electric_grid_model.
            #     sensitivity_voltage_magnitude_by_power_wye_active
            #     * nodal_power_vector_wye_candidate_active
            #     + linear_electric_grid_model.
            #     sensitivity_voltage_magnitude_by_power_wye_reactive
            #     * nodal_power_vector_wye_candidate_reactive
            #     + linear_electric_grid_model.
            #     sensitivity_voltage_magnitude_by_power_delta_active
            #     * nodal_power_vector_delta_candidate_active
            #     + linear_electric_grid_model.
            #     sensitivity_voltage_magnitude_by_power_delta_reactive
            #     * nodal_power_vector_delta_candidate_reactive
            # )
            branch_power_vector_1_squared_linear_model[:, multiplier_index] = (
                linear_electric_grid_model.
                sensitivity_power_branch_from_by_power_wye_active
                * nodal_power_vector_wye_candidate_active
                + linear_electric_grid_model.
                sensitivity_power_branch_from_by_power_wye_reactive
                * nodal_power_vector_wye_candidate_reactive
                + linear_electric_grid_model.
                sensitivity_power_branch_from_by_power_delta_active
                * nodal_power_vector_delta_candidate_active
                + linear_electric_grid_model.
                sensitivity_power_branch_from_by_power_delta_reactive
                * nodal_power_vector_delta_candidate_reactive
            )
            branch_power_vector_2_squared_linear_model[:, multiplier_index] = (
                linear_electric_grid_model.
                sensitivity_power_branch_to_by_power_wye_active
                * nodal_power_vector_wye_candidate_active
                + linear_electric_grid_model.
                sensitivity_power_branch_to_by_power_wye_reactive
                * nodal_power_vector_wye_candidate_reactive
                + linear_electric_grid_model.
                sensitivity_power_branch_to_by_power_delta_active
                * nodal_power_vector_delta_candidate_active
                + linear_electric_grid_model.
                sensitivity_power_branch_to_by_power_delta_reactive
                * nodal_power_vector_delta_candidate_reactive
            )
            total_loss_active_linear_model[multiplier_index] = (
                linear_electric_grid_model.
                sensitivity_loss_active_by_power_wye_active
                * nodal_power_vector_wye_candidate_active
                + linear_electric_grid_model.
                sensitivity_loss_active_by_power_wye_reactive
                * nodal_power_vector_wye_candidate_reactive
                + linear_electric_grid_model.
                sensitivity_loss_active_by_power_delta_active
                * nodal_power_vector_delta_candidate_active
                + linear_electric_grid_model.
                sensitivity_loss_active_by_power_delta_reactive
                * nodal_power_vector_delta_candidate_reactive
            )
            total_loss_reactive_linear_model[multiplier_index] = (
                linear_electric_grid_model.
                sensitivity_loss_reactive_by_power_wye_active
                * nodal_power_vector_wye_candidate_active
                + linear_electric_grid_model.
                sensitivity_loss_reactive_by_power_wye_reactive
                * nodal_power_vector_wye_candidate_reactive
                + linear_electric_grid_model.
                sensitivity_loss_reactive_by_power_delta_active
                * nodal_power_vector_delta_candidate_active
                + linear_electric_grid_model.
                sensitivity_loss_reactive_by_power_delta_reactive
                * nodal_power_vector_delta_candidate_reactive
            )

            # Calculate errors for voltage, power vectors and total losses.
            nodal_voltage_vector_error[multiplier_index] = (
                get_error(
                    nodal_voltage_vector_fixed_point[:, multiplier_index],
                    nodal_voltage_vector_linear_model[:, multiplier_index]
                )
            )
            nodal_voltage_vector_magnitude_error[multiplier_index] = (
                get_error(
                    nodal_voltage_vector_magnitude_fixed_point[:, multiplier_index],
                    nodal_voltage_vector_magnitude_linear_model[:, multiplier_index]
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
            total_loss_active_error[multiplier_index] = (
                get_error(
                    total_loss_active_fixed_point[multiplier_index],
                    total_loss_active_linear_model[multiplier_index]
                )
            )
            total_loss_reactive_error[multiplier_index] = (
                get_error(
                    total_loss_reactive_fixed_point[multiplier_index],
                    total_loss_reactive_linear_model[multiplier_index]
                )
            )
        end

        # TODO: Validate total_loss_active_error against MATLAB implementation.
        # TODO: Validate total_loss_reactive_error against MATLAB implementation.
        @Logging.info(
            "Linear electric grid model error:",
            linear_electric_grid_model_error = DataFrames.DataFrame(
                [
                    power_multipliers,
                    nodal_voltage_vector_error,
                    nodal_voltage_vector_magnitude_error,
                    branch_power_vector_1_squared_error,
                    branch_power_vector_2_squared_error,
                    total_loss_active_error,
                    total_loss_reactive_error
                ],
                [
                    :power_multipliers,
                    :nodal_voltage_vector_error,
                    :nodal_voltage_vector_magnitude_error,
                    :branch_power_vector_1_squared_error,
                    :branch_power_vector_2_squared_error,
                    :total_loss_active_error,
                    :total_loss_reactive_error
                ]
            )
        )

        # Evaluate test.
        Test.@test actual == expected
    end
end
