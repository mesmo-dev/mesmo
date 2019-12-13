# Power flow solver tests.

Test.@testset "Power flow solver tests" begin
    Test.@testset "Get voltage vector fixed point tests" begin
        for test_name in ["1", "2", "3"]
            # Load test data.
            path = (
                joinpath(
                    test_data_path,
                    "test_get_voltage_vector_" * test_name)
            )
            admittance_matrix = (
                SparseArrays.sparse(
                    parse.(
                        ComplexF64,
                        Matrix(
                            CSV.read(
                                joinpath(path, "admittance_matrix.csv"),
                                header=false
                            )
                        )[4:end, 4:end]
                    )
                )
            )
            transformation_matrix = (
                SparseArrays.sparse(
                    Matrix(
                        CSV.read(
                            joinpath(path, "transformation_matrix.csv"),
                            header=false
                        )
                    )[4:end, 4:end]
                )
            )
            power_vector_wye = (
                parse.(
                    ComplexF64,
                    Matrix(
                        CSV.read(
                            joinpath(path, "power_vector_wye.csv"),
                            header=false
                        )
                    )[4:end]
                )
            )
            power_vector_delta = (
                parse.(
                    ComplexF64,
                    Matrix(
                        CSV.read(
                            joinpath(path, "power_vector_delta.csv"),
                            header=false
                        )
                    )[4:end]
                )
            )
            voltage_vector_no_load = (
                parse.(
                    ComplexF64,
                    Matrix(
                        CSV.read(
                            joinpath(path, "voltage_vector_no_load.csv"),
                            header=false
                        )
                    )[4:end]
                )
            )
            voltage_vector_solution = (
                parse.(
                    ComplexF64,
                    Matrix(
                        CSV.read(
                            joinpath(path, "voltage_vector_solution.csv"),
                            header=false
                        )
                    )[4:end]
                )
            )

            # Define expected result.
            expected = voltage_vector_solution

            # Get actual result.
            @time_log "Get voltage vector fixed point test $test_name" actual = (
                FLEDGE.PowerFlowSolvers.get_voltage_fixed_point(
                    admittance_matrix,
                    transformation_matrix,
                    power_vector_wye,
                    power_vector_delta,
                    zeros(ComplexF64, size(power_vector_wye)),
                    zeros(ComplexF64, size(power_vector_delta)),
                    voltage_vector_no_load,
                    voltage_vector_no_load
                )
            )

            # Evaluate test.
            Test.@test all(isapprox.(actual, expected; atol=1.1))
        end
    end

    Test.@testset "Get voltage vector fixed point with model test" begin
        # Set up electric grid model.
        electric_grid_model = (
            FLEDGE.ElectricGridModels.ElectricGridModel(scenario_name)
        )

        # Define expected result.
        # - TODO: Replace type test with proper result check.
        FLEDGE.ElectricGridModels.initialize_open_dss_model(scenario_name)
        open_dss = FLEDGE.PowerFlowSolvers.get_voltage_open_dss()
        expected = Array{ComplexF64, 1}

        # Get actual result.
        @time_log "Get voltage vector fixed point with model test" actual = (
            FLEDGE.PowerFlowSolvers.get_voltage_fixed_point(
                electric_grid_model
            )
        )
        if test_plots
            Plots.bar(
                sort(
                    OpenDSSDirect.Circuit.AllNodeNames(),
                    lt=FLEDGE.Utils.natural_less_than
                ),
                (
                    abs.(actual)
                    - abs.(open_dss)
                );
                label="Error"
            )
            display(Plots.plot!(legend = :outertop))
            Plots.scatter(
                sort(
                    OpenDSSDirect.Circuit.AllNodeNames(),
                    lt=FLEDGE.Utils.natural_less_than
                ),
                (
                    abs.(open_dss)
                    ./ abs.(
                        electric_grid_model.node_voltage_vector_no_load
                    )
                );
                label="OpenDSS",
                marker=7
            )
            Plots.scatter!(
                sort(
                    OpenDSSDirect.Circuit.AllNodeNames(),
                    lt=FLEDGE.Utils.natural_less_than
                ),
                (
                    abs.(actual)
                    ./ abs.(
                        electric_grid_model.node_voltage_vector_no_load
                    )
                );
                label="Fixed point",
                marker=4
            )
            display(Plots.plot!(legend = :outertop))
        end
        actual = typeof(actual)

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Get branch power vectors with model test" begin
        # Set up electric grid model.
        electric_grid_model = (
            FLEDGE.ElectricGridModels.ElectricGridModel(scenario_name)
        )

        # Get voltage vector.
        node_voltage_vector = (
            FLEDGE.PowerFlowSolvers.get_voltage_fixed_point(
                electric_grid_model
            )
        )

        # Define expected result.
        # - TODO: Replace type test with proper result check.
        expected = Tuple{Array{Complex{Float64},1},Array{Complex{Float64},1}}

        # Get actual result.
        @time_log "Get branch power vectors with model test" (
            branch_power_vector_1,
            branch_power_vector_2
        ) = (
            FLEDGE.PowerFlowSolvers.get_branch_power_fixed_point(
                electric_grid_model,
                node_voltage_vector
            )
        )
        actual = typeof((branch_power_vector_1, branch_power_vector_2))

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Get total loss with model test" begin
        # Set up electric grid model.
        electric_grid_model = (
            FLEDGE.ElectricGridModels.ElectricGridModel(scenario_name)
        )

        # Get voltage vector.
        node_voltage_vector = (
            FLEDGE.PowerFlowSolvers.get_voltage_fixed_point(
                electric_grid_model
            )
        )

        # Define expected result.
        # - TODO: Replace type test with proper result check.
        expected = Complex{Float64}

        # Get actual result.
        @time_log "Get total loss with model test" total_loss = (
            FLEDGE.PowerFlowSolvers.get_loss_fixed_point(
                electric_grid_model,
                node_voltage_vector
            )
        )
        actual = typeof(total_loss)

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Get voltage vector by OpenDSS test" begin
        # Set up OpenDSS model.
        FLEDGE.ElectricGridModels.initialize_open_dss_model(scenario_name)

        # Define expected result.
        # - TODO: Replace type test with proper result check.
        expected = Array{ComplexF64, 1}

        # Get actual result.
        @time_log "Get voltage vector by OpenDSS test" actual = (
            typeof(FLEDGE.PowerFlowSolvers.get_voltage_open_dss())
        )

        # Evaluate test.
        Test.@test actual == expected
    end
end
