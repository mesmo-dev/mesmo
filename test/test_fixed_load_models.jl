# Fixed load model tests.

Test.@testset "Fixed load model tests" begin
    Test.@testset "Construct fixed load model test" begin
        # Obtain electric grid data.
        electric_grid_data = (
            FLEDGE.API.get_electric_grid_data(test_scenario_name)
        )

        # Define expected result.
        expected = FLEDGE.FixedLoadModels.FixedLoadModel

        # Get actual result.
        @time_log "Construct fixed load model test" actual = typeof(
            FLEDGE.FixedLoadModels.FixedLoadModel(
                electric_grid_data,
                "test_load"
            )
        )

        # Evaluate test.
        Test.@test actual == expected
    end
end