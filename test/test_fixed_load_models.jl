# Fixed load model tests.

Test.@testset "Fixed load model tests" begin
    Test.@testset "Construct fixed load model test" begin
        # Obtain electric grid data.
        fixed_load_data = (
            FLEDGE.API.get_fixed_load_data(test_scenario_name)
        )

        # Define expected result.
        expected = FLEDGE.FixedLoadModels.FixedLoadModel

        # Get actual result.
        @time_log "Construct fixed load model test" actual = typeof(
            FLEDGE.FixedLoadModels.FixedLoadModel(
                fixed_load_data,
                fixed_load_data.fixed_loads[:load_name][1]
            )
        )

        # Evaluate test.
        Test.@test actual == expected
    end
end