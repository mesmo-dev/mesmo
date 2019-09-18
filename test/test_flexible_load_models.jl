# Flexible load model tests.

Test.@testset "Flexible load model tests" begin
    Test.@testset "Construct flexible load model test" begin
        # Obtain electric grid data.
        flexible_load_data = (
            FLEDGE.API.get_flexible_load_data(test_scenario_name)
        )

        # Define expected result.
        expected = FLEDGE.FlexibleLoadModels.FlexibleLoadModel

        # Get actual result.
        @time_log "Construct flexible load model test" actual = typeof(
            FLEDGE.FlexibleLoadModels.FlexibleLoadModel(
                flexible_load_data,
                flexible_load_data.flexible_loads[:load_name][1]
            )
        )

        # Evaluate test.
        Test.@test actual == expected
    end
end