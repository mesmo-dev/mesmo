# Application programming interface tests.

Test.@testset "API tests" begin
    Test.@testset "Run operation problem test" begin
        # Define expected result.
        expected = true

        # Get actual result.
        @time_log "Run operation problem test" actual = (
            FLEDGE.API.run_operation_problem(scenario_name)
        )

        # Evaluate test.
        Test.@test actual == expected
    end
end
