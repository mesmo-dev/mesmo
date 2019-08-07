# Template for tests.

Test.@testset "Template tests" begin
    Test.@testset "Test which passes" begin
        # Define expected result.
        expected = 1

        # Get actual result.
        @time_log "Test which passes" actual = 1

        # Evaluate test.
        Test.@test actual ≈ expected
    end
end
