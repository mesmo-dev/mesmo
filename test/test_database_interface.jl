# Database interface tests.

Test.@testset "Database interface tests" begin
    Test.@testset "Connect database test" begin
        # Define expected result.
        expected = SQLite.DB

        # Get actual result.
        @time_log "Connect database test" actual = (
            typeof(FLEDGE.DatabaseInterface.connect_database())
        )

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Create database test" begin
        # Define expected result.
        expected = SQLite.DB

        # Get actual result.
        @time_log "Create database test" actual = (
            typeof(FLEDGE.DatabaseInterface.connect_database(
                overwrite_database=true
            ))
        )

        # Evaluate test.
        Test.@test actual == expected
    end
end
