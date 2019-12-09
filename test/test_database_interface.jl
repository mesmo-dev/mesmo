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

    Test.@testset "Timestep data test" begin
        # Define expected result.
        # - TODO: Replace type test with proper result check.
        expected = FLEDGE.DatabaseInterface.TimestepData

        # Get actual result.
        @time_log "Timestep data test" actual = (
            typeof(FLEDGE.DatabaseInterface.TimestepData(test_scenario_name))
        )

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Electric grid data test" begin
        # Define expected result.
        # - TODO: Replace type test with proper result check.
        expected = FLEDGE.DatabaseInterface.ElectricGridData

        # Get actual result.
        @time_log "Electric grid data test" actual = (
            typeof(FLEDGE.DatabaseInterface.ElectricGridData(test_scenario_name))
        )

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Fixed load data test" begin
        # Define expected result.
        # - TODO: Replace type test with proper result check.
        expected = FLEDGE.DatabaseInterface.FixedLoadData

        # Get actual result.
        @time_log "Fixed load data test" actual = (
            typeof(FLEDGE.DatabaseInterface.FixedLoadData(test_scenario_name))
        )

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "EV charger data test" begin
        # Define expected result.
        # - TODO: Replace type test with proper result check.
        expected = FLEDGE.DatabaseInterface.EVChargerData

        # Get actual result.
        @time_log "EV charger data test" actual = (
            typeof(FLEDGE.DatabaseInterface.EVChargerData(test_scenario_name))
        )

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Flexible load data test" begin
        # Define expected result.
        # - TODO: Replace type test with proper result check.
        expected = FLEDGE.DatabaseInterface.FlexibleLoadData

        # Get actual result.
        @time_log "Flexible load data test" actual = (
            typeof(FLEDGE.DatabaseInterface.FlexibleLoadData(test_scenario_name))
        )

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Price data test" begin
        # Define expected result.
        # - TODO: Replace type test with proper result check.
        expected = FLEDGE.DatabaseInterface.PriceData

        # Get actual result.
        @time_log "Get price data test" actual = (
            typeof(FLEDGE.DatabaseInterface.PriceData(test_scenario_name))
        )

        # Evaluate test.
        Test.@test actual == expected
    end
end
