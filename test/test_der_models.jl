# DER model tests.

Test.@testset "DER model tests" begin
    Test.@testset "Fixed load model test" begin
        # Obtain electric grid data.
        fixed_load_data = (
            FLEDGE.API.get_fixed_load_data(test_scenario_name)
        )

        # Define expected result.
        expected = FLEDGE.DERModels.FixedLoadModel

        # Get actual result.
        @time_log "Fixed load model test" actual = typeof(
            FLEDGE.DERModels.FixedLoadModel(
                fixed_load_data,
                fixed_load_data.fixed_loads[:load_name][1]
            )
        )

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "EV charger model test" begin
        # Obtain electric grid data.
        ev_charger_data = (
            FLEDGE.API.get_ev_charger_data(test_scenario_name)
        )

        # Define expected result.
        expected = FLEDGE.DERModels.EVChargerModel

        # Get actual result.
        @time_log "EV charger model test" actual = typeof(
            FLEDGE.DERModels.EVChargerModel(
                ev_charger_data,
                ev_charger_data.ev_chargers[:load_name][1]
            )
        )

        # Evaluate test.
        Test.@test actual == expected
    end

    Test.@testset "Flexible load model test" begin
        # Obtain electric grid data.
        flexible_load_data = (
            FLEDGE.API.get_flexible_load_data(test_scenario_name)
        )

        # Define expected result.
        expected = FLEDGE.DERModels.FlexibleLoadModel

        # Get actual result.
        @time_log "Flexible load model test" actual = typeof(
            FLEDGE.DERModels.FlexibleLoadModel(
                flexible_load_data,
                flexible_load_data.flexible_loads[:load_name][1]
            )
        )

        # Evaluate test.
        Test.@test actual == expected
    end
end
