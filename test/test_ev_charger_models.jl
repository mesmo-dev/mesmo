# EV charger model tests.

Test.@testset "EV charger model tests" begin
    Test.@testset "Construct EV charger model test" begin
        # Obtain electric grid data.
        ev_charger_data = (
            FLEDGE.API.get_ev_charger_data(test_scenario_name)
        )

        # Define expected result.
        expected = FLEDGE.EVChargerModels.EVChargerModel

        # Get actual result.
        @time_log "Construct EV charger model test" actual = typeof(
            FLEDGE.EVChargerModels.EVChargerModel(
                ev_charger_data,
                ev_charger_data.ev_chargers[:load_name][1]
            )
        )

        # Evaluate test.
        Test.@test actual == expected
    end
end