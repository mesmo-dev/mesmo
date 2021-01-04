"""
Script to run the dlmps analysis
"""

import analysis.input
import analysis.simulation
import analysis.analysis

import fledge.data_interface


# Global Settings
plots = False  # will generate plots if set to True
regenerate_scenario_data = False  # will re-generate the grid input data if set to True

path_to_grid_map = 'analysis/res/electric_grid_mapping.csv'
path_to_der_data = 'analysis/res/additional_electric_grid_ders.csv'

# Define grid / scenario names (the exported scenario data will have these names)
mv_grid_name = 'simple_mv_3node'
high_granularity_scenario_name = 'simple_high_granularity'
low_granularity_scenario_name = 'simple_low_granularity'
no_granularity_scenario_name = low_granularity_scenario_name  # use same scenario, only without the actual grid
# no_granularity does not generate new scenario data

der_penetration_scenario_data = {
    'no_penetration': 0.0,
    # 'low_penetration': 0.5,
    # 'high_penetration': 1.0,
}

# Generate the grids that are needed for different granularity levels (comment out if not needed)
if regenerate_scenario_data:
    # generate aggregated / combined grids
    analysis.input.combine_electric_grids(mv_grid_name, path_to_grid_map, high_granularity_scenario_name)
    analysis.input.aggregate_electric_grids(mv_grid_name, path_to_grid_map, low_granularity_scenario_name)
    # generate scenario data for DER penetration scenarios
    grid_scenarios = [low_granularity_scenario_name, high_granularity_scenario_name]
    for scenario_name in grid_scenarios:
        for der_penetration in der_penetration_scenario_data.keys():
            analysis.input.increase_der_penetration_of_scenario_on_lv_level(
                scenario_name=scenario_name,
                path_to_der_data=path_to_der_data,
                penetration_ratio=der_penetration_scenario_data[der_penetration],
                new_scenario_name=scenario_name + '_' + der_penetration
            )

# Recreate / overwrite database, to incorporate the new grids that we created
fledge.data_interface.recreate_database()
results_dict = {}
# Run different granularity / der penetration scenarios and analyze
for der_penetration in der_penetration_scenario_data:
    granularity_scenario_data = {
        'high_granularity': high_granularity_scenario_name + '_' + der_penetration,
        # 'high_granularity_mean': high_granularity_scenario_name + '_' + der_penetration,
        'low_granularity': low_granularity_scenario_name + '_' + der_penetration,
        'no_granularity': no_granularity_scenario_name + '_' + der_penetration
    }
    controller = analysis.simulation.Controller(
        granularity_scenario_data=granularity_scenario_data
    )
    # TODO: add custom constraints (using callback function? Or as arguments for the controller?)
    # add constraint on electricity use of flex building
    # for der_name in der_model_set.der_models.keys():
    #     der_model = der_model_set.der_models[der_name]
    #     if type(der_model) is fledge.der_models.FlexibleBuildingModel:
    #         der_model.output_maximum_timeseries['grid_electric_power'] \
    #             = (-1) * der_model.active_power_nominal_timeseries

    results_dict[der_penetration] = controller.run()

    results_path = fledge.utils.get_results_path('dlmp_analysis_plots', der_penetration)
    analysis_manager = analysis.analysis.AnalysisManager(results_path)
    analysis_manager.generate_result_plots()

