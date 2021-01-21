"""
Script to run the dlmps analysis
"""

import analysis.input
import analysis.simulation
import analysis.analysis

import fledge.data_interface


# Global Settings
plots = True  # will generate plots if set to True
regenerate_scenario_data = False  # will re-generate the grid input data if set to True

path_to_grid_map = 'analysis/res/electric_grid_mapping.csv'
path_to_der_data = 'analysis/res/additional_electric_grid_ders.csv'

# Define grid / scenario names
mv_grid_name = None  # if there is no MV-grid, set to None
high_granularity_scenario_name = 'kerber_dorfnetz'
# low_granularity_scenario_name = 'simple_low_granularity'
no_granularity_scenario_name = high_granularity_scenario_name  # use same scenario, only without the actual grid

granularity_levels = {
    'high_granularity': high_granularity_scenario_name,
    # 'high_granularity_mean': high_granularity_scenario_name,
    # 'low_granularity': low_granularity_scenario_name,
    'no_granularity': no_granularity_scenario_name
}

der_penetration_levels = {
    'no_penetration': 0.0,
    'low_penetration': 0.3,
    # 'high_penetration': 1.0,
}

# Generate the grids that are needed for different granularity levels (comment out if not needed)
scenario_factory = analysis.input.ScenarioFactory()
if regenerate_scenario_data:
    for scenario in granularity_levels:
        # generate aggregated / combined grids
        if 'high_granularity' in scenario:
            scenario_factory.combine_electric_grids(mv_grid_name, path_to_grid_map, granularity_levels[scenario])
        elif 'low_granularity' in scenario:
            scenario_factory.aggregate_electric_grids(mv_grid_name, path_to_grid_map, granularity_levels[scenario])

        # generate scenario data for DER penetration scenarios
        for der_penetration in der_penetration_levels:
            scenario_factory.increase_der_penetration_of_scenario(
                scenario_name=granularity_levels[scenario],
                path_to_der_data=path_to_der_data,
                penetration_ratio=der_penetration_levels[der_penetration],
                new_scenario_name=granularity_levels[scenario] + '_' + der_penetration
            )

# Recreate / overwrite database, to incorporate the new grids that we created
print('Re-loading data to incorporate the new grid data...', end='\r')
fledge.data_interface.recreate_database()
results_dict = {}
simulation_scenarios = granularity_levels.copy()
# Run different granularity / der penetration scenarios and analyze
for der_penetration in der_penetration_levels:
    for scenario in granularity_levels:
        # overwrite the scenario names based on der_penetration scenario naming
        simulation_scenarios[scenario] = granularity_levels[scenario] + '_' + der_penetration

    controller = analysis.simulation.Controller(
        simulation_scenarios=simulation_scenarios
    )
    # TODO: add custom constraints (using callback function? Or as arguments for the controller?)

    results_dict[der_penetration] = controller.run()

    results_path = fledge.utils.get_results_path('dlmp_analysis_plots', der_penetration)
    analysis_manager = analysis.analysis.AnalysisManager(results_dict, results_path)
    if plots:
        analysis_manager.generate_result_plots(der_penetration)

