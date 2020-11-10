"""Script to aggregate DERs from multiple low voltage grids to nodes of one MV grid based on a mapping table
"""

import pandas as pd
import shutil
import os

import fledge.data_interface

# Specify absolute path to fledge repo
# TODO: do this with system functions
path_to_fledge = '/Users/tomschelo/PycharmProjects/fledge/'

# Load csv-file as dataframe that contains the mapping of LV grids to MV nodes
map_grids_filename = 'electric_grid_mapping.csv'
map_grids = pd.read_csv(os.path.join(path_to_fledge, 'examples', map_grids_filename))

# Specify MV grid name
mv_scenario_name = 'cigre_mv_network_with_all_ders'
aggregated_scenario_name = 'cigre_aggregated_mv_lv_grid'  # a folder will be created with this name

# Create output folder for the new grid
output_path = os.path.join(path_to_fledge, 'data')
# Instantiate dedicated directory for current grid.
shutil.rmtree(os.path.join(output_path, aggregated_scenario_name), ignore_errors=True)
os.mkdir(os.path.join(output_path, aggregated_scenario_name))

# Recreate / overwrite database, to incorporate changes in the CSV files.
fledge.data_interface.recreate_database()

# Load data of MV grid which is the underlying grid for our new
aggregated_electric_grid_data = fledge.data_interface.ElectricGridData(mv_scenario_name)
aggregated_electric_grid_data.electric_grid.loc['electric_grid_name'] = aggregated_scenario_name
aggregated_electric_grid_data.electric_grid_nodes.loc[:, 'electric_grid_name'] = aggregated_scenario_name
aggregated_electric_grid_data.scenario_data.scenario.loc['scenario_name'] = aggregated_scenario_name
aggregated_electric_grid_data.scenario_data.scenario.loc['electric_grid_name'] = aggregated_scenario_name
aggregated_electric_grid_data.electric_grid_transformers.loc[:, 'electric_grid_name'] = aggregated_scenario_name
aggregated_electric_grid_data.electric_grid_ders.loc[:, 'electric_grid_name'] = aggregated_scenario_name
aggregated_electric_grid_data.electric_grid_lines.loc[:, 'electric_grid_name'] = aggregated_scenario_name

# Remove original DERs at the respective node (LV grid replaces the MV load)
# This must be done before adding the LV grid loads, as they would be deleted again otherwise
for index, row in map_grids.iterrows():
    mv_node_name = str(row['mv_node_name'])
    aggregated_electric_grid_data.electric_grid_ders = \
        aggregated_electric_grid_data.electric_grid_ders.drop(aggregated_electric_grid_data.electric_grid_ders[
                                                                  aggregated_electric_grid_data.electric_grid_ders[
                                                                    'node_name'] == mv_node_name].index)
    print(f'CAUTION: All original DERs at MV node {mv_node_name} have been removed and will be replaced with aggregated LV loads.')

# Loop through LV grids in the map and add them to the scenario
for index, row in map_grids.iterrows():
    lv_scenario_name = str(row['lv_grid_name'])
    mv_node_name = str(row['mv_node_name'])

    print(f'Adding DERs of LV grid {lv_scenario_name} to MV node {mv_node_name}.')

    # Load data of the lv grid
    lv_electric_grid_data = fledge.data_interface.ElectricGridData(lv_scenario_name)

    # Change node_name of DERs
    lv_electric_grid_data.electric_grid_ders.loc[:, 'node_name'] = mv_node_name
    # Change electric grid name
    lv_electric_grid_data.electric_grid_ders.loc[:, 'electric_grid_name'] = aggregated_scenario_name
    # Change DER names
    lv_electric_grid_data.electric_grid_ders.loc[:, 'der_name'] = \
        mv_node_name + '_' + lv_electric_grid_data.electric_grid_ders.loc[:, 'der_name']

    # Add the DERs to the main DER table
    aggregated_electric_grid_data.electric_grid_ders = \
        aggregated_electric_grid_data.electric_grid_ders.append(
            lv_electric_grid_data.electric_grid_ders
        ).reset_index(drop=True)

# Transform series into dataframes
aggregated_electric_grid_data.electric_grid = pd.DataFrame(aggregated_electric_grid_data.electric_grid).T
aggregated_electric_grid_data.scenario_data.scenario = pd.DataFrame(aggregated_electric_grid_data.scenario_data.scenario).T

# Drop some of the scenarios columns that are not part of scenario.csv
scenarios_columns = [
    "scenario_name",
    "electric_grid_name",
    "thermal_grid_name",
    "parameter_set",
    "timestep_start",
    "timestep_end",
    "timestep_interval"
]
aggregated_electric_grid_data.scenarios = aggregated_electric_grid_data.scenario_data.scenario[scenarios_columns]

# Drop columns of the lines table that are not part of the csv-file
lines_columns = [
    'line_name',
    'electric_grid_name',
    'line_type',
    'node_1_name',
    'node_2_name',
    'is_phase_1_connected',
    'is_phase_2_connected',
    'is_phase_3_connected',
    'length'
]
aggregated_electric_grid_data.electric_grid_lines = aggregated_electric_grid_data.electric_grid_lines[lines_columns]
# electric_grid_transformers
transformers_columns = [
    'electric_grid_name',
    'transformer_name',
    'transformer_type',
    'node_1_name',
    'node_2_name',
    'is_phase_1_connected',
    'is_phase_2_connected',
    'is_phase_3_connected',
    'connection',
    'apparent_power'
    ]
aggregated_electric_grid_data.electric_grid_transformers = \
    aggregated_electric_grid_data.electric_grid_transformers[transformers_columns]

# Export csv file to output_path/combined_scenario_name
aggregated_electric_grid_data.electric_grid_nodes.to_csv(os.path.join(output_path, aggregated_scenario_name, 'electric_grid_nodes.csv'), index=False)
aggregated_electric_grid_data.electric_grid_lines.to_csv(os.path.join(output_path, aggregated_scenario_name, 'electric_grid_lines.csv'), index=False)
aggregated_electric_grid_data.electric_grid_ders.to_csv(os.path.join(output_path, aggregated_scenario_name, 'electric_grid_ders.csv'), index=False)
aggregated_electric_grid_data.electric_grid.to_csv(os.path.join(output_path, aggregated_scenario_name, 'electric_grids.csv'), index=False)
aggregated_electric_grid_data.electric_grid_transformers.to_csv(os.path.join(output_path, aggregated_scenario_name, 'electric_grid_transformers.csv'), index=False)
aggregated_electric_grid_data.scenarios.to_csv(os.path.join(output_path, aggregated_scenario_name, 'scenarios.csv'), index=False, date_format='%Y-%m-%dT%H:%M:%S')

print(f'Done. The aggregated grid can be found in: {os.path.join(output_path, aggregated_scenario_name)}')
