"""Script to combine multiple grids to one large grids based on a mapping table
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
combined_scenario_name = 'cigre_combined_mv_lv_grid'  # a folder will be created with this name

# Create output folder for the new grid
output_path = os.path.join(path_to_fledge, 'data')
# Instantiate dedicated directory for current grid.
shutil.rmtree(os.path.join(output_path, combined_scenario_name), ignore_errors=True)
os.mkdir(os.path.join(output_path, combined_scenario_name))

# Recreate / overwrite database, to incorporate changes in the CSV files.
fledge.data_interface.recreate_database()

# Load data of MV grid
mv_electric_grid_data = fledge.data_interface.ElectricGridData(mv_scenario_name)

# Change electric grid name to new name
mv_electric_grid_data.electric_grid.loc['electric_grid_name'] = combined_scenario_name
mv_electric_grid_data.electric_grid_nodes.loc[:, 'electric_grid_name'] = combined_scenario_name
mv_electric_grid_data.scenario_data.scenario.loc['scenario_name'] = combined_scenario_name
mv_electric_grid_data.scenario_data.scenario.loc['electric_grid_name'] = combined_scenario_name
mv_electric_grid_data.electric_grid_transformers.loc[:, 'electric_grid_name'] = combined_scenario_name
mv_electric_grid_data.electric_grid_ders.loc[:, 'electric_grid_name'] = combined_scenario_name
mv_electric_grid_data.electric_grid_lines.loc[:, 'electric_grid_name'] = combined_scenario_name

# The MV grid becomes the new combined grid
combined_electric_grid_data = mv_electric_grid_data

# Remove original DERs at the respective node (LV grid replaces the MV load)
# This must be done before adding the LV grids, as they would be deleted again otherwise
for index, row in map_grids.iterrows():
    mv_node_name = str(row['mv_node_name'])
    combined_electric_grid_data.electric_grid_ders = \
        combined_electric_grid_data.electric_grid_ders.drop(combined_electric_grid_data.electric_grid_ders[
                                                                combined_electric_grid_data.electric_grid_ders[
                                                                    'node_name'] == mv_node_name].index)
    print(f'CAUTION: All DERs at MV node {mv_node_name} have been removed and will be replaced with LV grid.')

# Loop through LV grids in the map and add them to the scenario
for index, row in map_grids.iterrows():
    lv_scenario_name = str(row['lv_grid_name'])
    mv_node_name = str(row['mv_node_name'])

    print(f'Adding LV grid {lv_scenario_name} to MV node {mv_node_name}.')

    # Load data of the lv grid
    lv_electric_grid_data = fledge.data_interface.ElectricGridData(lv_scenario_name)

    # Get source node and rename it to the MV node name in the nodes table
    lv_source_node_name = lv_electric_grid_data.electric_grid['source_node_name']

    # Remove original source node from nodes table
    lv_electric_grid_data.electric_grid_nodes = \
        lv_electric_grid_data.electric_grid_nodes.drop(lv_electric_grid_data.electric_grid_nodes[
                                                                lv_electric_grid_data.electric_grid_nodes[
                                                                    'node_name'] == lv_source_node_name].index)

    # Change node names in all relevant tables
    lv_electric_grid_data.electric_grid_nodes.loc[:, 'node_name'] = \
        mv_node_name + '_' + lv_electric_grid_data.electric_grid_nodes.loc[:, 'node_name']
    lv_electric_grid_data.electric_grid_ders.loc[:, 'node_name'] = \
        mv_node_name + '_' + lv_electric_grid_data.electric_grid_ders.loc[:, 'node_name']
    lv_electric_grid_data.electric_grid_lines.loc[:, 'node_1_name'] = \
        mv_node_name + '_' + lv_electric_grid_data.electric_grid_lines.loc[:, 'node_1_name']
    lv_electric_grid_data.electric_grid_lines.loc[:, 'node_2_name'] = \
        mv_node_name + '_' + lv_electric_grid_data.electric_grid_lines.loc[:, 'node_2_name']

    # Change line names
    lv_electric_grid_data.electric_grid_lines.loc[:, 'line_name'] = \
        mv_node_name + '_' + lv_electric_grid_data.electric_grid_lines.loc[:, 'line_name']

    # Change trafo names
    lv_electric_grid_data.electric_grid_transformers.loc[:, 'transformer_name'] = \
        mv_node_name + '_' + lv_electric_grid_data.electric_grid_transformers.loc[:, 'transformer_name']

    # Change DER names
    lv_electric_grid_data.electric_grid_ders.loc[:, 'der_name'] = \
        mv_node_name + '_' + lv_electric_grid_data.electric_grid_ders.loc[:, 'der_name']

    # Check if first trafo node equals the source node of the network. If so, replace it by the MV node name
    for index_trafo, row_trafo in lv_electric_grid_data.electric_grid_transformers.iterrows():
        if row_trafo['node_1_name'] == lv_source_node_name:
            trafo_node_1_name = mv_node_name
        else:
            trafo_node_1_name = mv_node_name + '_' + \
                                lv_electric_grid_data.electric_grid_transformers.loc[index, 'node_1_name']
        lv_electric_grid_data.electric_grid_transformers.loc[index_trafo, 'node_1_name'] = trafo_node_1_name
    # Second trafo node name is always part of the LV network. Replace with new node name
    lv_electric_grid_data.electric_grid_transformers.loc[:, 'node_2_name'] = \
        mv_node_name + '_' + lv_electric_grid_data.electric_grid_transformers.loc[:, 'node_2_name']

    # Change electric_grid_name in the relevant lv tables
    lv_electric_grid_data.electric_grid_nodes.loc[:, 'electric_grid_name'] = combined_scenario_name
    lv_electric_grid_data.electric_grid_transformers.loc[:, 'electric_grid_name'] = combined_scenario_name
    lv_electric_grid_data.electric_grid_ders.loc[:, 'electric_grid_name'] = combined_scenario_name
    lv_electric_grid_data.electric_grid_lines.loc[:, 'electric_grid_name'] = combined_scenario_name

    # Add relevant tables to the new combined network
    # add all line types from lv grid to the mv grid
    combined_electric_grid_data.electric_grid_line_types = \
        combined_electric_grid_data.electric_grid_line_types.append(
            lv_electric_grid_data.electric_grid_line_types
        ).reset_index(drop=True)
    # add all line type matrices
    combined_electric_grid_data.electric_grid_line_types_matrices = \
        combined_electric_grid_data.electric_grid_line_types_matrices.append(
            lv_electric_grid_data.electric_grid_line_types_matrices
        ).reset_index(drop=True)
    # add lines to main lines table
    combined_electric_grid_data.electric_grid_lines = \
        combined_electric_grid_data.electric_grid_lines.append(
            lv_electric_grid_data.electric_grid_lines
        ).reset_index(drop=True)
    # add the DERs to the main DER table
    combined_electric_grid_data.electric_grid_ders = \
        combined_electric_grid_data.electric_grid_ders.append(
            lv_electric_grid_data.electric_grid_ders
        ).reset_index(drop=True)
    # add nodes to the main nodes table
    combined_electric_grid_data.electric_grid_nodes = \
        combined_electric_grid_data.electric_grid_nodes.append(
            lv_electric_grid_data.electric_grid_nodes
        ).reset_index(drop=True)
    # add transformers to the main transformers table
    combined_electric_grid_data.electric_grid_transformers = \
        combined_electric_grid_data.electric_grid_transformers.append(
            lv_electric_grid_data.electric_grid_transformers
        ).reset_index(drop=True)


# Separate tables into transformer_type and transformers --> only add unique transformer types
# electric_grid_transformer_types
transformer_type_columns = [
    'transformer_type',
    'resistance_percentage',
    'reactance_percentage',
    'tap_maximum_voltage_per_unit',
    'tap_minimum_voltage_per_unit'
    ]
combined_electric_grid_data.electric_grid_transformer_types = \
    combined_electric_grid_data.electric_grid_transformers[transformer_type_columns]
combined_electric_grid_data.electric_grid_transformer_types = \
    combined_electric_grid_data.electric_grid_transformer_types.drop_duplicates()
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
combined_electric_grid_data.electric_grid_transformers = \
    combined_electric_grid_data.electric_grid_transformers[transformers_columns]
combined_electric_grid_data.electric_grid_transformers = \
    combined_electric_grid_data.electric_grid_transformers.drop_duplicates()

# Drop duplicates in the line types (matrices) table
combined_electric_grid_data.electric_grid_line_types = \
    combined_electric_grid_data.electric_grid_line_types.drop_duplicates()
combined_electric_grid_data.electric_grid_line_types_matrices = \
    combined_electric_grid_data.electric_grid_line_types_matrices.drop_duplicates()

# Transform series into dataframes
combined_electric_grid_data.electric_grid = pd.DataFrame(combined_electric_grid_data.electric_grid).T
combined_electric_grid_data.scenario_data.scenario = pd.DataFrame(combined_electric_grid_data.scenario_data.scenario).T

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
combined_electric_grid_data.scenarios = combined_electric_grid_data.scenario_data.scenario[scenarios_columns]

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
combined_electric_grid_data.electric_grid_lines = combined_electric_grid_data.electric_grid_lines[lines_columns]

# Export csv file to output_path/combined_scenario_name
combined_electric_grid_data.electric_grid_nodes.to_csv(os.path.join(output_path, combined_scenario_name, 'electric_grid_nodes.csv'), index=False)
combined_electric_grid_data.electric_grid_lines.to_csv(os.path.join(output_path, combined_scenario_name, 'electric_grid_lines.csv'), index=False)
combined_electric_grid_data.electric_grid_ders.to_csv(os.path.join(output_path, combined_scenario_name, 'electric_grid_ders.csv'), index=False)
combined_electric_grid_data.electric_grid.to_csv(os.path.join(output_path, combined_scenario_name, 'electric_grids.csv'), index=False)
combined_electric_grid_data.electric_grid_transformers.to_csv(os.path.join(output_path, combined_scenario_name, 'electric_grid_transformers.csv'), index=False)
combined_electric_grid_data.scenarios.to_csv(os.path.join(output_path, combined_scenario_name, 'scenarios.csv'), index=False, date_format='%Y-%m-%dT%H:%M:%S')

# Do not export line types and matrices as well as transformer types as the names were not changed and they continue to exist in the original files
# combined_electric_grid_data.electric_grid_line_types.to_csv(os.path.join(output_path, combined_scenario_name, 'electric_grid_line_types.csv'), index=False)
# combined_electric_grid_data.electric_grid_line_types_matrices.to_csv(os.path.join(output_path, combined_scenario_name, 'electric_grid_line_types_matrices.csv'), index=False)
# combined_electric_grid_data.electric_grid_transformer_types.to_csv(os.path.join(output_path, combined_scenario_name, 'electric_grid_transformer_types.csv'), index=False)

print('Note: The geographical information of the nodes must be adjusted manually.')
print(f'Done. The combined grid can be found in: {os.path.join(output_path, combined_scenario_name)}')
