"""Module with helper functions specifically for the DLMP analysis
"""

import pandas as pd
import shutil
import os

import fledge.data_interface
import fledge.config

path_to_data = fledge.config.config['paths']['data']


def aggregate_electric_grids(
        mv_scenario_name: str,
        aggregated_scenario_name: str,
        path_to_map_grids: str
):
    """Method to aggregate DERs from multiple low voltage grids to nodes of one MV grid based on a mapping table
    """
    # Load csv-file as dataframe that contains the mapping of LV grids to MV nodes
    map_grids = __load_data(path_to_map_grids)

    # Create output folder for the new grid
    output_path = __create_output_folder(scenario_name=aggregated_scenario_name)

    # Load data of MV grid which is the underlying grid for our new
    aggregated_electric_grid_data = fledge.data_interface.ElectricGridData(mv_scenario_name)

    # Change electric grid name to new name
    __change_electric_grid_name(aggregated_electric_grid_data, aggregated_scenario_name)
    aggregated_electric_grid_data.scenario_data.scenario.loc['scenario_name'] = aggregated_scenario_name

    # Remove original DERs at the respective node (LV grid replaces the MV load)
    __remove_original_ders(aggregated_electric_grid_data, map_grids)

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

    # Format all dataframes correctly for export
    __format_grid_tables(aggregated_electric_grid_data)

    # call export function
    __export_grid_to_csv(aggregated_electric_grid_data, output_path)


def combine_electric_grids(
        mv_scenario_name: str,
        combined_scenario_name: str,
        map_grids_filename: str
):
    """Method to combine multiple grids to one large grids based on a mapping table
    """

    # Load csv-file as dataframe that contains the mapping of LV grids to MV nodes
    map_grids = __load_data(map_grids_filename)

    output_path = __create_output_folder(scenario_name=combined_scenario_name)

    # Load data of MV grid
    mv_electric_grid_data = fledge.data_interface.ElectricGridData(mv_scenario_name)

    # Change electric grid name to new name
    __change_electric_grid_name(mv_electric_grid_data, combined_scenario_name)
    mv_electric_grid_data.scenario_data.scenario.loc['scenario_name'] = combined_scenario_name

    # The MV grid becomes the new combined grid
    combined_electric_grid_data = mv_electric_grid_data

    # Remove original DERs from MV grid
    __remove_original_ders(combined_electric_grid_data, map_grids)

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

    # Format all dataframes correctly for export
    __format_grid_tables(combined_electric_grid_data)

    # call export function
    __export_grid_to_csv(combined_electric_grid_data, output_path)


def __load_data(
        path_to_map_grids: str
):
    map_grids = pd.read_csv(os.path.join(os.path.dirname(path_to_data), path_to_map_grids))
    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()
    return map_grids

def __create_output_folder(
        scenario_name: str
):
    # Create output folder for the new grid
    output_path = os.path.join(path_to_data, scenario_name)
    # Instantiate dedicated directory for current grid.
    shutil.rmtree(output_path, ignore_errors=True)
    os.mkdir(output_path)
    return output_path


def __change_electric_grid_name(
        grid_data: fledge.data_interface.ElectricGridData,
        new_grid_name: str
):
    grid_data.electric_grid.loc['electric_grid_name'] = new_grid_name
    grid_data.electric_grid_nodes.loc[:, 'electric_grid_name'] = new_grid_name
    grid_data.scenario_data.scenario.loc['electric_grid_name'] = new_grid_name
    grid_data.electric_grid_transformers.loc[:, 'electric_grid_name'] = new_grid_name
    grid_data.electric_grid_ders.loc[:, 'electric_grid_name'] = new_grid_name
    grid_data.electric_grid_lines.loc[:, 'electric_grid_name'] = new_grid_name


def __remove_original_ders(
        grid_data: fledge.data_interface.ElectricGridData,
        map_grids: pd.DataFrame
):
    # Remove original DERs at the respective node (LV grid replaces the MV load)
    # This must be done before adding the LV grids, as they would be deleted again otherwise
    for index, row in map_grids.iterrows():
        mv_node_name = str(row['mv_node_name'])
        grid_data.electric_grid_ders = \
            grid_data.electric_grid_ders.drop(grid_data.electric_grid_ders[
                                                                    grid_data.electric_grid_ders[
                                                                        'node_name'] == mv_node_name].index)
        print(f'CAUTION: All DERs at MV node {mv_node_name} have been removed and will be replaced with LV grid.')


def __format_grid_tables(
        grid_data: fledge.data_interface.ElectricGridData,
):
    # Transform series into dataframes
    grid_data.electric_grid = pd.DataFrame(grid_data.electric_grid).T
    grid_data.scenario_data.scenario = pd.DataFrame(
        grid_data.scenario_data.scenario).T

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
    grid_data.scenarios = grid_data.scenario_data.scenario[scenarios_columns]

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
    grid_data.electric_grid_lines = grid_data.electric_grid_lines[lines_columns]

    # Separate tables into transformer_type and transformers --> only add unique transformer types
    # electric_grid_transformer_types
    transformer_type_columns = [
        'transformer_type',
        'resistance_percentage',
        'reactance_percentage',
        'tap_maximum_voltage_per_unit',
        'tap_minimum_voltage_per_unit'
    ]
    grid_data.electric_grid_transformer_types = \
        grid_data.electric_grid_transformers[transformer_type_columns]
    grid_data.electric_grid_transformer_types = \
        grid_data.electric_grid_transformer_types.drop_duplicates()
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
    grid_data.electric_grid_transformers = \
        grid_data.electric_grid_transformers[transformers_columns]
    grid_data.electric_grid_transformers = \
        grid_data.electric_grid_transformers.drop_duplicates()

    # Drop duplicates in the line types (matrices) table
    grid_data.electric_grid_line_types = \
        grid_data.electric_grid_line_types.drop_duplicates()
    grid_data.electric_grid_line_types_matrices = \
        grid_data.electric_grid_line_types_matrices.drop_duplicates()


def __export_grid_to_csv(
        grid_data: fledge.data_interface.ElectricGridData,
        output_path: str
):
    # Export csv file to output_path/combined_scenario_name
    grid_data.electric_grid_nodes.to_csv(
        os.path.join(output_path, 'electric_grid_nodes.csv'), index=False)
    grid_data.electric_grid_lines.to_csv(
        os.path.join(output_path, 'electric_grid_lines.csv'), index=False)
    grid_data.electric_grid_ders.to_csv(
        os.path.join(output_path, 'electric_grid_ders.csv'), index=False)
    grid_data.electric_grid.to_csv(
        os.path.join(output_path, 'electric_grids.csv'), index=False)
    grid_data.electric_grid_transformers.to_csv(
        os.path.join(output_path, 'electric_grid_transformers.csv'), index=False)
    grid_data.scenarios.to_csv(os.path.join(output_path, 'scenarios.csv'),
                                                 index=False, date_format='%Y-%m-%dT%H:%M:%S')
    # Do not export line types and matrices as well as transformer types as the names were not changed and they continue to exist in the original files
    # combined_electric_grid_data.electric_grid_line_types.to_csv(os.path.join(output_path, combined_scenario_name, 'electric_grid_line_types.csv'), index=False)
    # combined_electric_grid_data.electric_grid_line_types_matrices.to_csv(os.path.join(output_path, combined_scenario_name, 'electric_grid_line_types_matrices.csv'), index=False)
    # combined_electric_grid_data.electric_grid_transformer_types.to_csv(os.path.join(output_path, combined_scenario_name, 'electric_grid_transformer_types.csv'), index=False)
    print('Note: The geographical information of the nodes must be adjusted manually.')
    print(f'Done. The grid can be found in: {output_path}')
