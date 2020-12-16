"""Module with helper functions specifically for the DLMP analysis
"""

import pandas as pd
import shutil
import os
import random
import numpy as np
import datetime

import fledge.data_interface
import fledge.config

path_to_data = fledge.config.config['paths']['data']


def generate_fixed_load_der_input_data(
        scenario_name: str,
        path_to_der_schedules_data: str,
        num_of_loads=np.inf
):
    # per default, this function generates a load profile at every node (num_of_loads=inf)

    der_schedules = __load_data(path_to_der_schedules_data)

    # load scenario data to get the relevant time period and grid data (nodes)
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)

    # in der_schedules, there is a unique timeseries for every week of the year with the week number as identifier
    # and related to the season
    [season, weeknums] = __get_season_of_scenario_data(scenario_data)

    grid_data = fledge.data_interface.ElectricGridData(scenario_name)
    nodes = grid_data.electric_grid_nodes

    der_data = fledge.data_interface.DERData(scenario_name)

    # Drop / remove all data from dataframe (if exist)
    der_data.ders.drop(der_data.ders.index, inplace=True)
    grid_data.electric_grid_ders.drop(grid_data.electric_grid_ders.index, inplace=True)
    print('Note: All original DERs were deleted from the electric grid model scenario.')

    counter = 1
    der_type = 'fixed_load'
    definition_type = 'schedule'
    power_factor = 0.95
    source_node_name = grid_data.electric_grid['source_node_name']
    for node_name in nodes['node_name']:
        if counter > num_of_loads:
            break
        # don't add a new der to the source node
        if node_name == source_node_name:
            continue

        der_name = 'Load R' + node_name
        # TODO: loop over all phases
        der_model_name = __pick_random_consumer_type() + '_' + season + '_' + random.choice(weeknums) + '_phase_0'

        active_power_nominal = (-1) * max(der_schedules.loc[der_schedules['definition_name'] == der_model_name, 'value'])
        reactive_power_nominal = active_power_nominal * np.tan(np.arccos(power_factor))

        # Add to ders data
        der_data.ders = der_data.ders.append({
            'der_name': der_name,
            'der_model_name': der_model_name,
            'der_type': der_type,
            'definition_name': der_model_name,
            'definition_type': definition_type
        }, ignore_index=True)
        # Add to electric grid ders dataframe
        grid_data.electric_grid_ders = grid_data.electric_grid_ders.append({
            'electric_grid_name': scenario_name,
            'der_name': der_name,
            'der_type': der_type,
            'der_model_name': der_model_name,
            'node_name': node_name,
            'is_phase_1_connected': '1',
            'is_phase_2_connected': '0',
            'is_phase_3_connected': '0',
            'connection': 'wye',
            'active_power_nominal': active_power_nominal,
            'reactive_power_nominal': reactive_power_nominal,
            'in_service': '1'
        }, ignore_index=True)

        counter += 1

    __export_der_data_to_csv(grid_data, der_data, output_path=os.path.join(path_to_data, scenario_name))


def __pick_random_consumer_type() -> str:
    consumer_types = {
        1: 'One_full-time_working_person',
        2: 'One_pensioneer',
        3: 'Two_full-time_working_persons',
        4: 'Two_pensioneers',
        5: 'One_full-time_and_one_part-time_working_person',
        6: 'Two_full-time_working_persons_one_child',
        7: 'One_full-time_and_one_part-time_working_person_one_child',
        8: 'Two_full-time_working_persons_two_children',
        9: 'One_full-time_and_one_part-time_working_person_two_children',
        10: 'Two_full-time_working_persons_three_children',
        11: 'One_full-time_and_one_part-time_working_person_three_children'
    }
    key_list = list(consumer_types.keys())
    return consumer_types[random.choice(key_list)]


def __get_season_of_scenario_data(
        scenario_data: fledge.data_interface.ScenarioData
) -> [str, list]:

    # TODO: this is currently copied from notebook, there should be one central spot!
    year = scenario_data.timesteps.year[0]
    # According to German seasons
    season_start_dates = {
        'spring': datetime.date(year, 3, 1),
        'summer': datetime.date(year, 6, 1),
        'fall': datetime.date(year, 9, 1),
        'winter_1': datetime.date(year, 12, 1),
        'winter_2': datetime.date(year, 1, 1)
    }
    season_end_dates = {
        'spring': datetime.datetime(year, 5, 31, 23, 59, 59),
        'summer': datetime.datetime(year, 8, 31, 23, 59, 59),
        'fall': datetime.datetime(year, 11, 30, 23, 59, 59),
        'winter_1': datetime.datetime(year, 12, 31, 23, 59, 59),
        'winter_2': datetime.datetime(year, 2, 28, 23, 59, 59)
    }

    # Adjust all dates so that every season starts on a Monday (for compatibility with fledge)
    for date in season_start_dates:
        if 'winter_2' not in date:
            season_start_dates[date] = season_start_dates[date] + datetime.timedelta(
                days=-season_start_dates[date].weekday())
        else:
            # Here we have to cut the first days of the year before the first Monday
            season_start_dates[date] = season_start_dates[date] + datetime.timedelta(
                days=-season_start_dates[date].weekday(), weeks=1)

    for date in season_end_dates:
        if season_end_dates[date].weekday() != 6:
            season_end_dates[date] = season_end_dates[date] + datetime.timedelta(
                days=-season_end_dates[date].weekday() - 1)

    seasons = {}
    for season in season_start_dates:
        if 'winter' not in season:
            seasons[season] = pd.date_range(start=season_start_dates[season], end=season_end_dates[season],
                                            freq='T'),  # 'T' for minutely intervals
        else:
            seasons['winter'] = pd.date_range(start=season_start_dates['winter_2'], end=season_end_dates['winter_2'],
                                              freq='T').append(
                pd.date_range(start=season_start_dates['winter_1'], end=season_end_dates['winter_1'], freq='T'))
    season = None
    for season in seasons.keys():
        if scenario_data.timesteps[0] in seasons[season]:
            break

    time_format = '%W'
    weeknums = seasons[season].strftime(time_format).unique().to_list()

    return [season, weeknums]


def increase_der_penetration_of_scenario_on_lv_level(
        scenario_name: str,
        path_to_der_data: str,
        penetration_ratio: float = 1.0,
        new_scenario_name: str = None
) -> str:
    der_data = __load_data(path_to_der_data)

    if new_scenario_name is None:
        new_scenario_name = scenario_name + '_increased_der_penetration'

    # Create output folder for the new grid
    output_path = __create_output_folder(scenario_name=new_scenario_name)

    grid_data = fledge.data_interface.ElectricGridData(scenario_name)

    # Loop through der data and add to the scenario
    for new_der_index, new_der_row in der_data.iterrows():
        additional_der_name = str(new_der_row['der_name'])
        num_of_lv_nodes = len(grid_data.electric_grid_ders[grid_data.electric_grid_ders['der_name'].str.contains('_')])
        count = 1
        for der_index, der_row in grid_data.electric_grid_ders.iterrows():
            if count > int(num_of_lv_nodes * penetration_ratio):
                break
            der_name = der_row['der_name']
            if '_' in der_name:
                node = der_row['node_name']
                print(f'Adding DER {additional_der_name} to node {node} in scenario {scenario_name}.')
                new_der_row['node_name'] = str(node)
                new_der_row['der_name'] = str(der_name) + '_' + additional_der_name + '_' + str(count)
                # Add the DER to the main DER table
                grid_data.electric_grid_ders = \
                    grid_data.electric_grid_ders.append(new_der_row).reset_index(drop=True)
                count += 1

    # Change electric grid name to new name
    __change_electric_grid_name(grid_data, new_scenario_name)
    grid_data.scenario_data.scenario.loc['scenario_name'] = new_scenario_name

    # call export function
    __export_grid_data_to_csv(grid_data, output_path)

    return new_scenario_name


def aggregate_electric_grids(
        mv_scenario_name: str,
        path_to_map_grids: str,
        aggregated_scenario_name: str = None
) -> str:
    """
    Method to aggregate DERs from multiple low voltage grids to nodes of one MV grid based on a mapping table
    :param mv_scenario_name: name of the scenario for the medium voltage electric grid
    :param path_to_map_grids: relative path within fledge project folder, e.g. 'examples/electric_grid_mapping.csv'
    :param aggregated_scenario_name: optional, name of the aggregated scenario (will also be returned by the function)
    :return: the name of the exported scenario
    """
    # Load csv-file as dataframe that contains the mapping of LV grids to MV nodes
    map_grids = __load_data(path_to_map_grids)

    if aggregated_scenario_name is None:
        aggregated_scenario_name = 'aggregated_' + mv_scenario_name

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
        # Check if DER name contains an underscore, if so, stop!
        if any(lv_electric_grid_data.electric_grid_ders.loc[:, 'der_name'].str.contains('_')):
            raise NameError('DERs of the LV grid may not have an underscore in their name!')
        lv_electric_grid_data.electric_grid_ders.loc[:, 'der_name'] = \
            mv_node_name + '_' + lv_electric_grid_data.electric_grid_ders.loc[:, 'der_name']

        # Add the DERs to the main DER table
        aggregated_electric_grid_data.electric_grid_ders = \
            aggregated_electric_grid_data.electric_grid_ders.append(
                lv_electric_grid_data.electric_grid_ders
            ).reset_index(drop=True)

    # call export function
    __export_grid_data_to_csv(aggregated_electric_grid_data, output_path)

    return aggregated_scenario_name


def combine_electric_grids(
        mv_scenario_name: str,
        map_grids_filename: str,
        combined_scenario_name: str = None
) -> str:
    """
    Method to combine multiple grids to one large grids based on a mapping table
    :param mv_scenario_name: name of the scenario for the medium voltage electric grid
    :param path_to_map_grids: relative path within fledge project folder, e.g. 'examples/electric_grid_mapping.csv'
    :param combined_scenario_name: optional, name of the combined scenario (will also be returned by the function)
    :return: the name of the exported scenario
    """

    # Load csv-file as dataframe that contains the mapping of LV grids to MV nodes
    map_grids = __load_data(map_grids_filename)

    if combined_scenario_name is None:
        combined_scenario_name = 'combined_' + mv_scenario_name

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

    # call export function
    __export_grid_data_to_csv(combined_electric_grid_data, output_path)

    return combined_scenario_name


def __load_data(
        path_to_csv_data: str,
) -> pd.DataFrame:
    print('Loading data...')
    csv_data = pd.read_csv(os.path.join(os.path.dirname(path_to_data), path_to_csv_data))
    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()
    return csv_data


def __create_output_folder(
        scenario_name: str
) -> str:
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


def __format_grid_data_tables(
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
        "price_type",
        "price_sensitivity_coefficient",
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


def __format_der_data_tables(
        der_data: fledge.data_interface.DERData,
):
    # Drop columns of the lines table that are not part of the csv-file
    der_models_columns = [
        'der_type',
        'der_model_name',
        'definition_type',
        'definition_name',
        'power_per_unit_minimum',
        'power_per_unit_maximum',
        'power_factor_minimum',
        'power_factor_maximum',
        'energy_storage_capacity_per_unit',
        'charging_efficiency',
        'self_discharge_rate',
        'marginal_cost'
    ]
    der_data.ders = der_data.ders[der_models_columns]


def __export_grid_data_to_csv(
        grid_data: fledge.data_interface.ElectricGridData,
        output_path: str
):
    # Format all dataframes correctly for export
    __format_grid_data_tables(grid_data)

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


def __export_der_data_to_csv(
        grid_data: fledge.data_interface.ElectricGridData,
        der_data: fledge.data_interface.DERData,
        output_path: str
):
    # Format all dataframes correctly for export
    __format_grid_data_tables(grid_data)
    __format_der_data_tables(der_data)

    grid_data.electric_grid_ders.to_csv(
        os.path.join(output_path, 'electric_grid_ders.csv'), index=False)
    der_data.ders.to_csv(
        os.path.join(output_path, 'der_models.csv'), index=False)
    print(f'Done exporting to: {output_path}')
