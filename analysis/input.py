"""
Module with helper classes and functions for generating the necessary input files for the DLMP analysis
"""

import pandas as pd
import shutil
import os
import random
import numpy as np

import fledge.data_interface
import fledge.config
import analysis.utils


class DataFactory(object):
    """Base class of all factory objects that can generate the csv-files (see below)"""
    data: object  # This is DERData, ElectricGridData, etc.
    csv_columns: dict
    csv_filenames: dict

    def format_data_tables(
            self,
            table_name: str = None
    ) -> pd.DataFrame:
        # Get the tables data
        if table_name is None:
            data_tables = vars(self.data)
        else:
            # If function is called with individual table request, instead of all
            data_tables = [table_name]
        table = None
        for table_name in data_tables:
            if table_name in self.csv_filenames.keys():
                if table_name is 'scenario_data':
                    table = self.data.scenario_data.scenario
                else:
                    table = getattr(self.data, table_name)
                # Transform Series to DataFrame if needed
                if type(table) is pd.Series:
                    table = pd.DataFrame(table).T
                if table_name in self.csv_columns.keys():
                    # Drop some of columns that are not part of final csv
                    table = table[self.csv_columns[table_name]]
                # Drop duplicates
                table = table.drop_duplicates()
                if table_name is 'scenario_data':
                    self.data.scenario_data.scenario = table
                else:
                    setattr(self.data, table_name, table)

        # return the last table (or the table the was formatted based on argument table_name
        return table

    def export_data_to_csv(
            self,
            output_path: str,
            table_name: str = None
    ):
        self.format_data_tables(table_name=table_name)
        # Get the tables data
        if table_name is None:
            data_tables = vars(self.data)
        else:
            # If function is called with individual table request, instead of all
            data_tables = [table_name]

        for table_name in data_tables:
            if table_name in self.csv_filenames.keys():
                if table_name is 'scenario_data':
                    table = self.data.scenario_data.scenario
                else:
                    table = getattr(self.data, table_name)
                table.to_csv(
                    os.path.join(output_path, self.csv_filenames[table_name]),
                    index=False,
                    date_format='%Y-%m-%dT%H:%M:%S'
                )
                print(f'Successfully exported {self.csv_filenames[table_name]} data to {output_path}')

    def add_row_with_new_values(
            self,
            key_value_pairs: dict,
            row_index: pd.Index = None,
            table_name: str = None
    ):
        """
        Copies last row of table or row provided by index (either all tables or specified by table_name) and edits the
        value if key in table column.
        :param key_value_pairs:
        :param row_index:
        :param table_name:
        :return:
        """
        # Get the tables data
        if table_name is None:
            data_tables = vars(self.data)
        else:
            # If function is called with individual table request, instead of all
            data_tables = [table_name]

        for table_name in data_tables:
            if table_name is 'scenario_data':
                table = self.data.scenario_data.scenario
            else:
                table = getattr(self.data, table_name)

            if type(table) is pd.Series:
                # Format the data_table, so we can actually add to it if it is a series
                table = self.format_data_tables(table_name=table_name)

            if row_index is not None:
                copy_row = table.loc[row_index, :].copy()
            else:
                # Get last row and append
                copy_row = table.tail(1).copy()
            value_changed = False
            for key in key_value_pairs:
                if key in table.keys():
                    copy_row[key] = key_value_pairs[key]
                    value_changed = True

            if value_changed:
                # Add the new row
                table = table.append(copy_row, ignore_index=True)
                if table_name is 'scenario_data':
                    self.data.scenario_data.scenario = table
                else:
                    setattr(self.data, table_name, table)

    def remove_entry_from_tables(
            self,
            entry_name: str,
            attribute_name: str,
            table_name: str = None
    ):
        # TODO needs proper testing
        raise NotImplementedError
        # Get the tables data
        # if table_name is None:
        #     data_tables = vars(self.data)
        # else:
        #     # If function is called with individual table request, instead of all
        #     data_tables = [table_name]
        #
        # for table_name in data_tables:
        #     if table_name is 'scenario_data':
        #         table = self.data.scenario_data.scenario
        #     else:
        #         table = getattr(self.data, table_name)
        #     if attribute_name in table.keys():
        #         if type(table) is pd.DataFrame:
        #                 table = table.drop(table.loc[table[attribute_name] == entry_name].index)
        #         elif type(table) is pd.Series:
        #             table.loc[attribute_name] = None
        #         else:
        #             print(f'{type(table)} not supported')
        #
        #         if table_name is 'scenario_data':
        #             self.data.scenario_data.scenario = table
        #         else:
        #             setattr(self.data, table_name, table)

    def add_data_to_tables(
            self,
            add_data: object,
            table_name: str = None
    ):
        """
        Adds data of the same type of data and table to the data object (of type DERData, etc.) stored in self
        :param add_data:
        :param table_name:
        :return:
        """
        if type(add_data) != type(self.data):
            print(f'Cannot add data of type {type(add_data)} to data of type {type(self.data)}')
            raise ValueError

        # Get the tables data
        if table_name is None:
            data_tables = vars(self.data)
        else:
            # If function is called with individual table request, instead of all
            data_tables = [table_name]

        for table_name in data_tables:
            if table_name is 'scenario_data':
                table = self.data.scenario_data.scenario
                add_table = add_data.scenario_data.scenario
            else:
                table = getattr(self.data, table_name)
                add_table = getattr(add_data, table_name)
            table = table.append(add_table).reset_index(drop=True)
            setattr(self.data, table_name, table)

    def change_attribute_value(
            self,
            new_value,
            attribute_name: str,
            row_index: pd.Index = None,
            table_name: str = None
    ):
        # Get the tables data
        if table_name is None:
            data_tables = vars(self.data)
        else:
            # If function is called with individual table request, instead of all
            data_tables = [table_name]

        for table_name in data_tables:
            if table_name is 'scenario_data':
                table = self.data.scenario_data.scenario
            else:
                table = getattr(self.data, table_name)
            if attribute_name in table.keys():
                if type(table) is pd.DataFrame:
                    if row_index is None:
                        table.loc[:, attribute_name] = new_value
                    else:
                        table.loc[row_index, attribute_name] = new_value
                elif type(table) is pd.Series:
                    table.loc[attribute_name] = new_value
                else:
                    print(f'{type(table)} not supported')

    def add_prefix_to_attribute(
            self,
            prefix: str,
            attribute_name: str,
            row_index: pd.Index = None,
            table_name: str = None
    ):
        # Get the tables data
        if table_name is None:
            data_tables = vars(self.data)
        else:
            # If function is called with individual table request, instead of all
            data_tables = [table_name]

        for table_name in data_tables:
            if table_name is 'scenario_data':
                table = self.data.scenario_data.scenario
            else:
                table = getattr(self.data, table_name)
            if attribute_name in table.keys():
                if type(table) is pd.DataFrame:
                    if row_index is None:
                        table.loc[:, attribute_name] = prefix + table.loc[:, attribute_name]
                    else:
                        table.loc[row_index, attribute_name] = prefix + table.loc[row_index, attribute_name]
                elif type(table) is pd.Series:
                    table.loc[attribute_name] = prefix + table.loc[attribute_name]
                else:
                    print(f'{type(table)} not supported')


class ElectricGridDataFactory(DataFactory):
    def __init__(
            self,
            data: fledge.data_interface.ElectricGridData
    ):
        self.data = data

        self.csv_columns = {
            'electric_grid_lines': [
                'line_name',
                'electric_grid_name',
                'line_type',
                'node_1_name',
                'node_2_name',
                'is_phase_1_connected',
                'is_phase_2_connected',
                'is_phase_3_connected',
                'length'
            ],
            'electric_grid_transformer_types': [
                'transformer_type',
                'resistance_percentage',
                'reactance_percentage',
                'tap_maximum_voltage_per_unit',
                'tap_minimum_voltage_per_unit'
            ],
            'electric_grid_transformers': [
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
            ],
            'electric_grid_ders': [
                'electric_grid_name',
                'der_name',
                'der_type',
                'der_model_name',
                'node_name',
                'is_phase_1_connected',
                'is_phase_2_connected',
                'is_phase_3_connected',
                'connection',
                'active_power_nominal',
                'reactive_power_nominal',
                'in_service'
            ]
        }

        self.csv_filenames = {
            'electric_grid_nodes': 'electric_grid_nodes.csv',
            'electric_grid_lines': 'electric_grid_lines.csv',
            'electric_grid': 'electric_grids.csv',
            'electric_grid_transformers': 'electric_grid_transformers.csv',
            'electric_grid_ders': 'electric_grid_ders.csv',
        }

    def remove_ders_from_grid(
            self,
            map_grids: pd.DataFrame
    ):
        # TODO: map_grids should be a property of the class
        # TODO: this should return the nodes of the removed DERs
        # Remove original DERs at the respective node (LV grid replaces the MV load)
        # This must be done before adding the LV grids, as they would be deleted again otherwise
        for index, row in map_grids.iterrows():
            mv_node_name = str(row['mv_node_name'])
            self.data.electric_grid_ders = \
                self.data.electric_grid_ders.drop(self.data.electric_grid_ders[
                                                      self.data.electric_grid_ders[
                                                          'node_name'] == mv_node_name].index)
            print(f'CAUTION: All DERs at MV node {mv_node_name} have been removed and will be replaced with LV grid.')

    def add_der_data_to_grid_data(
            self,
            der_data: fledge.data_interface.DERData
    ):
        # TODO: this function should do something more, like renaming the electric_grid_name attribute
        # This function adds all DERs in der_data to electric_grid_ders table
        table = der_data.ders.copy()
        # Drop some of columns that are not part of final csv
        der_data.electric_grid_ders = table[self.csv_columns['electric_grid_ders']]
        self.add_data_to_tables(der_data, table_name='electric_grid_ders')

    def add_der_to_electric_grid_data(
            self,
            der_table_row: pd.Series
    ):
        # Add the DER to the main DER table
        self.data.electric_grid_ders = (
            self.data.electric_grid_ders.append(der_table_row).reset_index(drop=True)
        )


class DERDataFactory(DataFactory):
    def __init__(
            self,
            data: fledge.data_interface.DERData
    ):
        self.data = data

        self.csv_columns = {
            'ders': [
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
        }

        self.csv_filenames = {
            'ders': 'der_models.csv'
        }

    def generate_der_data_for_der_type(
            self,
            der_type: str
    ):
        # Function adds a new der to existing DERData in self.data
        raise NotImplementedError

    # def generate_der_models_csv(
    #         self,
    #         der_schedules: pd.DataFrame,
    #         output_path: str
    # ):
    #     # Using the last scenario_name of the loop (it does not matter)
    #     # der_models.csv is only generated once for all possible der_model_names
    #     # Drop / remove all data from dataframe (if exist)
    #     ders = self.ders.copy()
    #     ders.drop(self.ders.index, inplace=True)
    #     for definition_name in der_schedules['definition_name'].unique():
    #         # Add to ders data
    #         der_data.ders = der_data.ders.append({
    #             'der_model_name': definition_name,
    #             'der_type': der_type,
    #             'definition_name': definition_name,
    #             'definition_type': definition_type
    #         }, ignore_index=True)
    #     self.export_data_to_csv(output_path=output_path, table_name='ders')


class ScenarioDataFactory(DataFactory):
    def __init__(
            self,
            data: fledge.data_interface.ScenarioData
    ):
        self.data = data

        self.csv_columns = {
            'scenario': [
                "scenario_name",
                "electric_grid_name",
                "thermal_grid_name",
                "parameter_set",
                "price_type",
                "price_sensitivity_coefficient",
                "timestep_start",
                "timestep_end",
                "timestep_interval",
                "base_apparent_power",
                "base_voltage"
            ]
        }

        self.csv_filenames = {
            'scenario': 'scenarios.csv',
        }


class ScenarioFactory(object):
    path_to_data: str
    calendar: analysis.utils.Calendar
    der_schedules: pd.DataFrame = None

    # TODO: Instead of reloading the database all the time, change some of the functions so that they use ScenarioData
    #  object instead of reloading based on scenario_name

    def __init__(self):
        self.path_to_data = fledge.config.config['paths']['data']
        self.calendar = analysis.utils.Calendar(2015)  # year 2015 as input data is from that year

    def aggregate_electric_grids(
            self,
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
        if mv_scenario_name is None:
            return ''

        # Load csv-file as dataframe that contains the mapping of LV grids to MV nodes
        map_grids = self.load_csv_into_dataframe(path_to_map_grids)

        if aggregated_scenario_name is None:
            aggregated_scenario_name = 'aggregated_' + mv_scenario_name

        # Create output folder for the new grid
        output_path = self.create_output_folder(scenario_name=aggregated_scenario_name)

        # Load data of MV grid which is the underlying grid for our new
        aggregated_electric_grid_data_factory = ElectricGridDataFactory(
            fledge.data_interface.ElectricGridData(mv_scenario_name)
        )

        new_values_dict = {
            'scenario_name': aggregated_scenario_name,
            'electric_grid_name': aggregated_scenario_name,

        }
        for attribute in new_values_dict:
            aggregated_electric_grid_data_factory.change_attribute_value(
                new_value=new_values_dict[attribute],
                attribute_name=attribute,
            )

        # Remove original DERs at the respective node (LV grid replaces the MV load)
        # TODO: this needs re-working
        aggregated_electric_grid_data_factory.remove_ders_from_grid(map_grids)

        # Loop through LV grids in the map and add them to the scenario
        for index, row in map_grids.iterrows():
            lv_scenario_name = str(row['lv_grid_name'])
            mv_node_name = str(row['mv_node_name'])

            print(f'Adding DERs of LV grid {lv_scenario_name} to MV node {mv_node_name}.')

            # Load data of the lv grid
            lv_electric_grid_data = fledge.data_interface.ElectricGridData(lv_scenario_name)
            lv_grid_data_factory = ElectricGridDataFactory(lv_electric_grid_data)

            new_values_dict = {
                'node_name': mv_node_name,
                'electric_grid_name': aggregated_scenario_name,

            }
            for attribute in new_values_dict:
                lv_grid_data_factory.change_attribute_value(
                    new_value=new_values_dict[attribute],
                    attribute_name=attribute,
                )
            # Change DER names
            # Check if DER name contains an underscore, if so, stop!
            if any(lv_electric_grid_data.electric_grid_ders.loc[:, 'der_name'].str.contains('_')):
                raise NameError('DERs of the LV grid may not have an underscore in their name!')

            prefix = mv_node_name + '_'
            lv_grid_data_factory.add_prefix_to_attribute(
                prefix=prefix,
                attribute_name='der_name',
                table_name='electric_grid_ders'
            )

            # Add the DERs to the main DER table
            aggregated_electric_grid_data_factory.add_data_to_tables(
                add_data=lv_electric_grid_data,
                table_name='electric_grid_ders'
            )

        # call export function
        aggregated_electric_grid_data_factory.export_data_to_csv(output_path)
        scenario_data_factory = ScenarioDataFactory(aggregated_electric_grid_data_factory.data.scenario_data)
        scenario_data_factory.export_data_to_csv(output_path)

        return aggregated_scenario_name

    def combine_electric_grids(
            self,
            mv_scenario_name: str,
            map_grids_filename: str,
            combined_scenario_name: str = None
    ) -> str:
        if mv_scenario_name is None:
            return ''
        """
        Method to combine multiple grids to one large grids based on a mapping table
        :param mv_scenario_name: name of the scenario for the medium voltage electric grid
        :param map_grids_filename: relative path within fledge project folder, e.g. 'examples/electric_grid_mapping.csv'
        :param combined_scenario_name: optional, name of the combined scenario (will also be returned by the function)
        :return: the name of the exported scenario
        """

        # Load csv-file as dataframe that contains the mapping of LV grids to MV nodes
        # TODO: who should actually call this function?
        # self.reload_database()
        map_grids = self.load_csv_into_dataframe(map_grids_filename)

        if combined_scenario_name is None:
            combined_scenario_name = 'combined_' + mv_scenario_name

        output_path = self.create_output_folder(scenario_name=combined_scenario_name)

        # Load data of MV grid
        mv_electric_grid_data = fledge.data_interface.ElectricGridData(mv_scenario_name)
        combined_grid_data_factory = ElectricGridDataFactory(mv_electric_grid_data)

        new_values_dict = {
            'scenario_name': combined_scenario_name,
            'electric_grid_name': combined_scenario_name,

        }
        for attribute in new_values_dict:
            combined_grid_data_factory.change_attribute_value(
                new_value=new_values_dict[attribute],
                attribute_name=attribute,
            )

        # Remove original DERs from MV grid
        combined_grid_data_factory.remove_ders_from_grid(map_grids)

        # Loop through LV grids in the map and add them to the scenario
        for index, row in map_grids.iterrows():
            lv_scenario_name = str(row['lv_grid_name'])
            mv_node_name = str(row['mv_node_name'])

            print(f'Adding LV grid {lv_scenario_name} to MV node {mv_node_name}.')

            # Load data of the lv grid
            lv_electric_grid_data = fledge.data_interface.ElectricGridData(lv_scenario_name)
            lv_grid_data_factory = ElectricGridDataFactory(lv_electric_grid_data)

            # Get source node and rename it to the MV node name in the nodes table
            lv_source_node_name = lv_electric_grid_data.electric_grid['source_node_name']

            # TODO: move to ElectricGridDataFactory as function?
            # Remove original source node from nodes table
            lv_electric_grid_data.electric_grid_nodes = \
                lv_electric_grid_data.electric_grid_nodes.drop(lv_electric_grid_data.electric_grid_nodes[
                                                                   lv_electric_grid_data.electric_grid_nodes[
                                                                       'node_name'] == lv_source_node_name].index)

            # Change names in all relevant tables by adding a prefix
            prefix = mv_node_name + '_'
            attributes = [
                'node_name',
                'node_1_name',
                'node_2_name',
                'line_name',
                'transformer_name',
                'der_name'
                ]
            for attribute in attributes:
                lv_grid_data_factory.add_prefix_to_attribute(
                    prefix=prefix,
                    attribute_name=attribute
                )

            # Check if first trafo node equals the source node of the network. If so, replace it by the MV node name
            for index_trafo, row_trafo in lv_electric_grid_data.electric_grid_transformers.iterrows():
                if row_trafo['node_1_name'] == prefix + lv_source_node_name:
                    lv_electric_grid_data.electric_grid_transformers.loc[index_trafo, 'node_1_name'] = mv_node_name

            # Change electric_grid_name in the relevant lv tables
            lv_grid_data_factory.change_attribute_value(
                new_value=combined_scenario_name,
                attribute_name='electric_grid_name',
            )

            # Add relevant tables to the new combined network
            # add all line types from lv grid to the mv grid
            data_tables = [
                'electric_grid_transformers',
                'electric_grid_nodes',
                'electric_grid_ders',
                'electric_grid_lines'
            ]
            for table_name in data_tables:
                combined_grid_data_factory.add_data_to_tables(
                    add_data=lv_grid_data_factory.data,
                    table_name=table_name
                )

        # call export functions
        combined_grid_data_factory.export_data_to_csv(output_path)
        scenario_data_factory = ScenarioDataFactory(combined_grid_data_factory.data.scenario_data)
        scenario_data_factory.export_data_to_csv(output_path)

        return combined_scenario_name

    def increase_der_penetration_of_scenario(
            self,
            scenario_name: str,
            path_to_der_data: str,
            assign_ders_from_list: bool = False,
            penetration_ratio: float = 1.0,
            new_scenario_name: str = None
    ) -> str:

        additional_der_data = self.load_csv_into_dataframe(path_to_der_data)

        if new_scenario_name is None:
            new_scenario_name = scenario_name + f'_increased_der_penetration_{str(penetration_ratio)}'

        new_scenario_name = new_scenario_name.replace('.', '_')  # to not have naming issues

        # Create output folder for the new grid
        output_path = self.create_output_folder(scenario_name=new_scenario_name)

        grid_data = fledge.data_interface.ElectricGridData(scenario_name)
        grid_data_factory = ElectricGridDataFactory(grid_data)

        # Loop through der data and add to the scenario
        # Only nodes with existing loads get assigned an additional load
        der_population = grid_data.electric_grid_ders['der_name'].to_list()
        num_of_ders = len(der_population)
        if assign_ders_from_list:
            # Assign every DER from additional_der_data to one existing DER node
            penetration_ratio = 0  # this will prevent the while loop from repeating
        for new_der_index, new_der_row in additional_der_data.iterrows():
            if len(der_population) == 0:
                break
            additional_der_name = str(new_der_row['der_name'])
            count = 1
            while True:
                # Randomly sample a DER from the der_population and remove it from population
                [der_name, der_population] = analysis.utils.Random.sample_and_remove(
                    population_list=der_population,
                    num_of_samples=1
                )
                der_name = der_name[0]  # sampling returns a list object
                node = grid_data.electric_grid_ders.loc[
                    grid_data.electric_grid_ders['der_name'] == der_name,
                    'node_name'
                ].to_list()[0]
                print(f'Adding DER {additional_der_name} to node {node} in scenario {new_scenario_name}.')
                new_der_row['node_name'] = str(node)
                new_der_row['der_name'] = str(der_name) + '_' + additional_der_name
                # Add the DER to the main DER table
                grid_data_factory.add_der_to_electric_grid_data(der_table_row=new_der_row)
                count += 1
                if (count > int(num_of_ders * penetration_ratio)) or (len(der_population) == 0):
                    break

        new_values_dict = {
            'scenario_name': new_scenario_name,
            'electric_grid_name': new_scenario_name,

        }
        for attribute in new_values_dict:
            grid_data_factory.change_attribute_value(
                new_value=new_values_dict[attribute],
                attribute_name=attribute,
            )

        # call export function
        grid_data_factory.export_data_to_csv(output_path)
        scenario_data_factory = ScenarioDataFactory(grid_data_factory.data.scenario_data)
        scenario_data_factory.export_data_to_csv(output_path)

        return new_scenario_name

    def add_price_type_scenario_to_scenario(
            self,
            scenario_name: str,
            price_type: str,
            new_scenario_name: str = None,
            price_sensitivity_coefficient: float = None
    ):
        scenario_data_factory = ScenarioDataFactory(fledge.data_interface.ScenarioData(scenario_name))
        if new_scenario_name is None:
            new_scenario_name = scenario_name + '_' + price_type

        key_value_pairs = {
            'price_type': price_type,
            'scenario_name': new_scenario_name
        }
        if price_sensitivity_coefficient is not None:
            key_value_pairs['price_sensitivity_coefficient'] = price_sensitivity_coefficient

        scenario_data_factory.add_row_with_new_values(
            key_value_pairs=key_value_pairs,
            table_name='scenario'
        )

        output_path = self.get_scenario_folder(scenario_name=scenario_name)
        scenario_data_factory.export_data_to_csv(
            output_path=output_path,
            table_name='scenario'
        )

    def generate_fixed_load_der_input_data(
            self,
            scenario_name: str,
            path_to_der_schedules_data: str,
            replace_ders: bool = True,
            node_names: list = None
    ):
        # per default, this function generates a load profile at every node with a DER (node_names = None)
        if self.der_schedules is None:
            self.der_schedules = self.load_csv_into_dataframe(path_to_der_schedules_data)

        # load scenario data to get the relevant time period and grid data (nodes)
        print(f'Adding DERs to scenario {scenario_name}')

        # Get scenario data
        scenario_data = fledge.data_interface.ScenarioData(scenario_name)
        # in der_schedules, there is a unique timeseries for every week of the year with the week number as
        # identifier and related to the season
        [season, weeknums] = self.calendar.get_season_of_scenario_data(scenario_data)

        grid_data = fledge.data_interface.ElectricGridData(scenario_name)
        grid_data_factory = ElectricGridDataFactory(grid_data)

        electric_grid_name = grid_data.electric_grid.electric_grid_name
        nodes = grid_data.electric_grid_nodes

        der_type = 'fixed_load'
        power_factor = 0.95
        source_node_name = grid_data.electric_grid['source_node_name']
        nodes = node_names if node_names is not None else nodes['node_name']
        for node_name in nodes:
            # Don't add a new der to the source node
            if node_name == source_node_name:
                continue

            der_index = grid_data.electric_grid_ders[grid_data.electric_grid_ders['node_name'] == node_name].index
            if len(der_index) == 0:
                # Only add DERs at nodes where a DER is already existent
                continue

            # Define the DER attributes
            # TODO this part should be a separate method (or from DERDataFactory)
            der_name = 'Load R' + node_name
            # TODO: loop over all phases
            der_model_name = (
                self.pick_random_consumer_type() + '_' + season + '_' + random.choice(weeknums) + '_phase_0'
            )
            # TODO use fledge's capacilities to calculate the mean over for timestep of 60 minutes
            active_power_vals = self.der_schedules.loc[self.der_schedules['definition_name'] == der_model_name, :]
            active_power_vals_aggregated = []
            for i in range(0, len(active_power_vals), 60):
                for val in scenario_data.timesteps.weekday.unique():
                    if (str(val) == active_power_vals[i:i + 60]['time_period'].str[1:2]).any():
                        active_power_vals_aggregated.append(np.mean(active_power_vals[i:i + 60]['value']))
            active_power_nominal = (-1) * max(active_power_vals_aggregated)
            # active_power_nominal = (-1) * max(active_power_vals['value'])
            reactive_power_nominal = active_power_nominal * np.tan(np.arccos(power_factor))

            key_value_pairs = {
                'electric_grid_name': electric_grid_name,
                'der_name': der_name,
                'der_type': der_type,
                'der_model_name': der_model_name,
                'node_name': node_name,
                'active_power_nominal': active_power_nominal,
                'reactive_power_nominal': reactive_power_nominal,
            }

            if replace_ders:
                # Get the DERs at the node and change their values
                print(f'Replacing DERs at node {node_name} with DER {der_name} of type {der_type}', end='\r')
                for attribute in key_value_pairs:
                    grid_data_factory.change_attribute_value(
                        new_value=key_value_pairs[attribute],
                        attribute_name=attribute,
                        row_index=der_index,
                        table_name='electric_grid_ders'
                    )
            else:
                print(f'Adding DER {der_name} of type {der_type} to node {node_name}', end='\r')
                # Copy DER and add a changed version to the end of the table
                # Add to electric grid ders dataframe
                grid_data_factory.add_row_with_new_values(
                    key_value_pairs=key_value_pairs,
                    table_name='electric_grid_ders'
                )

        grid_data_factory.export_data_to_csv(
            output_path=os.path.join(self.path_to_data, scenario_name),
            table_name='electric_grid_ders')

    def generate_der_models_csv_for_scenario(
            self,
            scenario_name: str,
            der_type: str = 'fixed_load',
            definition_type: str = 'schedule'
    ):
        # Using the last scenario_name of the loop (it does not matter)
        # der_models.csv is only generated once for all possible der_model_names
        # Drop / remove all data from dataframe (if exist)
        der_data = fledge.data_interface.DERData(scenario_name)
        der_data.ders.drop(der_data.ders.index, inplace=True)
        for definition_name in self.der_schedules['definition_name'].unique():
            # Add to ders data
            der_data.ders = der_data.ders.append({
                'der_model_name': definition_name,
                'der_type': der_type,
                'definition_name': definition_name,
                'definition_type': definition_type
            }, ignore_index=True)
        der_data_factory = DERDataFactory(der_data)
        der_data_factory.export_data_to_csv(output_path=os.path.join(self.path_to_data, scenario_name))

    @staticmethod
    def pick_random_consumer_type() -> str:
        # TODO: this must go somewhere else
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

    @staticmethod
    def reload_database():
        print('Reloading database...')
        # Recreate / overwrite database, to incorporate changes in the CSV files.
        fledge.data_interface.recreate_database()

    def create_output_folder(
            self,
            scenario_name: str
    ) -> str:
        # Create output folder for the new grid
        output_path = os.path.join(self.path_to_data, scenario_name)
        # Instantiate dedicated directory for current grid.
        shutil.rmtree(output_path, ignore_errors=True)
        os.mkdir(output_path)
        return output_path

    def get_scenario_folder(
            self,
            scenario_name: str
    ) -> str:
        print(f'Assuming dedicated folder for scenario {scenario_name}')
        return os.path.join(self.path_to_data, scenario_name)

    def load_csv_into_dataframe(
            self,
            path_to_csv_data: str,
    ) -> pd.DataFrame:
        print(f'Loading data from {path_to_csv_data}...')
        return pd.read_csv(os.path.join(os.path.dirname(self.path_to_data), path_to_csv_data))
