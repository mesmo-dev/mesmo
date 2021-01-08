"""
Module that helps plotting and analyzing the results of the simulation runs
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import fledge.problems
import fledge.plots
import fledge.config

logger = fledge.config.get_logger(__name__)


class AnalysisManager(object):
    results_path: str
    results_dict: dict

    def __init__(
            self,
            results_dict: dict,
            results_path: str = None
    ):
        self.results_path = results_path
        self.results_dict = results_dict

    def generate_result_plots(
            self,
            der_penetration: str
    ):

        power_flow_results = self.results_dict[der_penetration]['power_flow_results']
        optimization_results =  self.results_dict[der_penetration]['optimization_results']
        # price_timeseries =
        # system_costs = results_dict['system_costs']
        scenarios = self.get_list_of_scenario_strings(power_flow_results)
        electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenarios[0])
        scenario_data = fledge.data_interface.ScenarioData(scenarios[0])  # only used for timesteps
        plotter = Plots(
            electric_grid_model=electric_grid_model,
            scenario_data=scenario_data,
            results_path=self.results_path
        )
        # Plots over assets
        plots = {
            'Branch power (direction 1) magnitude [p.u.]': 'branch_power_magnitude_vector_1_per_unit',
            # 'Branch power (direction 2) magnitude [p.u.]': 'branch_power_magnitude_vector_2_per_unit',
            'Node voltage magnitude [p.u.]': 'node_voltage_magnitude_vector_per_unit',
            'DER active power magnitude [p.u.]': 'der_active_power_vector_per_unit',
        }
        plotter.plot_assets(plots, power_flow_results, scenarios)
        # Plots for system losses
        plots = {
            'System losses': 'loss_active',
            'Total system losses': 'loss_active',
        }
        plotter.plot_losses(plots, power_flow_results, scenarios)
        # Plots for system costs
        # plots = {
        #     'Objective value': 'opt_objective_',
        # }
        # plotter.plot_system_costs(plots)

        plots = {
            'Branch power (direction 1) magnitude [p.u.]': 'branch_power_magnitude_vector_1_per_unit',
            # 'Node voltage magnitude [p.u.]': 'node_voltage_magnitude_vector_per_unit',
        }
        plotter.plot_grid(plots, power_flow_results, scenarios)

        logger.info('Done.')

    @staticmethod
    def get_list_of_scenario_strings(
            pf_results_dict: dict
    ) -> list:
        scenarios = []
        for key in pf_results_dict.keys():
            scenarios.append(key)
        return scenarios


class Plots(object):
    results_path: str
    electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault
    scenario_data: fledge.data_interface.ScenarioData

    def __init__(
            self,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            scenario_data: fledge.data_interface.ScenarioData,
            results_path: str = None
    ):
        self.electric_grid_model = electric_grid_model
        self.scenario_data = scenario_data
        self.results_path = results_path

    # [x] branch power magnitude,
    # [x] node voltage magnitude
    # [ ] total losses
    # [ ] reactive losses
    # [x] der dispatch
    # [ ] objective result / system costs
    # [ ] prices over time
    # [ ] branch power magnitude over time per branch
    # [ ] node voltage magnitude over time per node
    # [ ] der output over time
    # [ ] dlmp validation
    # TODO: [ ] load flow calculation comparison --> ask Sebastian

    def plot_assets(
            self,
            plots: dict,
            power_flow_results: dict,
            scenarios: list
    ):

        for plot in plots:
            minimum = None
            maximum = None
            if 'Node voltage' in plot:
                assets = self.electric_grid_model.nodes
                # minimum = voltage_min
                # maximum = voltage_max
            elif 'Branch power' in plot:
                assets = self.electric_grid_model.branches
                minimum = 0.0
                maximum = 1.0
            elif 'DER active power' in plot:
                assets = self.electric_grid_model.ders
                minimum = 0.0
                maximum = 1.0
            else:
                return
            plot_types = {
                'time': self.scenario_data.timesteps,
                'asset': assets
            }
            for type in plot_types:
                for x_label in plot_types[type]:
                    plt.figure()
                    if type is 'time':
                        title_label = ' at: ' + x_label.strftime("%H-%M-%S")
                    else:
                        title_label = x_label[0] + ': ' + x_label[1]
                    plt.title(plot + ' ' + title_label)
                    marker_index = 4
                    for scenario_name in scenarios:
                        pf_results_scenario_name = [key for key in power_flow_results.keys() if scenario_name in key][0]
                        pf_results_of_scenario = power_flow_results[pf_results_scenario_name]
                        if type is 'time':
                            y_values = pf_results_of_scenario[plots[plot]].loc[x_label, :].reindex(assets).values
                        else:
                            y_values = pf_results_of_scenario[plots[plot]].loc[:, (x_label[0], x_label[1], slice(None))]
                        if type is 'time':
                            plt.scatter(
                                range(len(assets)),
                                y_values,
                                marker=marker_index,
                                label=scenario_name
                            )
                        else:
                            plt.plot(
                                self.scenario_data.timesteps,
                                y_values,
                                marker=marker_index,
                                label=scenario_name
                            )
                        marker_index += 1
                    if (minimum is not None) and (maximum is not None):
                        label_min = 'Minimum'
                        label_max = 'Maximum'
                        if type is 'time':
                            x_range = [range(len(assets))[0], range(len(assets))[-1]]
                        else:
                            x_range = [self.scenario_data.timesteps[0], self.scenario_data.timesteps[-1]]
                        plt.plot(x_range, [minimum, minimum], 'k-', color='r', label=label_min)
                        plt.plot(x_range, [maximum, maximum], 'k-', color='r', label=label_max)
                        # plt.ylim((minimum, maximum))
                    handles, labels = plt.gca().get_legend_handles_labels()
                    # sort both labels and handles by labels
                    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                    plt.legend(handles, labels)
                    if type is 'time':
                        plt.xticks(
                            range(len(assets)),
                            assets,
                            rotation=45,
                            ha='right'
                        )
                    plt.grid()
                    plt.tight_layout()
                    timestep = x_label if type is 'time' else None
                    file_name = plot + '_' + x_label[1] if type is 'asset' else plot
                    self.__save_or_show_plot(file_name, timestep=timestep)

    def plot_losses(
            self,
            plots: dict,
            power_flow_results: dict,
            scenarios: list
    ):
        for plot in plots:
            if 'Total' in plot:
                x_index = ['total']
            elif 'System losses' in plot:
                x_index = self.scenario_data.timesteps
            else:
                return
            plt.title(plot)
            marker_index = 4
            for scenario_name in scenarios:
                pf_results_scenario_name = [key for key in power_flow_results if scenario_name in key][0]
                pf_results_of_scenario = power_flow_results[pf_results_scenario_name]
                if 'Total' in plot:
                    y_values = sum(pf_results_of_scenario[plots[plot]].values)
                else:
                    y_values = pf_results_of_scenario[plots[plot]].values
                plt.scatter(
                    range(len(x_index)),
                    y_values,
                    marker=marker_index
                )
                marker_index += 1
            plt.xticks(
                range(len(x_index)),
                x_index,
                rotation=45,
                ha='right'
            )
            plt.legend(scenarios)
            plt.grid()
            plt.tight_layout()
            self.__save_or_show_plot(plot)

    def plot_grid(
            self,
            plots: dict,
            power_flow_results: dict,
            scenarios: list
    ):

        electric_grid_graph = fledge.plots.ElectricGridGraph(self.scenario_data.scenario.scenario_name)

        for plot in plots:
            for scenario_name in scenarios:
                pf_results_scenario_name = [key for key in power_flow_results.keys() if scenario_name in key][0]
                pf_results_of_scenario = power_flow_results[pf_results_scenario_name]
                if 'Node voltage' in plot:
                    # Plot electric grid nodes voltage drop.
                    fledge.plots.plot_grid_node_utilization(
                        self.electric_grid_model,
                        electric_grid_graph,
                        pf_results_of_scenario[plots[plot]],
                        self.results_path,
                        value_unit='p.u.',
                        # vmin=,
                        # vmax=,
                    )
                elif 'Branch power' in plot:
                    # Plot electric grid line utilization.
                    fledge.plots.plot_grid_line_utilization(
                        self.electric_grid_model,
                        electric_grid_graph,
                        pf_results_of_scenario[plots[plot]],
                        self.results_path,
                        value_unit='p.u.',
                        # vmin=,
                        # vmax=,
                    )
                else:
                    logger.info(f'Plot with name "{plot}" not supported.')
                    return

    def plot_grid_dlmps(
            self
    ):
        pass
        # # Plot electric grid DLMPs in grid.
        # dlmp_types = [
        #     'electric_grid_energy_dlmp_node_active_power',
        #     'electric_grid_voltage_dlmp_node_active_power',
        #     'electric_grid_congestion_dlmp_node_active_power',
        #     'electric_grid_loss_dlmp_node_active_power'
        # ]
        # for timestep in self.scenario_data.timesteps:
        #     for dlmp_type in dlmp_types:
        #         node_color = (
        #                 dlmps[dlmp_type].loc[timestep, :].groupby('node_name').mean().reindex(
        #                     electric_grid_graph.nodes).values
        #                 * 1.0e3
        #         )
        #         plt.title(
        #             f"{dlmp_type.replace('_', ' ').capitalize().replace('dlmp', 'DLMP')}"
        #             f" at {timestep.strftime('%H:%M:%S')}"
        #         )
        #         nx.draw(
        #             electric_grid_graph,
        #             pos=electric_grid_graph.node_positions,
        #             nodelist=(
        #                 electric_grid_model.nodes[
        #                     fledge.utils.get_index(electric_grid_model.nodes, node_type='source')
        #                 ].get_level_values('node_name')[:1].to_list()
        #             ),
        #             edgelist=[],
        #             node_size=150.0,
        #             node_color='red'
        #         )
        #         nx.draw(
        #             electric_grid_graph,
        #             pos=electric_grid_graph.node_positions,
        #             arrows=False,
        #             node_size=100.0,
        #             node_color=node_color,
        #             edgecolors='black',  # Make node border visible.
        #             with_labels=False
        #         )
        #         sm = (
        #             plt.cm.ScalarMappable(
        #                 norm=plt.Normalize(
        #                     vmin=np.min(node_color),
        #                     vmax=np.max(node_color)
        #                 )
        #             )
        #         )
        #         cb = plt.colorbar(sm, shrink=0.9)
        #         cb.set_label('Price [S$/MWh]')
        #         plt.tight_layout()
        #         plt.savefig(os.path.join(results_path, f'{dlmp_type}_{timestep.strftime("%H-%M-%S")}.png'))
        #         # plt.show()
        #         plt.close()

    def __save_or_show_plot(
            self,
            plot_name: str,
            timestep: pd.Timestamp = None
    ):
        if self.results_path is None:
            plt.show()
        else:
            if timestep is None:
                file_name = f'{plot_name}.png'
            else:
                file_name = f'{plot_name}_{timestep.strftime("%Y-%m-%d_%H-%M-%S")}.png'
            file_path = os.path.join(self.results_path, file_name)
            plt.savefig(file_path)
            print(f'Saved plot: {file_name}', end="\r")
        plt.close()
