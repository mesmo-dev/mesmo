"""Project SITEM baseline scenario evaluation script.

- This script relies on the 'ema_sample_grid' scenario which is not included in this repository. If you have the
  scenario definition files, add the path to the definition in `config.yml` at `additional_data_paths=[]`.
- This script depends on `contextily`, which is not included in the package dependencies, but can be installed
  under Anaconda via `conda install -c conda-forge contextily`.
"""

import contextily as ctx  # TODO: Document contextily dependency.
import cv2  # TODO: Document opencv dependency.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import re

import fledge.data_interface
import fledge.plots
import fledge.problems
import fledge.utils


def main():

    # Settings.
    scenario_name = 'ema_sample_grid'
    results_path = fledge.utils.get_results_path('run_sitem_baseline', scenario_name)
    plot_detailed_grid = True

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain nominal operation problem & solution.
    problem = fledge.problems.NominalOperationProblem(scenario_name)
    problem.solve()
    results = problem.get_results()

    # Obtain additional results.
    branch_power_vector_magnitude_relative = (
        (np.abs(results['branch_power_vector_1']) + np.abs(results['branch_power_vector_2'])) / 2
        / problem.electric_grid_model.branch_power_vector_magnitude_reference
    )
    branch_power_vector_magnitude_relative.loc['maximum', :] = branch_power_vector_magnitude_relative.max(axis='rows')
    node_voltage_vector_magnitude_per_unit = (
        np.abs(results['node_voltage_vector'])
        / np.abs(problem.electric_grid_model.node_voltage_vector_reference)
    )
    node_voltage_vector_magnitude_per_unit.loc['maximum', :] = node_voltage_vector_magnitude_per_unit.max(axis='rows')
    node_voltage_vector_magnitude_per_unit.loc['minimum', :] = node_voltage_vector_magnitude_per_unit.min(axis='rows')
    results.update({
        'branch_power_vector_magnitude_relative': branch_power_vector_magnitude_relative,
        'node_voltage_vector_magnitude_per_unit': node_voltage_vector_magnitude_per_unit
    })

    # Print results.
    print(results)

    # Store results to CSV.
    results.to_csv(results_path)

    # Obtain electric grid data / graph.
    electric_grid_data = fledge.data_interface.ElectricGridData(scenario_name)
    electric_grid_graph = fledge.plots.ElectricGridGraph(scenario_name)
    transformers = (
        problem.electric_grid_model.branches[
            fledge.utils.get_index(problem.electric_grid_model.branches, branch_type='transformer')
        ]
    )
    lines_by_edges = electric_grid_data.electric_grid_lines.loc[:, 'line_name']
    lines_by_edges.index = (
        electric_grid_data.electric_grid_lines.loc[:, ['node_1_name', 'node_2_name']].itertuples(index=False)
    )
    lines_by_edges = lines_by_edges.reindex(electric_grid_graph.edges)
    lines = (
        problem.electric_grid_model.branches[
            fledge.utils.get_index(
                problem.electric_grid_model.branches,
                branch_type='line'
            )
        ]
    )
    lines = (
        pd.MultiIndex.from_tuples(
            pd.Series(lines.tolist(), index=lines.get_level_values('branch_name')).reindex(lines_by_edges).tolist()
        )
    )
    nodes = (
        problem.electric_grid_model.nodes[
            problem.electric_grid_model.nodes.get_level_values('node_name').isin(electric_grid_graph.nodes)
        ]
    )

    # Obtain substation nodes / utilization.
    nodes_substation = (
        electric_grid_data.electric_grid_transformers.loc[
            transformers.get_level_values('branch_name'),
            'node_2_name'
        ].to_list()
    )
    node_utilization = 100.0 * branch_power_vector_magnitude_relative.loc[:, transformers]
    node_utilization.columns = nodes_substation

    # Plot electric grid substation utilization as nodes.
    for timestep in node_utilization.index:
        vmin = 20.0
        vmax = 120.0
        plt.figure(
            figsize=[12.0, 6.0],  # Arbitrary convenient figure size.
            dpi=300
        )
        plt.title(
            f"Substation utilization: {timestep.strftime('%H:%M:%S') if type(timestep) is pd.Timestamp else timestep}"
        )
        nx.draw(
            electric_grid_graph,
            nodelist=nodes_substation,
            edgelist=[],
            pos=electric_grid_graph.node_positions,
            node_size=200.0,
            node_color=node_utilization.loc[timestep, :].tolist(),
            vmin=vmin,
            vmax=vmax,
            edgecolors='black',
            # Uncomment below to print utilization as node labels.
            labels=node_utilization.loc[timestep, :].round().astype(np.int).to_dict(),
            font_size=7.0,
            font_color='white',
            font_family='Arial'
        )
        # Adjust axis limits, to get a better view of surrounding map.
        xlim = plt.xlim()
        xlim = (xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0]))
        plt.xlim(xlim)
        ylim = plt.ylim()
        ylim = (ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))
        plt.ylim(ylim)
        # Add colorbar.
        sm = (
            plt.cm.ScalarMappable(
                norm=plt.Normalize(
                    vmin=vmin,
                    vmax=vmax
                )
            )
        )
        cb = plt.colorbar(sm, shrink=0.9)
        cb.set_label('Utilization [%]')
        # Add basemap / open street map for better orientation.
        # ctx.add_basemap(
        #     plt.gca(),
        #     crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
        #     source=ctx.providers.CartoDB.Positron,
        #     attribution=False  # Do not show copyright notice.
        # )
        name_string = re.sub(r'\W+', '-', f'{timestep}')
        plt.savefig(os.path.join(results_path, f'substation_utilization_{name_string}.png'), bbox_inches='tight')
        # plt.show()
        plt.close()

    # Stitch images to video.
    images = []
    for timestep in problem.timesteps:
        name_string = re.sub(r'\W+', '-', f'{timestep}')
        images.append(cv2.imread(os.path.join(results_path, f'substation_utilization_{name_string}.png')))
    video_writer = (
        cv2.VideoWriter(
            os.path.join(results_path, 'substation_utilization.avi'),  # Filename.
            cv2.VideoWriter_fourcc(*'XVID'),  # Format.
            2.0,  # FPS.
            images[0].shape[1::-1]  # Size.
        )
    )
    for image in images:
        video_writer.write(image)
    video_writer.release()
    cv2.destroyAllWindows()

    # Plot electric line loading.
    if plot_detailed_grid:
        for timestep in problem.timesteps:
            vmin = 20.0
            vmax = 120.0
            plt.figure(
                figsize=[12.0, 6.0],  # Arbitrary convenient figure size.
                dpi=300
            )
            plt.title(
                f"Line utilization: {timestep.strftime('%H:%M:%S') if type(timestep) is pd.Timestamp else timestep}"
            )
            nx.draw(
                electric_grid_graph,
                nodelist=nodes_substation,
                edgelist=[],
                pos=electric_grid_graph.node_positions,
                node_size=100.0,
                node_color='red'
            )
            nx.draw(
                electric_grid_graph,
                # nodelist=[],
                # edgelist=[],
                pos=electric_grid_graph.node_positions,
                node_size=10.0,
                node_color='black',
                arrows=False,
                width=5.0,
                edge_vmin=vmin,
                edge_vmax=vmax,
                edge_color=(100.0 * branch_power_vector_magnitude_relative.loc[timestep, lines]).tolist(),
            )
            # Adjust axis limits, to get a better view of surrounding map.
            xlim = plt.xlim()
            xlim = (xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0]))
            plt.xlim(xlim)
            ylim = plt.ylim()
            ylim = (ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))
            plt.ylim(ylim)
            # Add colorbar.
            sm = (
                plt.cm.ScalarMappable(
                    norm=plt.Normalize(
                        vmin=vmin,
                        vmax=vmax
                    )
                )
            )
            cb = plt.colorbar(sm, shrink=0.9)
            cb.set_label('Utilization [%]')
            # Add basemap / open street map for better orientation.
            ctx.add_basemap(
                plt.gca(),
                crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
                source=ctx.providers.CartoDB.Positron,
                attribution=False  # Do not show copyright notice.
            )
            name_string = re.sub(r'\W+', '-', f'{timestep}')
            plt.savefig(os.path.join(results_path, f'line_utilization_{name_string}.png'), bbox_inches='tight')
            # plt.show()
            plt.close()

        # Stitch images to video.
        images = []
        for timestep in problem.timesteps:
            name_string = re.sub(r'\W+', '-', f'{timestep}')
            images.append(cv2.imread(os.path.join(results_path, f'line_utilization_{name_string}.png')))
        video_writer = (
            cv2.VideoWriter(
                os.path.join(results_path, 'line_utilization.avi'),  # Filename.
                cv2.VideoWriter_fourcc(*'XVID'),  # Format.
                2.0,  # FPS.
                images[0].shape[1::-1]  # Size.
            )
        )
        for image in images:
            video_writer.write(image)
        video_writer.release()
        cv2.destroyAllWindows()

    # Plot electric line loading.
    if plot_detailed_grid:
        for timestep in problem.timesteps:
            vmin = 0.0
            vmax = 10.0
            plt.figure(
                figsize=[12.0, 6.0],  # Arbitrary convenient figure size.
                dpi=300
            )
            plt.title(
                f"Node voltage drop: {timestep.strftime('%H:%M:%S') if type(timestep) is pd.Timestamp else timestep}"
            )
            nx.draw(
                electric_grid_graph,
                nodelist=nodes_substation,
                edgelist=[],
                pos=electric_grid_graph.node_positions,
                node_size=100.0,
                node_color='red'
            )
            nx.draw(
                electric_grid_graph,
                nodelist=nodes.get_level_values('node_name').tolist(),
                # edgelist=[],
                pos=electric_grid_graph.node_positions,
                node_size=50.0,
                arrows=False,
                # width=5.0,
                vmin=vmin,
                vmax=vmax,
                node_color=(-100.0 * (node_voltage_vector_magnitude_per_unit.loc[timestep, nodes] - 1.0)).tolist(),
                edgecolors='black',
            )
            # Adjust axis limits, to get a better view of surrounding map.
            xlim = plt.xlim()
            xlim = (xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0]))
            plt.xlim(xlim)
            ylim = plt.ylim()
            ylim = (ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))
            plt.ylim(ylim)
            # Add colorbar.
            sm = (
                plt.cm.ScalarMappable(
                    norm=plt.Normalize(
                        vmin=vmin,
                        vmax=vmax
                    )
                )
            )
            cb = plt.colorbar(sm, shrink=0.9)
            cb.set_label('Voltage drop [%]')
            # Add basemap / open street map for better orientation.
            ctx.add_basemap(
                plt.gca(),
                crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
                source=ctx.providers.CartoDB.Positron,
                attribution=False  # Do not show copyright notice.
            )
            name_string = re.sub(r'\W+', '-', f'{timestep}')
            plt.savefig(os.path.join(results_path, f'node_voltage_{name_string}.png'), bbox_inches='tight')
            # plt.show()
            plt.close()

        # Stitch images to video.
        images = []
        for timestep in problem.timesteps:
            name_string = re.sub(r'\W+', '-', f'{timestep}')
            images.append(cv2.imread(os.path.join(results_path, f'node_voltage_{name_string}.png')))
        video_writer = (
            cv2.VideoWriter(
                os.path.join(results_path, 'node_voltage.avi'),  # Filename.
                cv2.VideoWriter_fourcc(*'XVID'),  # Format.
                2.0,  # FPS.
                images[0].shape[1::-1]  # Size.
            )
        )
        for image in images:
            video_writer.write(image)
        video_writer.release()
        cv2.destroyAllWindows()

    # Plot some results.
    plt.title('Branch utilization [%]')
    plt.bar(
        range(len(problem.electric_grid_model.branches)),
        100.0 * branch_power_vector_magnitude_relative.loc['maximum', :]
    )
    plt.hlines(100.0, -0.5, len(problem.electric_grid_model.branches) - 0.5, colors='red')
    plt.xticks(
        range(len(problem.electric_grid_model.branches)),
        problem.electric_grid_model.branches,
        rotation=45,
        ha='right'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    # plt.show()

    plt.title('Transformer utilization [%]')
    transformers = (
        problem.electric_grid_model.branches[
            fledge.utils.get_index(problem.electric_grid_model.branches, branch_type='transformer')
        ]
    )
    plt.bar(
        range(len(transformers)),
        100.0 * branch_power_vector_magnitude_relative.loc['maximum', transformers]
    )
    plt.hlines(100.0, -0.5, len(transformers) - 0.5, colors='red')
    plt.xticks(
        range(len(transformers)),
        transformers,
        rotation=45,
        ha='right'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    # plt.show()

    plt.title('Maximum voltage drop [%]')
    plt.bar(
        range(len(problem.electric_grid_model.nodes)),
        100.0 * (node_voltage_vector_magnitude_per_unit.loc['minimum', :] - 1.0)
    )
    plt.hlines(-5.0, -0.5, len(problem.electric_grid_model.nodes) - 0.5, colors='red')
    plt.xticks(
        range(len(problem.electric_grid_model.nodes)),
        problem.electric_grid_model.nodes,
        rotation=45,
        ha='right'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    # plt.show()

    # Print results path.
    os.startfile(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
