"""Project SITEM scenario evaluation script.

- This script depends on `contextily`, which is not included in the package dependencies, but can be installed
  under Anaconda via `conda install -c conda-forge contextily`.
"""

import contextily as ctx  # TODO: Document contextily dependency.
import cv2
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
    scenario_name = 'singapore_district25'
    results_path = fledge.utils.get_results_path('run_sitem_baseline', scenario_name)
    plot_detailed_grid = True

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain nominal operation problem & solution.
    problem = fledge.problems.NominalOperationProblem(scenario_name)
    problem.solve()
    results = problem.get_results()

    # Obtain additional results.
    branch_power_vector_magnitude_per_unit = (
        (np.abs(results['branch_power_vector_1']) + np.abs(results['branch_power_vector_2'])) / 2
        / problem.electric_grid_model.branch_power_vector_magnitude_reference
    )
    branch_power_vector_magnitude_per_unit.loc['maximum', :] = branch_power_vector_magnitude_per_unit.max(axis='rows')
    node_voltage_vector_magnitude_per_unit = (
        np.abs(results['node_voltage_vector'])
        / np.abs(problem.electric_grid_model.node_voltage_vector_reference)
    )
    node_voltage_vector_magnitude_per_unit.loc['maximum', :] = node_voltage_vector_magnitude_per_unit.max(axis='rows')
    node_voltage_vector_magnitude_per_unit.loc['minimum', :] = node_voltage_vector_magnitude_per_unit.min(axis='rows')
    results.update({
        'branch_power_vector_magnitude_per_unit': branch_power_vector_magnitude_per_unit,
        'node_voltage_vector_magnitude_per_unit': node_voltage_vector_magnitude_per_unit
    })

    # Print results.
    print(results)

    # Store results to CSV.
    results.to_csv(results_path)

    # Obtain electric grid graph.
    electric_grid_graph = fledge.plots.ElectricGridGraph(scenario_name)

    # Obtain element indexes for graph plots.
    # TODO: Consider including these within the graph object.
    transformer_nodes = (
        problem.electric_grid_model.nodes[
            np.array(np.nonzero(
                problem.electric_grid_model.branch_incidence_2_matrix[
                    fledge.utils.get_index(problem.electric_grid_model.branches, branch_type='transformer'),
                    :
                ] > 0
            ))[:, 1]
        ]
    )

    # Plot electric grid transformer utilization.
    for timestep in branch_power_vector_magnitude_per_unit.index:
        vmin = 20.0
        vmax = 120.0
        plt.figure(
            figsize=[12.0, 6.0],  # Arbitrary convenient figure size.
            dpi=300
        )
        plt.title(
            f"Substation utilization: {timestep.strftime('%H:%M:%S') if type(timestep) is pd.Timestamp else timestep}"
        )
        # Plot nodes all nodes, but with node size 0.0, just to get appropriate map extent.
        nx.draw(
            electric_grid_graph,
            edgelist=[],
            pos=electric_grid_graph.node_positions,
            node_size=0.0
        )
        nx.draw(
            electric_grid_graph,
            nodelist=transformer_nodes.get_level_values('node_name').tolist(),
            edgelist=[],
            pos=electric_grid_graph.node_positions,
            node_size=200.0,
            node_color=(100.0 * branch_power_vector_magnitude_per_unit.loc[timestep, problem.electric_grid_model.transformers]).tolist(),
            vmin=vmin,
            vmax=vmax,
            edgecolors='black',
            # Uncomment below to print utilization as node labels.
            # labels=node_utilization.loc[timestep, :].round().astype(np.int).to_dict(),
            # font_size=7.0,
            # font_color='white',
            # font_family='Arial'
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
        plt.savefig(os.path.join(results_path, f'transformer_utilization_{name_string}.png'), bbox_inches='tight')
        # plt.show()
        plt.close()

    # Stitch images to video.
    images = []
    for timestep in problem.timesteps:
        name_string = re.sub(r'\W+', '-', f'{timestep}')
        images.append(cv2.imread(os.path.join(results_path, f'transformer_utilization_{name_string}.png')))
    video_writer = (
        cv2.VideoWriter(
            os.path.join(results_path, 'transformer_utilization.avi'),  # Filename.
            cv2.VideoWriter_fourcc(*'XVID'),  # Format.
            2.0,  # FPS.
            images[0].shape[1::-1]  # Size.
        )
    )
    for image in images:
        video_writer.write(image)
    video_writer.release()
    cv2.destroyAllWindows()

    # Plot electric grid line utilization.
    if plot_detailed_grid:
        for timestep in branch_power_vector_magnitude_per_unit.index:
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
                nodelist=transformer_nodes.get_level_values('node_name').tolist(),
                edgelist=[],
                pos=electric_grid_graph.node_positions,
                node_size=100.0,
                node_color='red'
            )
            nx.draw(
                electric_grid_graph,
                edgelist=electric_grid_graph.edge_by_line_name.loc[problem.electric_grid_model.lines.get_level_values('branch_name')].tolist(),
                pos=electric_grid_graph.node_positions,
                node_size=10.0,
                node_color='black',
                arrows=False,
                width=5.0,
                edge_vmin=vmin,
                edge_vmax=vmax,
                edge_color=(100.0 * branch_power_vector_magnitude_per_unit.loc[timestep, problem.electric_grid_model.lines]).tolist(),
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

    # Plot electric grid nodes voltage drop.
    if plot_detailed_grid:
        for timestep in node_voltage_vector_magnitude_per_unit.index:
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
                nodelist=transformer_nodes.get_level_values('node_name').tolist(),
                edgelist=[],
                pos=electric_grid_graph.node_positions,
                node_size=100.0,
                node_color='red'
            )
            nx.draw(
                electric_grid_graph,
                nodelist=problem.electric_grid_model.nodes.get_level_values('node_name').tolist(),
                pos=electric_grid_graph.node_positions,
                node_size=50.0,
                arrows=False,
                vmin=vmin,
                vmax=vmax,
                node_color=(-100.0 * (node_voltage_vector_magnitude_per_unit.loc[timestep, :] - 1.0)).tolist(),
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
            plt.savefig(os.path.join(results_path, f'node_voltage_drop_{name_string}.png'), bbox_inches='tight')
            # plt.show()
            plt.close()

        # Stitch images to video.
        images = []
        for timestep in problem.timesteps:
            name_string = re.sub(r'\W+', '-', f'{timestep}')
            images.append(cv2.imread(os.path.join(results_path, f'node_voltage_drop_{name_string}.png')))
        video_writer = (
            cv2.VideoWriter(
                os.path.join(results_path, 'node_voltage_drop.avi'),  # Filename.
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
        100.0 * branch_power_vector_magnitude_per_unit.loc['maximum', :]
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
    plt.close()

    plt.title('Transformer utilization [%]')
    problem.electric_grid_model.transformers = (
        problem.electric_grid_model.branches[
            fledge.utils.get_index(problem.electric_grid_model.branches, branch_type='transformer')
        ]
    )
    plt.bar(
        range(len(problem.electric_grid_model.transformers)),
        100.0 * branch_power_vector_magnitude_per_unit.loc['maximum', problem.electric_grid_model.transformers]
    )
    plt.hlines(100.0, -0.5, len(problem.electric_grid_model.transformers) - 0.5, colors='red')
    plt.xticks(
        range(len(problem.electric_grid_model.transformers)),
        problem.electric_grid_model.transformers,
        rotation=45,
        ha='right'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{plt.gca().get_title()}.png'))
    # plt.show()
    plt.close()

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
    plt.close()

    # Print results path.
    os.startfile(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
