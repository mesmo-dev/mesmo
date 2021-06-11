"""Example script for setting up and solving an single step electric grid power flow problem."""

import cvxpy as cp
import itertools
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge

def main():

    # Settings.
    scenario_name = 'test_2node'
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain scenario data.
    # - This contains general information, e.g. the base power values which are needed for the plots below.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)

    # Obtain electric grid model.
    # - The ElectricGridModelDefault object defines index sets for node names / branch names / der names / phases /
    #   node types / branch types, the nodal admittance / transformation matrices, branch admittance /
    #   incidence matrices, DER incidence matrices and no load voltage vector as well as nominal power vector.
    # - The model is created for the electric grid which is defined for the given scenario in the data directory.
    electric_grid_data = fledge.data_interface.ElectricGridData(scenario_name)
    electric_grid_data = fledge.electric_grid_models.ElectricGridModel.process_line_types_overhead(electric_grid_data)
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(electric_grid_data)

    # Obtain the nominal DER power vector.
    der_power_vector_nominal = electric_grid_model.der_power_vector_reference

    # Obtain power flow solution.
    # - The power flow solution object obtains the solution for nodal voltage vector / branch power vector
    #   and total loss (all complex valued) for the given DER power vector.
    # - There are different power flow solution objects depending on the solution algorithm / method:
    #   `PowerFlowSolutionFixedPoint`, `PowerFlowSolutionZBus`.
    #   (`PowerFlowSolutionOpenDSS` requires a `ElectricGridModelOpenDSS` instead of `ElectricGridModelDefault`.)
    power_flow_solution = (
        fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model, der_power_vector_nominal)
    )

    ####################################################################################################################
    # Update line parameters.
    ####################################################################################################################

    # Process over-head line type definitions.
    for line_type, line_type_data in electric_grid_data.electric_grid_line_types_overhead.iterrows():

        # Obtain data shorthands.
        # - Only for phases which have `conductor_id` defined in `electric_grid_line_types_overhead`.
        phases = (
            pd.Index([
                1 if pd.notnull(line_type_data.at['phase_1_conductor_id']) else None,
                2 if pd.notnull(line_type_data.at['phase_2_conductor_id']) else None,
                3 if pd.notnull(line_type_data.at['phase_3_conductor_id']) else None,
                'n' if pd.notnull(line_type_data.at['neutral_conductor_id']) else None
            ]).dropna()
        )
        phase_conductor_id = (
            pd.Series({
                1: line_type_data.at['phase_1_conductor_id'],
                2: line_type_data.at['phase_2_conductor_id'],
                3: line_type_data.at['phase_3_conductor_id'],
                'n': line_type_data.at['neutral_conductor_id']
            }).loc[phases]
        )
        phase_y = (
            pd.Series({
                1: line_type_data.at['phase_1_y'],
                2: line_type_data.at['phase_2_y'],
                3: line_type_data.at['phase_3_y'],
                'n': line_type_data.at['neutral_y']
            }).loc[phases]
        )
        phase_xy = (
            pd.Series({
                1: np.array([line_type_data.at['phase_1_x'], line_type_data.at['phase_1_y']]),
                2: np.array([line_type_data.at['phase_2_x'], line_type_data.at['phase_2_y']]),
                3: np.array([line_type_data.at['phase_3_x'], line_type_data.at['phase_3_y']]),
                'n': np.array([line_type_data.at['neutral_x'], line_type_data.at['neutral_y']])
            }).loc[phases]
        )
        phase_conductor_diameter = (
            pd.Series([
                electric_grid_data.electric_grid_line_types_overhead_conductors.at[
                    phase_conductor_id.at[phase], 'conductor_diameter'
                ]
                for phase in phases
            ], index=phases)
            * 1e-3  # mm to m.
        )
        phase_conductor_geometric_mean_radius = (
            pd.Series([
                electric_grid_data.electric_grid_line_types_overhead_conductors.at[
                    phase_conductor_id.at[phase], 'conductor_geometric_mean_radius'
                ]
                for phase in phases
            ], index=phases)
            * 1e-3  # mm to m.
        )
        phase_conductor_resistance = (
            pd.Series([
                electric_grid_data.electric_grid_line_types_overhead_conductors.at[
                    phase_conductor_id.at[phase], 'conductor_resistance'
                ]
                for phase in phases
            ], index=phases)
        )
        phase_conductor_maximum_current = (
            pd.Series([
                electric_grid_data.electric_grid_line_types_overhead_conductors.at[
                    phase_conductor_id.at[phase], 'conductor_maximum_current'
                ]
                for phase in phases
            ], index=phases)
        )

        # Obtain shorthands for neutral / non-neutral phases.
        # - This is needed for Kron reduction.
        phases_neutral = phases[phases.isin(['n'])]
        phases_non_neutral = phases[~phases.isin(['n'])]

        # Other parameter shorthands.
        frequency = electric_grid_data.electric_grid.at['base_frequency']  # In Hz.
        earth_resistivity = line_type_data.at['earth_resistivity']  # In Ωm.
        air_permittivity = line_type_data.at['air_permittivity']  # In nF/km.
        g_factor = 1e-4  # In Ω/km from 0.1609347e-3 Ω/mile from Kersting <https://doi.org/10.1201/9781315120782>.

        # Obtain impedance matrix in Ω/km based on Kersting <https://doi.org/10.1201/9781315120782>.
        z_matrix = pd.DataFrame(index=phases, columns=phases, dtype=complex)
        for phase_row, phase_col in itertools.product(phases, phases):
            # Calculate geometric parameters.
            d_distance = np.linalg.norm(phase_xy.at[phase_row] - phase_xy.at[phase_col])
            s_distance = np.linalg.norm(phase_xy.at[phase_row] - np.array([1, -1]) * phase_xy.at[phase_col])
            s_angle = np.pi / 2 - np.arcsin((phase_y.at[phase_row] + phase_y.at[phase_col]) / s_distance)
            # Calculate Kersting / Carson parameters.
            k_factor = (
                8.565e-4 * s_distance * np.sqrt(frequency / earth_resistivity)
            )
            p_factor = (
                np.pi / 8
                - (3 * np.sqrt(2)) ** -1 * k_factor * np.cos(s_angle)
                - k_factor ** 2 / 16 * np.cos(2 * s_angle) * (0.6728 + np.log(2 / k_factor))
            )
            q_factor = (
                -0.0386
                + 0.5 * np.log(2 / k_factor)
                + (3 * np.sqrt(2)) ** -1 * k_factor * np.cos(2 * s_angle)
            )
            x_factor = (
                2 * np.pi * frequency * g_factor
                * np.log(
                    phase_conductor_diameter[phase_row]
                    / phase_conductor_geometric_mean_radius.at[phase_row]
                )
            )
            # Calculate admittance according to Kersting / Carson <https://doi.org/10.1201/9781315120782>.
            if phase_row == phase_col:
                z_matrix.at[phase_row, phase_col] = (
                    phase_conductor_resistance.at[phase_row]
                    + 4 * np.pi * frequency * p_factor * g_factor
                    + 1j * (
                        x_factor
                        + 2 * np.pi * frequency * g_factor
                        * np.log(s_distance / phase_conductor_diameter[phase_row])
                        + 4 * np.pi * frequency * q_factor * g_factor
                    )
                )
            else:
                z_matrix.at[phase_row, phase_col] = (
                    4 * np.pi * frequency * p_factor * g_factor
                    + 1j * (
                        2 * np.pi * frequency * g_factor
                        * np.log(s_distance / d_distance)
                        + 4 * np.pi * frequency * q_factor * g_factor
                    )
                )

        # Apply Kron reduction.
        z_matrix = (
            pd.DataFrame(
                (
                    z_matrix.loc[phases_non_neutral, phases_non_neutral].values
                    - z_matrix.loc[phases_non_neutral, phases_neutral].values
                    @ z_matrix.loc[phases_neutral, phases_neutral].values ** -1  # Inverse of scalar value.
                    @ z_matrix.loc[phases_neutral, phases_non_neutral].values
                ),
                index=phases_non_neutral,
                columns=phases_non_neutral
            )
        )

        # Obtain potentials matrix in km/nF based on Kersting <https://doi.org/10.1201/9781315120782>.
        p_matrix = pd.DataFrame(index=phases, columns=phases, dtype=float)
        for phase_row, phase_col in itertools.product(phases, phases):
            # Calculate geometric parameters.
            d_distance = np.linalg.norm(phase_xy.at[phase_row] - phase_xy.at[phase_col])
            s_distance = np.linalg.norm(phase_xy.at[phase_row] - np.array([1, -1]) * phase_xy.at[phase_col])
            # Calculate potential according to Kersting <https://doi.org/10.1201/9781315120782>.
            if phase_row == phase_col:
                p_matrix.at[phase_row, phase_col] = (
                    1 / (2 * np.pi * air_permittivity)
                    * np.log(s_distance / phase_conductor_diameter.at[phase_row])
                )
            else:
                p_matrix.at[phase_row, phase_col] = (
                    1 / (2 * np.pi * air_permittivity)
                    * np.log(s_distance / d_distance)
                )

        # Apply Kron reduction.
        p_matrix = (
            pd.DataFrame(
                (
                    p_matrix.loc[phases_non_neutral, phases_non_neutral].values
                    - p_matrix.loc[phases_non_neutral, phases_neutral].values
                    @ p_matrix.loc[phases_neutral, phases_neutral].values ** -1  # Inverse of scalar value.
                    @ p_matrix.loc[phases_neutral, phases_non_neutral].values
                ),
                index=phases_non_neutral,
                columns=phases_non_neutral
            )
        )

        # Obtain capacitance matrix in nF/km.
        c_matrix = pd.DataFrame(np.linalg.inv(p_matrix), index=phases_non_neutral, columns=phases_non_neutral)

        # Obtain final element matrices.
        resistance_matrix = z_matrix.apply(np.real)  # In Ω/km.
        reactance_matrix = z_matrix.apply(np.imag)  # In Ω/km.
        capacitance_matrix = c_matrix  # In nF/km.

        # Add to line type matrices definition.
        for phase_row in phases_non_neutral:
            for phase_col in phases_non_neutral[phases_non_neutral <= phase_row]:
                electric_grid_data.electric_grid_line_types_matrices = (
                    electric_grid_data.electric_grid_line_types_matrices.append(
                        pd.Series({
                            'line_type': line_type,
                            'row': phase_row,
                            'col': phase_col,
                            'resistance': resistance_matrix.at[phase_row, phase_col],
                            'reactance': reactance_matrix.at[phase_row, phase_col],
                            'capacitance': capacitance_matrix.at[phase_row, phase_col]
                        }),
                        ignore_index=True
                    )
                )

        # Obtain number of phases.
        electric_grid_data.electric_grid_line_types.loc[line_type, 'n_phases'] = len(phases_non_neutral)

        # Obtain maximum current.
        # TODO: Validate this.
        electric_grid_data.electric_grid_line_types.loc[line_type, 'maximum_current'] = (
            phase_conductor_maximum_current.loc[phases_non_neutral].mean()
        )

    ####################################################################################################################
    # Update line parameters.
    ####################################################################################################################

    # Update electric grid model.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(electric_grid_data)

    # Update power flow solution.
    power_flow_solution = (
        fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model, der_power_vector_nominal)
    )

    # Obtain results (as numpy arrays).
    der_power_vector = power_flow_solution.der_power_vector  # The DER power vector is also stored in the solution.
    node_voltage_vector = power_flow_solution.node_voltage_vector
    node_voltage_vector_magnitude = np.abs(power_flow_solution.node_voltage_vector)
    branch_power_vector_1 = power_flow_solution.branch_power_vector_1
    branch_power_vector_2 = power_flow_solution.branch_power_vector_2
    loss = power_flow_solution.loss

    # Print results.
    print(f"der_power_vector = \n{der_power_vector}")
    print(f"node_voltage_vector = \n{node_voltage_vector}")
    print(f"node_voltage_vector_magnitude = \n{node_voltage_vector_magnitude}")
    print(f"branch_power_vector_1 = \n{branch_power_vector_1}")
    print(f"branch_power_vector_2 = \n{branch_power_vector_2}")
    print(f"loss = {loss}")

    # Save results to CSV.
    np.savetxt(os.path.join(results_path, f'der_power_vector.csv'), der_power_vector, delimiter=',')
    np.savetxt(os.path.join(results_path, f'node_voltage_vector.csv'), node_voltage_vector, delimiter=',')
    np.savetxt(os.path.join(results_path, f'node_voltage_vector_magnitude.csv'), node_voltage_vector_magnitude, delimiter=',')
    np.savetxt(os.path.join(results_path, f'branch_power_vector_1.csv'), branch_power_vector_1, delimiter=',')
    np.savetxt(os.path.join(results_path, f'branch_power_vector_2.csv'), branch_power_vector_2, delimiter=',')
    np.savetxt(os.path.join(results_path, f'loss.csv'), loss, delimiter=',')

    # Plot some results.

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.ders.to_series().astype(str),  # Labels.
        y=(np.real(der_power_vector) * scenario_data.scenario.at['base_apparent_power'] / 1e3),  # Values in kW.
    )
    figure.update_layout(
        title="DER active power",
        yaxis_title="Active power [kW]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.ders.to_series().astype(str),  # Labels.
        y=(np.imag(der_power_vector) * scenario_data.scenario.at['base_apparent_power'] / 1e3),  # Values in kW.
    )
    figure.update_layout(
        title="DER reactive power",
        yaxis_title="Reactive power [kVA<sub>r</sub>]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.nodes.to_series().astype(str),  # Labels.
        y=(node_voltage_vector_magnitude * scenario_data.scenario.at['base_voltage'] / 1e3),  # Values in kV.
    )
    figure.update_layout(
        title="Node voltage magnitude",
        yaxis_title="Voltage magnitude [kV]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.nodes.to_series().astype(str),  # Labels.
        y=(node_voltage_vector_magnitude / np.abs(electric_grid_model.node_voltage_vector_reference)),  # Values in p.u.
    )
    figure.update_layout(
        title="Node voltage per-unit magnitude",
        yaxis_title="Voltage magnitude [p.u.]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.branches.to_series().astype(str),  # Labels.
        y=(np.abs(branch_power_vector_1) * scenario_data.scenario.at['base_apparent_power'] / 1e3),  # Values in kW.
    )
    figure.update_layout(
        title="Branch apparent power flow (direction 1)",
        yaxis_title="Apparent power [kVA]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.branches.to_series().astype(str),  # Labels.
        y=(np.abs(branch_power_vector_2) * scenario_data.scenario.at['base_apparent_power'] / 1e3),  # Values in kW.
    )
    figure.update_layout(
        title="Branch apparent power flow (direction 2)",
        yaxis_title="Apparent power [kVA]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
