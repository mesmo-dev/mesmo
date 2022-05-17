"""Example script for testing / validating the electric grid power flow solution."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mesmo


def main():

    # Settings.
    scenario_name = mesmo.config.config["tests"]["scenario_name"]
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)
    power_multipliers = np.arange(-0.2, 1.2, 0.1)
    power_flow_solution_method_1 = mesmo.electric_grid_models.PowerFlowSolutionFixedPoint
    method_1_label = "Method 1: Fixed Point"
    power_flow_solution_method_2 = mesmo.electric_grid_models.PowerFlowSolutionOpenDSS
    method_2_label = "Method 2: OpenDSS"

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain base scaling parameters.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    base_power = scenario_data.scenario.at["base_apparent_power"]
    base_voltage = scenario_data.scenario.at["base_voltage"]

    # Obtain electric grid model.
    electric_grid_model = mesmo.electric_grid_models.ElectricGridModelOpenDSS(scenario_name)

    # Obtain nominal power flow solutions.
    power_flow_solution_nominal_method_1 = power_flow_solution_method_1(electric_grid_model)
    power_flow_solution_nominal_method_2 = power_flow_solution_method_2(electric_grid_model)

    # Instantiate results variables.
    der_power_vector = pd.DataFrame(index=power_multipliers, columns=electric_grid_model.ders, dtype=float)
    node_voltage_vector_method_1 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.nodes, dtype=complex
    )
    node_voltage_vector_method_2 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.nodes, dtype=complex
    )
    node_voltage_vector_magnitude_method_1 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.nodes, dtype=float
    )
    node_voltage_vector_magnitude_method_2 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.nodes, dtype=float
    )
    branch_power_vector_1_method_1 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_1_method_2 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_2_method_1 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_2_method_2 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_1_magnitude_method_1 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_1_magnitude_method_2 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_2_magnitude_method_1 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_2_magnitude_method_2 = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    loss_active_method_1 = pd.Series(index=power_multipliers, dtype=float)
    loss_active_method_2 = pd.Series(index=power_multipliers, dtype=float)
    loss_reactive_method_1 = pd.Series(index=power_multipliers, dtype=float)
    loss_reactive_method_2 = pd.Series(index=power_multipliers, dtype=float)

    # Obtain DER power / change.
    der_power_vector.loc[:, :] = np.transpose([power_multipliers]) @ np.array(
        [electric_grid_model.der_power_vector_reference]
    )

    # Obtain solutions.
    power_flow_solutions_method_1 = mesmo.utils.starmap(
        power_flow_solution_method_1, [(electric_grid_model, row) for row in der_power_vector.values]
    )
    power_flow_solutions_method_1 = dict(zip(power_multipliers, power_flow_solutions_method_1))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions_method_1[power_multiplier]
        node_voltage_vector_method_1.loc[power_multiplier, :] = power_flow_solution.node_voltage_vector
        node_voltage_vector_magnitude_method_1.loc[power_multiplier, :] = np.abs(
            power_flow_solution.node_voltage_vector
        )
        branch_power_vector_1_method_1.loc[power_multiplier, :] = power_flow_solution.branch_power_vector_1
        branch_power_vector_2_method_1.loc[power_multiplier, :] = power_flow_solution.branch_power_vector_2
        branch_power_vector_1_magnitude_method_1.loc[power_multiplier, :] = np.abs(
            power_flow_solution.branch_power_vector_1
        )
        branch_power_vector_2_magnitude_method_1.loc[power_multiplier, :] = np.abs(
            power_flow_solution.branch_power_vector_2
        )
        loss_active_method_1.loc[power_multiplier] = np.real(power_flow_solution.loss)
        loss_reactive_method_1.loc[power_multiplier] = np.imag(power_flow_solution.loss)

    power_flow_solutions_method_2 = mesmo.utils.starmap(
        power_flow_solution_method_2, [(electric_grid_model, row) for row in der_power_vector.values]
    )
    power_flow_solutions_method_2 = dict(zip(power_multipliers, power_flow_solutions_method_2))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions_method_2[power_multiplier]
        node_voltage_vector_method_2.loc[power_multiplier, :] = power_flow_solution.node_voltage_vector
        node_voltage_vector_magnitude_method_2.loc[power_multiplier, :] = np.abs(
            power_flow_solution.node_voltage_vector
        )
        branch_power_vector_1_method_2.loc[power_multiplier, :] = power_flow_solution.branch_power_vector_1
        branch_power_vector_2_method_2.loc[power_multiplier, :] = power_flow_solution.branch_power_vector_2
        branch_power_vector_1_magnitude_method_2.loc[power_multiplier, :] = np.abs(
            power_flow_solution.branch_power_vector_1
        )
        branch_power_vector_2_magnitude_method_2.loc[power_multiplier, :] = np.abs(
            power_flow_solution.branch_power_vector_2
        )
        loss_active_method_2.loc[power_multiplier] = np.real(power_flow_solution.loss)
        loss_reactive_method_2.loc[power_multiplier] = np.imag(power_flow_solution.loss)

    # Obtain error values.
    node_voltage_vector_error = 100.0 * (
        (node_voltage_vector_method_1 - node_voltage_vector_method_2) / node_voltage_vector_method_2
    ).abs().mean(axis="columns")
    node_voltage_vector_magnitude_error = 100.0 * (
        (node_voltage_vector_magnitude_method_1 - node_voltage_vector_magnitude_method_2)
        / node_voltage_vector_magnitude_method_2
    ).mean(axis="columns")
    branch_power_vector_1_magnitude_error = 100.0 * (
        (branch_power_vector_1_magnitude_method_1 - branch_power_vector_1_magnitude_method_2)
        / branch_power_vector_1_magnitude_method_2
    ).mean(axis="columns")
    branch_power_vector_2_magnitude_error = 100.0 * (
        (branch_power_vector_2_magnitude_method_1 - branch_power_vector_2_magnitude_method_2)
        / branch_power_vector_2_magnitude_method_2
    ).mean(axis="columns")
    loss_active_error = 100.0 * ((loss_active_method_1 - loss_active_method_2) / loss_active_method_2)
    loss_reactive_error = 100.0 * ((loss_reactive_method_1 - loss_reactive_method_2) / loss_reactive_method_2)

    # Obtain error table.
    power_flow_solution_error = pd.DataFrame(
        [
            node_voltage_vector_error,
            node_voltage_vector_magnitude_error,
            branch_power_vector_1_magnitude_error,
            branch_power_vector_2_magnitude_error,
            loss_active_error,
            loss_reactive_error,
        ],
        index=[
            "node_voltage_vector_error",
            "node_voltage_vector_magnitude_error",
            "branch_power_vector_1_magnitude_error",
            "branch_power_vector_2_magnitude_error",
            "loss_active_error",
            "loss_reactive_error",
        ],
    )
    power_flow_solution_error = power_flow_solution_error.round(2)

    # Print results.
    print(f"power_flow_solution_error =\n{power_flow_solution_error}")

    # Apply base scaling to obtain actual unit values.
    electric_grid_model.node_voltage_vector_reference *= base_voltage
    power_flow_solution_nominal_method_1.der_power_vector *= base_power
    power_flow_solution_nominal_method_1.node_voltage_vector *= base_voltage
    power_flow_solution_nominal_method_1.branch_power_vector_1 *= base_power
    power_flow_solution_nominal_method_1.branch_power_vector_2 *= base_power
    power_flow_solution_nominal_method_1.loss *= base_power
    power_flow_solution_nominal_method_2.der_power_vector *= base_power
    power_flow_solution_nominal_method_2.node_voltage_vector *= base_voltage
    power_flow_solution_nominal_method_2.branch_power_vector_1 *= base_power
    power_flow_solution_nominal_method_2.branch_power_vector_2 *= base_power
    power_flow_solution_nominal_method_2.loss *= base_power
    der_power_vector *= base_power
    node_voltage_vector_method_1 *= base_voltage
    node_voltage_vector_method_2 *= base_voltage
    node_voltage_vector_magnitude_method_1 *= base_voltage
    node_voltage_vector_magnitude_method_2 *= base_voltage
    branch_power_vector_1_method_1 *= base_power
    branch_power_vector_1_method_2 *= base_power
    branch_power_vector_2_method_1 *= base_power
    branch_power_vector_2_method_2 *= base_power
    branch_power_vector_1_magnitude_method_1 *= base_power
    branch_power_vector_1_magnitude_method_2 *= base_power
    branch_power_vector_2_magnitude_method_1 *= base_power
    branch_power_vector_2_magnitude_method_2 *= base_power
    loss_active_method_1 *= base_power
    loss_active_method_2 *= base_power
    loss_reactive_method_1 *= base_power
    loss_reactive_method_2 *= base_power

    # Store results as CSV.
    der_power_vector.to_csv(results_path / "der_power_vector.csv")
    node_voltage_vector_method_1.to_csv(results_path / "node_voltage_vector_method_1.csv")
    node_voltage_vector_method_2.to_csv(results_path / "node_voltage_vector_method_2.csv")
    node_voltage_vector_magnitude_method_1.to_csv(results_path / "node_voltage_vector_magnitude_method_1.csv")
    node_voltage_vector_magnitude_method_2.to_csv(results_path / "node_voltage_vector_magnitude_method_2.csv")
    branch_power_vector_1_magnitude_method_1.to_csv(results_path / "branch_power_vector_1_magnitude_method_1.csv")
    branch_power_vector_1_magnitude_method_2.to_csv(results_path / "branch_power_vector_1_magnitude_method_2.csv")
    branch_power_vector_2_magnitude_method_1.to_csv(results_path / "branch_power_vector_2_magnitude_method_1.csv")
    branch_power_vector_2_magnitude_method_2.to_csv(results_path / "branch_power_vector_2_magnitude_method_2.csv")
    loss_active_method_1.to_csv(results_path / "loss_active_method_1.csv")
    loss_active_method_2.to_csv(results_path / "loss_active_method_2.csv")
    loss_reactive_method_1.to_csv(results_path / "loss_reactive_method_1.csv")
    loss_reactive_method_2.to_csv(results_path / "loss_reactive_method_2.csv")
    power_flow_solution_error.to_csv(results_path / "power_flow_solution_error.csv")

    # Plot results.

    # Nominal voltage magnitude.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[f"{node}" for node in electric_grid_model.nodes.droplevel("node_type")],
            y=(
                np.abs(power_flow_solution_nominal_method_1.node_voltage_vector)
                / np.abs(electric_grid_model.node_voltage_vector_reference)
            ),
            name=method_1_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-right", size=10),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[f"{node}" for node in electric_grid_model.nodes.droplevel("node_type")],
            y=(
                np.abs(power_flow_solution_nominal_method_2.node_voltage_vector)
                / np.abs(electric_grid_model.node_voltage_vector_reference)
            ),
            name=method_2_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-left", size=10),
        )
    )
    figure.update_layout(
        title=f"Node voltage magnitude [p.u.]",
        yaxis_title="Voltage magnitude [p.u.]",
        xaxis_title="Node",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "nominal_voltage_magnitude"))

    # Nominal branch power magnitude 1.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[f"{branch}" for branch in electric_grid_model.branches],
            y=(
                np.abs(power_flow_solution_nominal_method_1.branch_power_vector_1)
                / electric_grid_model.branch_power_vector_magnitude_reference
            ),
            name=method_1_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-right", size=10),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[f"{branch}" for branch in electric_grid_model.branches],
            y=(
                np.abs(power_flow_solution_nominal_method_2.branch_power_vector_1)
                / electric_grid_model.branch_power_vector_magnitude_reference
            ),
            name=method_2_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-left", size=10),
        )
    )
    figure.update_layout(
        title=f"Branch power (direction 1) magnitude [p.u.]",
        yaxis_title="Branch power magnitude [p.u.]",
        xaxis_title="Branch",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "nominal_branch_power_1_magnitude"))

    # Nominal branch power magnitude 2.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[f"{branch}" for branch in electric_grid_model.branches],
            y=(
                np.abs(power_flow_solution_nominal_method_1.branch_power_vector_2)
                / electric_grid_model.branch_power_vector_magnitude_reference
            ),
            name=method_1_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-right", size=10),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[f"{branch}" for branch in electric_grid_model.branches],
            y=(
                np.abs(power_flow_solution_nominal_method_2.branch_power_vector_2)
                / electric_grid_model.branch_power_vector_magnitude_reference
            ),
            name=method_2_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-left", size=10),
        )
    )
    figure.update_layout(
        title=f"Branch power (direction 2) magnitude [p.u.]",
        yaxis_title="Branch power magnitude [p.u.]",
        xaxis_title="Branch",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "nominal_branch_power_2_magnitude"))

    # Nominal losses.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=["active", "reactive"],
            y=[
                np.real(power_flow_solution_nominal_method_1.loss)[0],
                np.imag(power_flow_solution_nominal_method_1.loss)[0],
            ],
            name=method_1_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-right", size=10),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=["active", "reactive"],
            y=[
                np.real(power_flow_solution_nominal_method_1.loss)[0],
                np.imag(power_flow_solution_nominal_method_1.loss)[0],
            ],
            name=method_2_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-left", size=10),
        )
    )
    figure.update_layout(
        title=f"Total loss [VA]",
        yaxis_title="Total loss [VA]",
        xaxis=go.layout.XAxis(range=[-1, 2]),
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "nominal_loss"))

    # Voltage magnitude.
    for node_index, node in enumerate(electric_grid_model.nodes):
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=power_multipliers, y=node_voltage_vector_magnitude_method_1.loc[:, node], name=method_1_label)
        )
        figure.add_trace(
            go.Scatter(x=power_multipliers, y=node_voltage_vector_magnitude_method_2.loc[:, node], name=method_2_label)
        )
        figure.update_layout(
            title=f"Voltage magnitude [V] for<br>(node_type, node_name, phase): {node}",
            yaxis_title="Voltage magnitude [V]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"voltage_magnitude_{node}"))

    # Branch power magnitude.
    for branch_index, branch in enumerate(electric_grid_model.branches):
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=branch_power_vector_1_magnitude_method_1.loc[:, branch], name=method_1_label
            )
        )
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=branch_power_vector_1_magnitude_method_2.loc[:, branch], name=method_2_label
            )
        )
        figure.update_layout(
            title=f"Branch power 1 magnitude [VA] for<br>(branch_type, branch_name, phase): {branch}",
            yaxis_title="Branch power magnitude [VA]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"branch_power_1_magnitude_{branch}"))

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=branch_power_vector_2_magnitude_method_1.loc[:, branch], name=method_1_label
            )
        )
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=branch_power_vector_2_magnitude_method_2.loc[:, branch], name=method_2_label
            )
        )
        figure.update_layout(
            title=f"Branch power 2 magnitude [VA] for<br>(branch_type, branch_name, phase): {branch}",
            yaxis_title="Branch power magnitude [VA]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"branch_power_2_magnitude_{branch}"))

    # Loss active.
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=power_multipliers, y=loss_active_method_1, name=method_1_label))
    figure.add_trace(go.Scatter(x=power_multipliers, y=loss_active_method_2, name=method_2_label))
    figure.update_layout(
        title=f"Total loss active [W]",
        yaxis_title="Total loss active [W]",
        xaxis_title="Power multiplier",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "loss_active"))

    # Loss reactive.
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=power_multipliers, y=loss_reactive_method_1, name=method_1_label))
    figure.add_trace(go.Scatter(x=power_multipliers, y=loss_reactive_method_2, name=method_2_label))
    figure.update_layout(
        title=f"Total loss reactive [VAr]",
        yaxis_title="Total loss reactive [VAr]",
        xaxis_title="Power multiplier",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "loss_reactive"))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
