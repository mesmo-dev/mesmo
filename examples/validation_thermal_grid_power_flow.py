"""Example script for testing / validating the thermal grid power flow solution."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mesmo


def main():

    # Settings.
    scenario_name = "singapore_tanjongpagar"
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)
    power_multipliers = np.arange(-0.2, 1.2, 0.1)  # TODO: Fix handling of zero branch flows in Newton-Raphson.
    power_flow_solution_method_1 = mesmo.thermal_grid_models.ThermalPowerFlowSolutionNewtonRaphson
    method_1_label = "Method 1: Newton Raphson"
    power_flow_solution_method_2 = mesmo.thermal_grid_models.ThermalPowerFlowSolutionExplicit
    method_2_label = "Method 2: Explicit"

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain base scaling parameters.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    base_power = scenario_data.scenario.at["base_thermal_power"]

    # Obtain thermal grid model.
    thermal_grid_model = mesmo.thermal_grid_models.ThermalGridModel(scenario_name)

    # Obtain nominal power flow solutions.
    power_flow_solution_nominal_method_1 = power_flow_solution_method_1(thermal_grid_model)
    power_flow_solution_nominal_method_2 = power_flow_solution_method_2(thermal_grid_model)

    # Instantiate results variables.
    der_thermal_power_vector = pd.DataFrame(index=power_multipliers, columns=thermal_grid_model.ders, dtype=float)
    node_head_vector_method_1 = pd.DataFrame(index=power_multipliers, columns=thermal_grid_model.nodes, dtype=float)
    node_head_vector_method_2 = pd.DataFrame(index=power_multipliers, columns=thermal_grid_model.nodes, dtype=float)
    branch_flow_vector_method_1 = pd.DataFrame(
        index=power_multipliers, columns=thermal_grid_model.branches, dtype=float
    )
    branch_flow_vector_method_2 = pd.DataFrame(
        index=power_multipliers, columns=thermal_grid_model.branches, dtype=float
    )
    pump_power_method_1 = pd.Series(index=power_multipliers, dtype=float)
    pump_power_method_2 = pd.Series(index=power_multipliers, dtype=float)

    # Obtain DER power / change.
    der_thermal_power_vector.loc[:, :] = np.transpose([power_multipliers]) @ np.array(
        [thermal_grid_model.der_thermal_power_vector_reference]
    )

    # Obtain solutions.
    power_flow_solutions_method_1 = mesmo.utils.starmap(
        power_flow_solution_method_1, [(thermal_grid_model, row) for row in der_thermal_power_vector.values]
    )
    power_flow_solutions_method_1 = dict(zip(power_multipliers, power_flow_solutions_method_1))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions_method_1[power_multiplier]
        node_head_vector_method_1.loc[power_multiplier, :] = power_flow_solution.node_head_vector
        branch_flow_vector_method_1.loc[power_multiplier, :] = power_flow_solution.branch_flow_vector
        pump_power_method_1.loc[power_multiplier] = power_flow_solution.pump_power

    power_flow_solutions_method_2 = mesmo.utils.starmap(
        power_flow_solution_method_2, [(thermal_grid_model, row) for row in der_thermal_power_vector.values]
    )
    power_flow_solutions_method_2 = dict(zip(power_multipliers, power_flow_solutions_method_2))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions_method_2[power_multiplier]
        node_head_vector_method_2.loc[power_multiplier, :] = power_flow_solution.node_head_vector
        branch_flow_vector_method_2.loc[power_multiplier, :] = power_flow_solution.branch_flow_vector
        pump_power_method_2.loc[power_multiplier] = power_flow_solution.pump_power

    # Obtain error values.
    node_head_vector_error = 100.0 * (
        (node_head_vector_method_1 - node_head_vector_method_2) / node_head_vector_method_2
    ).abs().mean(axis="columns")
    branch_flow_vector_error = 100.0 * (
        (branch_flow_vector_method_1 - branch_flow_vector_method_2) / branch_flow_vector_method_2
    ).mean(axis="columns")
    pump_power_error = 100.0 * ((pump_power_method_1 - pump_power_method_2) / pump_power_method_2)

    # Obtain error table.
    power_flow_solution_error = pd.DataFrame(
        [
            node_head_vector_error,
            branch_flow_vector_error,
            pump_power_error,
        ],
        index=[
            "node_head_vector_error",
            "branch_flow_vector_error",
            "pump_power_error",
        ],
    )
    power_flow_solution_error = power_flow_solution_error.round(2)

    # Print results.
    print(f"power_flow_solution_error =\n{power_flow_solution_error}")

    # Apply base scaling to obtain actual unit values.
    power_flow_solution_nominal_method_1.der_thermal_power_vector *= base_power
    power_flow_solution_nominal_method_1.pump_power *= base_power
    power_flow_solution_nominal_method_2.der_thermal_power_vector *= base_power
    power_flow_solution_nominal_method_2.pump_power *= base_power
    der_thermal_power_vector *= base_power
    pump_power_method_1 *= base_power
    pump_power_method_2 *= base_power

    # Store results as CSV.
    der_thermal_power_vector.to_csv(results_path / "der_thermal_power_vector.csv")
    node_head_vector_method_1.to_csv(results_path / "node_head_vector_method_1.csv")
    node_head_vector_method_2.to_csv(results_path / "node_head_vector_method_2.csv")
    pump_power_method_1.to_csv(results_path / "pump_power_method_1.csv")
    pump_power_method_2.to_csv(results_path / "pump_power_method_2.csv")
    power_flow_solution_error.to_csv(results_path / "power_flow_solution_error.csv")

    # Plot results.

    # Nominal head.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[f"{node}" for node in thermal_grid_model.nodes.droplevel("node_type")],
            y=power_flow_solution_nominal_method_1.node_head_vector,
            name=method_1_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-right", size=10),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[f"{node}" for node in thermal_grid_model.nodes.droplevel("node_type")],
            y=power_flow_solution_nominal_method_2.node_head_vector,
            name=method_2_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-left", size=10),
        )
    )
    figure.update_layout(
        title="Node pressure head [m]",
        yaxis_title="Pressure head [m]",
        xaxis_title="Node",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "nominal_head"))

    # Nominal branch flow.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[f"{branch}" for branch in thermal_grid_model.branches],
            y=(
                np.abs(power_flow_solution_nominal_method_1.branch_flow_vector)
                / thermal_grid_model.branch_flow_vector_reference
            ),
            name=method_1_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-right", size=10),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[f"{branch}" for branch in thermal_grid_model.branches],
            y=(
                np.abs(power_flow_solution_nominal_method_2.branch_flow_vector)
                / thermal_grid_model.branch_flow_vector_reference
            ),
            name=method_2_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-left", size=10),
        )
    )
    figure.update_layout(
        title="Branch flow magnitude [p.u.]",
        yaxis_title="Branch flow magnitude [p.u.]",
        xaxis_title="Branch",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "nominal_branch_flow_magnitude"))

    # Nominal pump power.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[""],
            y=[power_flow_solution_nominal_method_1.pump_power],
            name=method_1_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-right", size=10),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[""],
            y=[power_flow_solution_nominal_method_2.pump_power],
            name=method_2_label,
            mode="markers",
            marker=go.scatter.Marker(symbol="arrow-left", size=10),
        )
    )
    figure.update_layout(
        title="Total pump power [W]",
        yaxis_title="Total pump power [W]",
        xaxis=go.layout.XAxis(range=[-1, 2]),
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "nominal_pump_power"))

    # Node head.
    for node_index, node in enumerate(thermal_grid_model.nodes):
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=power_multipliers, y=node_head_vector_method_1.loc[:, node], name=method_1_label))
        figure.add_trace(go.Scatter(x=power_multipliers, y=node_head_vector_method_2.loc[:, node], name=method_2_label))
        figure.update_layout(
            title=f"Pressure head [m] for<br>(node_type, node_name, phase): {node}",
            yaxis_title="Pressure head [m]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"node_head_{node}"))

    # Branch power magnitude.
    for branch_index, branch in enumerate(thermal_grid_model.branches):
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=power_multipliers, y=branch_flow_vector_method_1.loc[:, branch], name=method_1_label)
        )
        figure.add_trace(
            go.Scatter(x=power_multipliers, y=branch_flow_vector_method_2.loc[:, branch], name=method_2_label)
        )
        figure.update_layout(
            title=f"Branch flow [m³/s] for<br>(branch_type, branch_name, phase): {branch}",
            yaxis_title="Branch flow [m³/s]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"branch_flow_{branch}"))

    # Pump power.
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=power_multipliers, y=pump_power_method_1, name=method_1_label))
    figure.add_trace(go.Scatter(x=power_multipliers, y=pump_power_method_2, name=method_2_label))
    figure.update_layout(
        title="Total pump power [W]",
        yaxis_title="Total pump power [W]",
        xaxis_title="Power multiplier",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "loss_active"))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
