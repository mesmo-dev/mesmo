"""Example script for testing / validating the linear thermal grid model."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mesmo


def main():

    # Settings.
    scenario_name = mesmo.config.config["tests"]["thermal_grid_scenario_name"]
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)
    power_multipliers = np.arange(-0.2, 1.8, 0.1)
    # Select linear thermal grid model type that is being validated.
    linear_thermal_grid_model_method = mesmo.thermal_grid_models.LinearThermalGridModelGlobal
    # Select power flow solution method that is used as reference.
    power_flow_solution_method = mesmo.thermal_grid_models.ThermalPowerFlowSolution

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain base scaling parameters.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    base_power = scenario_data.scenario.at["base_thermal_power"]
    base_head = 1.0

    # Obtain thermal grid model.
    thermal_grid_model = mesmo.thermal_grid_models.ThermalGridModel(scenario_name)

    # Obtain power flow solution for nominal power conditions.
    power_flow_solution_initial = power_flow_solution_method(thermal_grid_model)

    # Obtain linear thermal grid model for nominal power conditions.
    linear_thermal_grid_model = linear_thermal_grid_model_method(thermal_grid_model, power_flow_solution_initial)

    # Instantiate results variables.
    der_thermal_power_vector = pd.DataFrame(index=power_multipliers, columns=thermal_grid_model.ders, dtype=float)
    der_thermal_power_vector_change = pd.DataFrame(
        index=power_multipliers, columns=thermal_grid_model.ders, dtype=float
    )
    node_head_vector_power_flow = pd.DataFrame(index=power_multipliers, columns=thermal_grid_model.nodes, dtype=float)
    node_head_vector_linear_model = pd.DataFrame(index=power_multipliers, columns=thermal_grid_model.nodes, dtype=float)
    branch_flow_vector_power_flow = pd.DataFrame(
        index=power_multipliers, columns=thermal_grid_model.branches, dtype=float
    )
    branch_flow_vector_linear_model = pd.DataFrame(
        index=power_multipliers, columns=thermal_grid_model.branches, dtype=float
    )
    pump_power_power_flow = pd.Series(index=power_multipliers, dtype=float)
    pump_power_linear_model = pd.Series(index=power_multipliers, dtype=float)

    # Obtain DER power / change.
    der_thermal_power_vector.loc[:, :] = np.transpose([power_multipliers]) @ np.array(
        [power_flow_solution_initial.der_thermal_power_vector]
    )
    der_thermal_power_vector_change.loc[:, :] = np.transpose([power_multipliers - 1]) @ np.array(
        [power_flow_solution_initial.der_thermal_power_vector]
    )

    # Obtain power flow solutions.
    power_flow_solutions = mesmo.utils.starmap(
        power_flow_solution_method,
        [(thermal_grid_model, row) for row in der_thermal_power_vector.values],
    )
    power_flow_solutions = dict(zip(power_multipliers, power_flow_solutions))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions[power_multiplier]
        node_head_vector_power_flow.loc[power_multiplier, :] = power_flow_solution.node_head_vector
        branch_flow_vector_power_flow.loc[power_multiplier, :] = power_flow_solution.branch_flow_vector
        pump_power_power_flow.loc[power_multiplier] = np.real(power_flow_solution.pump_power)

    # Obtain linear model solutions.
    node_head_vector_linear_model.loc[:, :] = (
        np.transpose([power_flow_solution_initial.node_head_vector] * len(power_multipliers))
        + linear_thermal_grid_model.sensitivity_node_head_by_der_power
        @ np.transpose(der_thermal_power_vector_change.values)
    ).transpose()
    branch_flow_vector_linear_model.loc[:, :] = (
        np.transpose([power_flow_solution_initial.branch_flow_vector] * len(power_multipliers))
        + linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
        @ np.transpose(der_thermal_power_vector_change.values)
    ).transpose()
    pump_power_linear_model.loc[:] = (
        np.transpose([np.real(power_flow_solution_initial.pump_power)] * len(power_multipliers))
        + linear_thermal_grid_model.sensitivity_pump_power_by_der_power
        @ np.transpose(der_thermal_power_vector_change.values)
    ).ravel()

    # Obtain error values.
    node_head_vector_error = 100.0 * (
        (node_head_vector_linear_model - node_head_vector_power_flow) / node_head_vector_power_flow
    ).abs().mean(axis="columns")
    branch_flow_vector_error = 100.0 * (
        (branch_flow_vector_linear_model - branch_flow_vector_power_flow) / branch_flow_vector_power_flow
    ).mean(axis="columns")
    pump_power_error = 100.0 * ((pump_power_linear_model - pump_power_power_flow) / pump_power_power_flow)

    # Obtain error table.
    linear_thermal_grid_model_error = pd.DataFrame(
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
    linear_thermal_grid_model_error = linear_thermal_grid_model_error.round(2)

    # Print results.
    print(f"linear_thermal_grid_model_error =\n{linear_thermal_grid_model_error}")

    # Apply base scaling to obtain actual unit values.
    thermal_grid_model.node_head_vector_reference *= base_head
    power_flow_solution_initial.der_thermal_power_vector *= base_power
    power_flow_solution_initial.node_head_vector *= base_head
    power_flow_solution_initial.pump_power *= base_power
    der_thermal_power_vector *= base_power
    der_thermal_power_vector_change *= base_power
    node_head_vector_power_flow *= base_head
    node_head_vector_linear_model *= base_head
    pump_power_power_flow *= base_power
    pump_power_linear_model *= base_power

    # Store results as CSV.
    der_thermal_power_vector.to_csv(results_path / "der_power_vector_active.csv")
    der_thermal_power_vector_change.to_csv(results_path / "der_power_vector_active_change.csv")
    node_head_vector_power_flow.to_csv(results_path / "node_head_vector_power_flow.csv")
    node_head_vector_linear_model.to_csv(results_path / "node_head_vector_linear_model.csv")
    branch_flow_vector_power_flow.to_csv(results_path / "branch_flow_vector_power_flow.csv")
    branch_flow_vector_linear_model.to_csv(results_path / "branch_flow_vector_linear_model.csv")
    pump_power_power_flow.to_csv(results_path / "pump_power_power_flow.csv")
    pump_power_linear_model.to_csv(results_path / "pump_power_linear_model.csv")
    linear_thermal_grid_model_error.to_csv(results_path / "linear_thermal_grid_model_error.csv")

    # Plot results.

    # Head.
    for node_index, node in enumerate(thermal_grid_model.nodes):

        figure = go.Figure()
        figure.add_trace(go.Scatter(x=power_multipliers, y=node_head_vector_power_flow.loc[:, node], name="Power flow"))
        figure.add_trace(
            go.Scatter(x=power_multipliers, y=node_head_vector_linear_model.loc[:, node], name="Linear model")
        )
        figure.add_trace(
            go.Scatter(
                x=[1.0],
                y=[power_flow_solution_initial.node_head_vector[node_index]],
                name="Initial point",
                marker=go.scatter.Marker(size=15.0),
                line=go.scatter.Line(width=0.0),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                name="No load",
                marker=go.scatter.Marker(size=15.0),
                line=go.scatter.Line(width=0.0),
            )
        )
        figure.update_layout(
            title=f"Head [m] for<br>(node_type, node_name, phase): {node}",
            yaxis_title="Head [m]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"head_{node}"))

    # Branch flow.
    for branch_index, branch in enumerate(thermal_grid_model.branches):

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=power_multipliers, y=branch_flow_vector_power_flow.loc[:, branch], name="Power flow")
        )
        figure.add_trace(
            go.Scatter(x=power_multipliers, y=branch_flow_vector_linear_model.loc[:, branch], name="Linear model")
        )
        figure.add_trace(
            go.Scatter(
                x=[1.0],
                y=[power_flow_solution_initial.branch_flow_vector[branch_index]],
                name="Initial point",
                marker=go.scatter.Marker(size=15.0),
                line=go.scatter.Line(width=0.0),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[0.0], y=[0.0], name="No load", marker=go.scatter.Marker(size=15.0), line=go.scatter.Line(width=0.0)
            )
        )
        figure.update_layout(
            title=f"Branch volume flow [m³/s] for<br>(branch_type, branch_name, phase): {branch}",
            yaxis_title="Branch volume flow [m³/s]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"branch_flow_{branch}"))

    # Pump power.
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=power_multipliers, y=pump_power_power_flow, name="Power flow"))
    figure.add_trace(go.Scatter(x=power_multipliers, y=pump_power_linear_model, name="Linear model"))
    figure.add_trace(
        go.Scatter(
            x=[1.0],
            y=np.array([np.real(power_flow_solution_initial.pump_power)]).ravel().tolist(),
            name="Initial point",
            marker=go.scatter.Marker(size=15.0),
            line=go.scatter.Line(width=0.0),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0.0], y=[0.0], name="No load", marker=go.scatter.Marker(size=15.0), line=go.scatter.Line(width=0.0)
        )
    )
    figure.update_layout(
        title="Total pump power [W]",
        yaxis_title="Total pump power [W]",
        xaxis_title="Power multiplier",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "pump_power"))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
