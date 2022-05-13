"""Example script for testing / validating the linear electric grid model."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mesmo


def main():

    # Settings.
    scenario_name = mesmo.config.config["tests"]["scenario_name"]
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)
    power_multipliers = np.arange(-0.2, 1.8, 0.1)
    # Select linear electric grid model type that is being validated.
    linear_electric_grid_model_method = mesmo.electric_grid_models.LinearElectricGridModelGlobal
    # Select power flow solution method that is used as reference.
    power_flow_solution_method = mesmo.electric_grid_models.PowerFlowSolutionOpenDSS

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain base scaling parameters.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    base_power = scenario_data.scenario.at["base_apparent_power"]
    base_voltage = scenario_data.scenario.at["base_voltage"]

    # Obtain electric grid model.
    if power_flow_solution_method is mesmo.electric_grid_models.PowerFlowSolutionOpenDSS:
        electric_grid_model = mesmo.electric_grid_models.ElectricGridModelOpenDSS(scenario_name)
    else:
        electric_grid_model = mesmo.electric_grid_models.ElectricGridModel(scenario_name)

    # Obtain power flow solution for nominal power conditions.
    power_flow_solution_initial = power_flow_solution_method(electric_grid_model)

    # Obtain linear electric grid model for nominal power conditions.
    linear_electric_grid_model = linear_electric_grid_model_method(electric_grid_model, power_flow_solution_initial)

    # Instantiate results variables.
    der_power_vector_active = pd.DataFrame(index=power_multipliers, columns=electric_grid_model.ders, dtype=float)
    der_power_vector_reactive = pd.DataFrame(index=power_multipliers, columns=electric_grid_model.ders, dtype=float)
    der_power_vector_active_change = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.ders, dtype=float
    )
    der_power_vector_reactive_change = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.ders, dtype=float
    )
    node_voltage_vector_power_flow = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.nodes, dtype=complex
    )
    node_voltage_vector_linear_model = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.nodes, dtype=complex
    )
    node_voltage_vector_magnitude_power_flow = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.nodes, dtype=float
    )
    node_voltage_vector_magnitude_linear_model = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.nodes, dtype=float
    )
    branch_power_vector_1_power_flow = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_1_linear_model = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_2_power_flow = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_2_linear_model = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_1_magnitude_power_flow = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_1_magnitude_linear_model = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_2_magnitude_power_flow = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    branch_power_vector_2_magnitude_linear_model = pd.DataFrame(
        index=power_multipliers, columns=electric_grid_model.branches, dtype=float
    )
    loss_active_power_flow = pd.Series(index=power_multipliers, dtype=float)
    loss_active_linear_model = pd.Series(index=power_multipliers, dtype=float)
    loss_reactive_power_flow = pd.Series(index=power_multipliers, dtype=float)
    loss_reactive_linear_model = pd.Series(index=power_multipliers, dtype=float)

    # Obtain DER power / change.
    der_power_vector_active.loc[:, :] = np.transpose([power_multipliers]) @ np.array(
        [np.real(power_flow_solution_initial.der_power_vector)]
    )
    der_power_vector_reactive.loc[:, :] = np.transpose([power_multipliers]) @ np.array(
        [np.imag(power_flow_solution_initial.der_power_vector)]
    )
    der_power_vector_active_change.loc[:, :] = np.transpose([power_multipliers - 1]) @ np.array(
        [np.real(power_flow_solution_initial.der_power_vector)]
    )
    der_power_vector_reactive_change.loc[:, :] = np.transpose([power_multipliers - 1]) @ np.array(
        [np.imag(power_flow_solution_initial.der_power_vector)]
    )

    # Obtain power flow solutions.
    power_flow_solutions = mesmo.utils.starmap(
        power_flow_solution_method,
        [(electric_grid_model, row) for row in (der_power_vector_active + 1.0j * der_power_vector_reactive).values],
    )
    power_flow_solutions = dict(zip(power_multipliers, power_flow_solutions))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions[power_multiplier]
        node_voltage_vector_power_flow.loc[power_multiplier, :] = power_flow_solution.node_voltage_vector
        node_voltage_vector_magnitude_power_flow.loc[power_multiplier, :] = np.abs(
            power_flow_solution.node_voltage_vector
        )
        branch_power_vector_1_power_flow.loc[power_multiplier, :] = power_flow_solution.branch_power_vector_1
        branch_power_vector_2_power_flow.loc[power_multiplier, :] = power_flow_solution.branch_power_vector_2
        branch_power_vector_1_magnitude_power_flow.loc[power_multiplier, :] = np.abs(
            power_flow_solution.branch_power_vector_1
        )
        branch_power_vector_2_magnitude_power_flow.loc[power_multiplier, :] = np.abs(
            power_flow_solution.branch_power_vector_2
        )
        loss_active_power_flow.loc[power_multiplier] = np.real(power_flow_solution.loss)
        loss_reactive_power_flow.loc[power_multiplier] = np.imag(power_flow_solution.loss)

    # Obtain linear model solutions.
    node_voltage_vector_linear_model.loc[:, :] = (
        np.transpose([power_flow_solution_initial.node_voltage_vector] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_voltage_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_voltage_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    node_voltage_vector_magnitude_linear_model.loc[:, :] = (
        np.transpose([np.abs(power_flow_solution_initial.node_voltage_vector)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    branch_power_vector_1_linear_model.loc[:, :] = (
        np.transpose([power_flow_solution_initial.branch_power_vector_1] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_branch_power_1_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_branch_power_1_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    branch_power_vector_2_linear_model.loc[:, :] = (
        np.transpose([power_flow_solution_initial.branch_power_vector_2] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_branch_power_2_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_branch_power_2_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    branch_power_vector_1_magnitude_linear_model.loc[:, :] = (
        np.transpose([np.abs(power_flow_solution_initial.branch_power_vector_1)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    branch_power_vector_2_magnitude_linear_model.loc[:, :] = (
        np.transpose([np.abs(power_flow_solution_initial.branch_power_vector_2)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    loss_active_linear_model.loc[:] = (
        np.transpose([np.real(power_flow_solution_initial.loss)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).ravel()
    loss_reactive_linear_model.loc[:] = (
        np.transpose([np.imag(power_flow_solution_initial.loss)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).ravel()

    # Obtain error values.
    node_voltage_vector_error = 100.0 * (
        (node_voltage_vector_linear_model - node_voltage_vector_power_flow) / node_voltage_vector_power_flow
    ).abs().mean(axis="columns")
    node_voltage_vector_error_real = 100.0 * (
        (node_voltage_vector_linear_model.apply(np.real) - node_voltage_vector_power_flow.apply(np.real))
        / node_voltage_vector_power_flow.apply(np.real)
    ).mean(axis="columns")
    node_voltage_vector_error_imag = 100.0 * (
        (node_voltage_vector_linear_model.apply(np.imag) - node_voltage_vector_power_flow.apply(np.imag))
        / node_voltage_vector_power_flow.apply(np.imag)
    ).mean(axis="columns")
    node_voltage_vector_magnitude_error = 100.0 * (
        (node_voltage_vector_magnitude_linear_model - node_voltage_vector_magnitude_power_flow)
        / node_voltage_vector_magnitude_power_flow
    ).mean(axis="columns")
    branch_power_vector_1_magnitude_error = 100.0 * (
        (branch_power_vector_1_magnitude_linear_model - branch_power_vector_1_magnitude_power_flow)
        / branch_power_vector_1_magnitude_power_flow
    ).mean(axis="columns")
    branch_power_vector_2_magnitude_error = 100.0 * (
        (branch_power_vector_2_magnitude_linear_model - branch_power_vector_2_magnitude_power_flow)
        / branch_power_vector_2_magnitude_power_flow
    ).mean(axis="columns")
    loss_active_error = 100.0 * ((loss_active_linear_model - loss_active_power_flow) / loss_active_power_flow)
    loss_reactive_error = 100.0 * ((loss_reactive_linear_model - loss_reactive_power_flow) / loss_reactive_power_flow)

    # Obtain error table.
    linear_electric_grid_model_error = pd.DataFrame(
        [
            node_voltage_vector_error,
            node_voltage_vector_error_real,
            node_voltage_vector_error_imag,
            node_voltage_vector_magnitude_error,
            branch_power_vector_1_magnitude_error,
            branch_power_vector_2_magnitude_error,
            loss_active_error,
            loss_reactive_error,
        ],
        index=[
            "node_voltage_vector_error",
            "node_voltage_vector_error_real",
            "node_voltage_vector_error_imag",
            "node_voltage_vector_magnitude_error",
            "branch_power_vector_1_magnitude_error",
            "branch_power_vector_2_magnitude_error",
            "loss_active_error",
            "loss_reactive_error",
        ],
    )
    linear_electric_grid_model_error = linear_electric_grid_model_error.round(2)

    # Print results.
    print(f"linear_electric_grid_model_error =\n{linear_electric_grid_model_error}")

    # Apply base scaling to obtain actual unit values.
    electric_grid_model.node_voltage_vector_reference *= base_voltage
    power_flow_solution_initial.der_power_vector *= base_power
    power_flow_solution_initial.node_voltage_vector *= base_voltage
    power_flow_solution_initial.branch_power_vector_1 *= base_power
    power_flow_solution_initial.branch_power_vector_2 *= base_power
    power_flow_solution_initial.loss *= base_power
    der_power_vector_active *= base_power
    der_power_vector_reactive *= base_power
    der_power_vector_active_change *= base_power
    der_power_vector_reactive_change *= base_power
    node_voltage_vector_power_flow *= base_voltage
    node_voltage_vector_linear_model *= base_voltage
    node_voltage_vector_magnitude_power_flow *= base_voltage
    node_voltage_vector_magnitude_linear_model *= base_voltage
    branch_power_vector_1_power_flow *= base_power
    branch_power_vector_1_linear_model *= base_power
    branch_power_vector_2_power_flow *= base_power
    branch_power_vector_2_linear_model *= base_power
    branch_power_vector_1_magnitude_power_flow *= base_power
    branch_power_vector_1_magnitude_linear_model *= base_power
    branch_power_vector_2_magnitude_power_flow *= base_power
    branch_power_vector_2_magnitude_linear_model *= base_power
    loss_active_power_flow *= base_power
    loss_active_linear_model *= base_power
    loss_reactive_power_flow *= base_power
    loss_reactive_linear_model *= base_power

    # Store results as CSV.
    der_power_vector_active.to_csv(results_path / "der_power_vector_active.csv")
    der_power_vector_reactive.to_csv(results_path / "der_power_vector_reactive.csv")
    der_power_vector_active_change.to_csv(results_path / "der_power_vector_active_change.csv")
    der_power_vector_reactive_change.to_csv(results_path / "der_power_vector_reactive_change.csv")
    node_voltage_vector_power_flow.to_csv(results_path / "node_voltage_vector_power_flow.csv")
    node_voltage_vector_linear_model.to_csv(results_path / "node_voltage_vector_linear_model.csv")
    node_voltage_vector_magnitude_power_flow.to_csv(results_path / "node_voltage_vector_magnitude_power_flow.csv")
    node_voltage_vector_magnitude_linear_model.to_csv(results_path / "node_voltage_vector_magnitude_linear_model.csv")
    branch_power_vector_1_power_flow.to_csv(results_path / "branch_power_vector_1_power_flow.csv")
    branch_power_vector_1_linear_model.to_csv(results_path / "branch_power_vector_1_linear_model.csv")
    branch_power_vector_2_power_flow.to_csv(results_path / "branch_power_vector_2_power_flow.csv")
    branch_power_vector_2_linear_model.to_csv(results_path / "branch_power_vector_2_linear_model.csv")
    branch_power_vector_1_magnitude_power_flow.to_csv(results_path / "branch_power_vector_1_magnitude_power_flow.csv")
    branch_power_vector_1_magnitude_linear_model.to_csv(
        results_path / "branch_power_vector_1_magnitude_linear_model.csv"
    )
    branch_power_vector_2_magnitude_power_flow.to_csv(results_path / "branch_power_vector_2_magnitude_power_flow.csv")
    branch_power_vector_2_magnitude_linear_model.to_csv(
        results_path / "branch_power_vector_2_magnitude_linear_model.csv"
    )
    loss_active_power_flow.to_csv(results_path / "loss_active_power_flow.csv")
    loss_active_linear_model.to_csv(results_path / "loss_active_linear_model.csv")
    loss_reactive_power_flow.to_csv(results_path / "loss_reactive_power_flow.csv")
    loss_reactive_linear_model.to_csv(results_path / "loss_reactive_linear_model.csv")
    linear_electric_grid_model_error.to_csv(results_path / "linear_electric_grid_model_error.csv")

    # Plot results.

    # Voltage.
    for node_index, node in enumerate(electric_grid_model.nodes):

        # Voltage magnitude.
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=power_multipliers, y=node_voltage_vector_magnitude_power_flow.loc[:, node], name="Power flow")
        )
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=node_voltage_vector_magnitude_linear_model.loc[:, node], name="Linear model"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[1.0],
                y=[abs(power_flow_solution_initial.node_voltage_vector[node_index])],
                name="Initial point",
                marker=go.scatter.Marker(size=15.0),
                line=go.scatter.Line(width=0.0),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[0.0],
                y=[abs(electric_grid_model.node_voltage_vector_reference[node_index])],
                name="No load",
                marker=go.scatter.Marker(size=15.0),
                line=go.scatter.Line(width=0.0),
            )
        )
        figure.update_layout(
            title=f"Voltage magnitude [V] for<br>(node_type, node_name, phase): {node}",
            yaxis_title="Voltage magnitude [V]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"voltage_magnitude_{node}"))

        # Voltage real component.
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=power_multipliers, y=np.real(node_voltage_vector_power_flow.loc[:, node]), name="Power flow")
        )
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=np.real(node_voltage_vector_linear_model.loc[:, node]), name="Linear model"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[1.0],
                y=[np.real(power_flow_solution_initial.node_voltage_vector[node_index])],
                name="Initial point",
                marker=go.scatter.Marker(size=15.0),
                line=go.scatter.Line(width=0.0),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[0.0],
                y=[np.real(electric_grid_model.node_voltage_vector_reference[node_index])],
                name="No load",
                marker=go.scatter.Marker(size=15.0),
                line=go.scatter.Line(width=0.0),
            )
        )
        figure.update_layout(
            title=f"Voltage (real component) [V] for<br>(node_type, node_name, phase): {node}",
            yaxis_title="Voltage magnitude [V]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"voltage_real_{node}"))

        # Voltage imaginary component.
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=power_multipliers, y=np.imag(node_voltage_vector_power_flow.loc[:, node]), name="Power flow")
        )
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=np.imag(node_voltage_vector_linear_model.loc[:, node]), name="Linear model"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[1.0],
                y=[np.imag(power_flow_solution_initial.node_voltage_vector[node_index])],
                name="Initial point",
                marker=go.scatter.Marker(size=15.0),
                line=go.scatter.Line(width=0.0),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[0.0],
                y=[np.imag(electric_grid_model.node_voltage_vector_reference[node_index])],
                name="No load",
                marker=go.scatter.Marker(size=15.0),
                line=go.scatter.Line(width=0.0),
            )
        )
        figure.update_layout(
            title=f"Voltage (imaginary component) [V] for<br>(node_type, node_name, phase): {node}",
            yaxis_title="Voltage magnitude [V]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"voltage_imag_{node}"))

    # Branch flow.
    for branch_index, branch in enumerate(electric_grid_model.branches):

        # Branch power active component 1.
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=np.real(branch_power_vector_1_power_flow.loc[:, branch]), name="Power flow"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=np.real(branch_power_vector_1_linear_model.loc[:, branch]), name="Linear model"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[1.0],
                y=[np.real(power_flow_solution_initial.branch_power_vector_1[branch_index])],
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
            title=f"Branch power 1 active component [W] for<br>(branch_type, branch_name, phase): {branch}",
            yaxis_title="Branch power active component [W]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"branch_power_1_active_{branch}"))

        # Branch power reactive component 1.
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=np.imag(branch_power_vector_1_power_flow.loc[:, branch]), name="Power flow"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=np.imag(branch_power_vector_1_linear_model.loc[:, branch]), name="Linear model"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[1.0],
                y=[np.imag(power_flow_solution_initial.branch_power_vector_1[branch_index])],
                name="Initial point",
                marker=go.scatter.Marker(size=15.0),
                line=go.scatter.Line(width=0.0),
            )
        )
        figure.update_layout(
            title=f"Branch power 1 reactive component [VAr] for<br>(branch_type, branch_name, phase): {branch}",
            yaxis_title="Branch power reactive component [VAr]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"branch_power_1_reactive_{branch}"))

        # Branch power active component 2.
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=np.real(branch_power_vector_2_power_flow.loc[:, branch]), name="Power flow"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=np.real(branch_power_vector_2_linear_model.loc[:, branch]), name="Linear model"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[1.0],
                y=[np.real(power_flow_solution_initial.branch_power_vector_2[branch_index])],
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
            title=f"Branch power 2 active component [W] for<br>(branch_type, branch_name, phase): {branch}",
            yaxis_title="Branch power active component [W]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"branch_power_2_active_{branch}"))

        # Branch power reactive component 2.
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=np.imag(branch_power_vector_2_power_flow.loc[:, branch]), name="Power flow"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=np.imag(branch_power_vector_2_linear_model.loc[:, branch]), name="Linear model"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[1.0],
                y=[np.imag(power_flow_solution_initial.branch_power_vector_2[branch_index])],
                name="Initial point",
                marker=go.scatter.Marker(size=15.0),
                line=go.scatter.Line(width=0.0),
            )
        )
        figure.update_layout(
            title=f"Branch power 2 reactive component [VAr] for<br>(branch_type, branch_name, phase): {branch}",
            yaxis_title="Branch power reactive component [VAr]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"branch_power_2_reactive_{branch}"))

        # Branch power magnitude 1.
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=branch_power_vector_1_magnitude_power_flow.loc[:, branch], name="Power flow"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=branch_power_vector_1_magnitude_linear_model.loc[:, branch], name="Linear model"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[1.0],
                y=[abs(power_flow_solution_initial.branch_power_vector_1[branch_index])],
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
            title=f"Branch power 1 magnitude [VA] for<br>(branch_type, branch_name, phase): {branch}",
            yaxis_title="Branch power magnitude [VA]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"branch_power_1_magnitude_{branch}"))

        # Branch power magnitude 2.
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=branch_power_vector_2_magnitude_power_flow.loc[:, branch], name="Power flow"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=power_multipliers, y=branch_power_vector_2_magnitude_linear_model.loc[:, branch], name="Linear model"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[1.0],
                y=[abs(power_flow_solution_initial.branch_power_vector_2[branch_index])],
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
            title=f"Branch power 2 magnitude [VA] for<br>(branch_type, branch_name, phase): {branch}",
            yaxis_title="Branch power magnitude [VA]",
            xaxis_title="Power multiplier",
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
        )
        mesmo.utils.write_figure_plotly(figure, (results_path / f"branch_power_2_magnitude_{branch}"))

    # Loss active.
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=power_multipliers, y=loss_active_power_flow, name="Power flow"))
    figure.add_trace(go.Scatter(x=power_multipliers, y=loss_active_linear_model, name="Linear model"))
    figure.add_trace(
        go.Scatter(
            x=[1.0],
            y=np.array([np.real(power_flow_solution_initial.loss)]).ravel().tolist(),
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
        title=f"Total loss active [W]",
        yaxis_title="Total loss active [W]",
        xaxis_title="Power multiplier",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    mesmo.utils.write_figure_plotly(figure, (results_path / "loss_active"))

    # Loss reactive.
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=power_multipliers, y=loss_reactive_power_flow, name="Power flow"))
    figure.add_trace(go.Scatter(x=power_multipliers, y=loss_reactive_linear_model, name="Linear model"))
    figure.add_trace(
        go.Scatter(
            x=[1.0],
            y=np.array([np.imag(power_flow_solution_initial.loss)]).ravel().tolist(),
            name="Initial point",
            marker=go.scatter.Marker(size=15.0),
            line=go.scatter.Line(width=0.0),
        )
    )
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
