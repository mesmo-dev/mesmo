"""DRO complete problem and run script."""

import csv
import gurobipy as gp
import numpy as np
import pandas as pd
import pathlib
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp
from tqdm import tqdm

import mesmo
from dro_stage_1 import Stage1
from dro_stage_2 import Stage2
from dro_data_interface import DRODataSet
from dro_data_interface import DROAmbiguitySet


def main():

    # Settings.
    scenario_name = "paper_2021_zhang_dro"
    enable_electric_grid_model = True
    mesmo.data_interface.recreate_database()

    # Get results path.
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Obtain data objects.
    dro_data_set = DRODataSet((pathlib.Path(__file__).parent / "dro_data"))

    # Define scenario sets.
    price_categories = ["energy", "up_reserve", "down_reserve"]

    # Obtain individual stage optimization problems.
    stage_1 = Stage1(scenario_name, dro_data_set, enable_electric_grid_model=enable_electric_grid_model)
    stage_2 = Stage2(scenario_name, dro_data_set, enable_electric_grid_model=enable_electric_grid_model)

    # Instantiate optimization problem.
    optimization_problem = mesmo.solutions.OptimizationProblem()

    mesmo.utils.logger.info("Define DRO constants and variables")

    # Define constants. TO-DO clean up inputs
    dro_ambiguity_set = DROAmbiguitySet(
        scenario_name, stage_2.optimization_problem, stage_2.delta_index, dro_data_set, stage_2.delta_disturbances
    )
    dro_gamma = 1e0 * dro_ambiguity_set.gamma
    dro_delta_lower_bound = 1e0 * dro_ambiguity_set.delta_lower_bound
    dro_delta_upper_bound = 1e0 * dro_ambiguity_set.delta_upper_bound
    dro_epsilon_upper_bound = 1e0 * dro_ambiguity_set.epsilon_upper_bound

    # Define variables.
    uncertainty_index = price_categories + stage_2.delta_disturbances.to_list()
    timesteps = stage_1.der_model_set.timesteps
    optimization_problem.define_variable(
        "stage_1_vector",
        stage_1_index=range(len(stage_1.optimization_problem.variables)),
    )
    optimization_problem.define_variable("alpha", uncertainty_index=uncertainty_index, timestep=timesteps)
    optimization_problem.define_variable("beta", uncertainty_index=uncertainty_index, timestep=timesteps)
    optimization_problem.define_variable("sigma", index=[0])
    optimization_problem.define_variable("k_vector_stage_2", stage_2_index=range(len(stage_2.stage_2_index)))
    optimization_problem.define_variable(
        "k_matrix_stage_2_delta",
        uncertainty_index=uncertainty_index,
        timestep=timesteps,
        stage_2_index=range(len(stage_2.stage_2_index)),
    )
    optimization_problem.define_variable(
        "k_matrix_stage_2_epsilon",
        uncertainty_index=uncertainty_index,
        timestep=timesteps,
        stage_2_index=range(len(stage_2.stage_2_index)),
    )

    # Dual variables: Objective.
    optimization_problem.define_variable(
        "mu",
        part=["objective"],
        dual_index=["1", "2", "3a", "3b", "4"],
        uncertainty_index=uncertainty_index,
        timestep=timesteps,
    )
    optimization_problem.define_variable(
        "nu",
        part=["objective"],
        dual_index=["1", "2", "3", "4"],
        uncertainty_index=uncertainty_index,
        timestep=timesteps,
    )

    # Dual variables: Stage 2.
    optimization_problem.define_variable(
        "mu",
        part=["stage_2"],
        dual_index=["1", "2", "3a", "3b", "4"],
        constraint_index=range(len(stage_2.t_vector)),
        uncertainty_index=uncertainty_index,
        timestep=timesteps,
    )
    optimization_problem.define_variable(
        "nu",
        part=["stage_2"],
        dual_index=["1", "2", "3", "4"],
        constraint_index=range(len(stage_2.t_vector)),
        uncertainty_index=uncertainty_index,
        timestep=timesteps,
    )

    # Testing variable alignment.
    # print(
    #     optimization_problem.variables.loc[mesmo.utils.get_index(
    #         optimization_problem.variables,
    #         name='mu', part='stage_2', dual_index='1',
    #     )].dropna(axis='columns', how='all')
    # )
    # print(
    #     optimization_problem.variables.loc[mesmo.utils.get_index(
    #         optimization_problem.variables,
    #         name='mu', part='objective', dual_index='1'
    #     )].dropna(axis='columns', how='all')
    # )
    # print(
    #     optimization_problem.variables.loc[mesmo.utils.get_index(
    #         optimization_problem.variables,
    #         name='alpha'
    #     )].dropna(axis='columns', how='all')
    # )
    # print(
    #     optimization_problem.variables.loc[mesmo.utils.get_index(
    #         optimization_problem.variables,
    #         name='k_matrix_stage_2_delta'
    #     )].dropna(axis='columns', how='all')
    # )

    optimization_problem.define_constraint(
        ("variable", stage_1.r_matrix_1_stage_1, dict(name="stage_1_vector")), "<=", ("constant", stage_1.t_vector_1)
    )

    optimization_problem.define_constraint(("variable", 1.0, dict(name="beta")), ">=", ("constant", 0.0))

    mesmo.utils.logger.info("Define linear constraints: Objective-A")
    optimization_problem.define_constraint(
        ("variable", stage_2.w_vector_2_stage_2, dict(name="k_vector_stage_2")),
        ("variable", 1.0, dict(name="sigma")),
        ">=",
        ("variable", np.array([1.0 + dro_delta_lower_bound.values]), dict(name="mu", part="objective", dual_index="1")),
        ("variable", np.array([1.0 - dro_delta_lower_bound.values]), dict(name="nu", part="objective", dual_index="1")),
        ("variable", np.array([1.0 - dro_delta_upper_bound.values]), dict(name="mu", part="objective", dual_index="2")),
        ("variable", np.array([1.0 + dro_delta_upper_bound.values]), dict(name="nu", part="objective", dual_index="2")),
        ("variable", 0.5 * np.ones((1, len(stage_2.delta_index))), dict(name="mu", part="objective", dual_index="3a")),
        ("variable", 0.5 * np.ones((1, len(stage_2.delta_index))), dict(name="nu", part="objective", dual_index="3")),
        ("variable", np.array([dro_epsilon_upper_bound.values]), dict(name="nu", part="objective", dual_index="4")),
    )

    mesmo.utils.logger.info("Define linear constraints: Objective-B")
    delta_len = len(timesteps) * len(uncertainty_index)
    optimization_problem.define_constraint(
        ("variable", np.transpose(stage_2.w_matrix_2_stage_1_delta), dict(name="stage_1_vector")),
        ("variable", sp.block_diag([stage_2.w_vector_2_stage_2] * delta_len), dict(name="k_matrix_stage_2_delta")),
        ("variable", 1.0, dict(name="alpha")),
        "==",
        ("variable", -1.0, dict(name="mu", part="objective", dual_index="1")),
        ("variable", +1.0, dict(name="nu", part="objective", dual_index="1")),
        ("variable", +1.0, dict(name="mu", part="objective", dual_index="2")),
        ("variable", -1.0, dict(name="nu", part="objective", dual_index="2")),
        ("variable", 1.0, dict(name="mu", part="objective", dual_index="3b")),
    )

    mesmo.utils.logger.info("Define linear constraints: Objective-C")
    delta_len = len(timesteps) * len(uncertainty_index)
    optimization_problem.define_constraint(
        ("variable", sp.block_diag([stage_2.w_vector_2_stage_2] * delta_len), dict(name="k_matrix_stage_2_epsilon")),
        ("variable", 1.0, dict(name="beta")),
        "==",
        ("variable", -0.5, dict(name="mu", part="objective", dual_index="3a")),
        ("variable", 0.5, dict(name="nu", part="objective", dual_index="3")),
        ("variable", +1.0, dict(name="mu", part="objective", dual_index="4")),
    )

    mesmo.utils.logger.info("Define linear constraints: Stage 2-A")
    delta_len = len(timesteps) * len(uncertainty_index)
    constraint_len = len(stage_2.t_vector)
    optimization_problem.define_constraint(
        ("constant", stage_2.t_vector),
        ("variable", -1.0 * stage_2.r_matrix_2_stage_1, dict(name="stage_1_vector")),
        ("variable", -1.0 * stage_2.r_matrix_2_stage_2, dict(name="k_vector_stage_2")),
        ">=",
        (
            "variable",
            sp.block_diag([1.0 + np.array([dro_delta_lower_bound.values])] * constraint_len),
            dict(name="mu", part="stage_2", dual_index="1"),
        ),
        (
            "variable",
            sp.block_diag([1.0 - np.array([dro_delta_lower_bound.values])] * constraint_len),
            dict(name="nu", part="stage_2", dual_index="1"),
        ),
        (
            "variable",
            sp.block_diag([1.0 - np.array([dro_delta_upper_bound.values])] * constraint_len),
            dict(name="mu", part="stage_2", dual_index="2"),
        ),
        (
            "variable",
            sp.block_diag([1.0 + np.array([dro_delta_upper_bound.values])] * constraint_len),
            dict(name="nu", part="stage_2", dual_index="2"),
        ),
        (
            "variable",
            0.5 * sp.block_diag([np.ones((1, delta_len))] * constraint_len),
            dict(name="mu", part="stage_2", dual_index="3a"),
        ),
        (
            "variable",
            0.5 * sp.block_diag([np.ones((1, delta_len))] * constraint_len),
            dict(name="nu", part="stage_2", dual_index="3"),
        ),
        (
            "variable",
            1.0 * sp.block_diag([np.array([dro_epsilon_upper_bound.values])] * constraint_len),
            dict(name="nu", part="stage_2", dual_index="4"),
        ),
    )

    mesmo.utils.logger.info("Define linear constraints: Stage 2-B")
    delta_len = len(timesteps) * len(uncertainty_index)
    constraint_len = len(stage_2.t_vector)
    optimization_problem.define_constraint(
        ("constant", -1.0 * stage_2.r_matrix_2_delta.toarray().ravel()),
        (
            "variable",
            -1.0
            * (
                sp.vstack(
                    [
                        sp.block_diag([stage_2.r_matrix_2_stage_2[row, :]] * delta_len)
                        for row in range(np.shape(stage_2.r_matrix_2_stage_2)[0])
                    ]
                )
            ),
            dict(name="k_matrix_stage_2_delta"),
        ),
        "==",
        ("variable", -1.0 * sp.eye(delta_len * constraint_len), dict(name="mu", part="stage_2", dual_index="1")),
        ("variable", +1.0 * sp.eye(delta_len * constraint_len), dict(name="nu", part="stage_2", dual_index="1")),
        ("variable", +1.0 * sp.eye(delta_len * constraint_len), dict(name="mu", part="stage_2", dual_index="2")),
        ("variable", -1.0 * sp.eye(delta_len * constraint_len), dict(name="nu", part="stage_2", dual_index="2")),
        ("variable", 1.0 * sp.eye(delta_len * constraint_len), dict(name="mu", part="stage_2", dual_index="3b")),
    )

    mesmo.utils.logger.info("Define linear constraints: Stage 2-C")
    delta_len = len(timesteps) * len(uncertainty_index)
    constraint_len = len(stage_2.t_vector)
    optimization_problem.define_constraint(
        (
            "variable",
            -1.0
            * (
                sp.vstack(
                    [
                        sp.block_diag([stage_2.r_matrix_2_stage_2[row, :]] * delta_len)
                        for row in range(np.shape(stage_2.r_matrix_2_stage_2)[0])
                    ]
                )
            ),
            dict(name="k_matrix_stage_2_epsilon"),
        ),
        "==",
        ("variable", -0.5 * sp.eye(delta_len * constraint_len), dict(name="mu", part="stage_2", dual_index="3a")),
        ("variable", +0.5 * sp.eye(delta_len * constraint_len), dict(name="nu", part="stage_2", dual_index="3")),
        ("variable", +1.0 * sp.eye(delta_len * constraint_len), dict(name="mu", part="stage_2", dual_index="4")),
    )

    optimization_problem.define_objective(
        ("variable", -1.0 * stage_1.w_vector_1_stage_1, dict(name="stage_1_vector")),
        ("variable", 1.0 * dro_gamma.T, dict(name="beta")),
        ("variable", 1.0, dict(name="sigma")),
    )

    mesmo.utils.logger.info("Get Gurobi problem.")
    gurobipy_problem, x_vector, constraints, objective = optimization_problem.get_gurobi_problem()

    mesmo.utils.logger.info("Define SOC constraints: Part 1 of 2")
    mu_index = mesmo.utils.get_index(optimization_problem.variables, name="mu", dual_index=["1", "2", "4"])
    nu_index = mesmo.utils.get_index(optimization_problem.variables, name="nu", dual_index=["1", "2", "4"])
    gurobipy_problem.addConstr(x_vector[nu_index] >= 0)
    soc_constraints_1 = [
        f" soc_constraints_1[{soc_i}]:"
        f" [ x_vector[{mu_i}] * x_vector[{mu_i}] - x_vector[{nu_i}] * x_vector[{nu_i}] ] <= 0"
        for soc_i, mu_i, nu_i in tqdm(zip(range(len(nu_index)), mu_index, nu_index), total=len(nu_index))
    ]

    mesmo.utils.logger.info("Define SOC constraints: Part 2 of 2")
    mu_index_a = mesmo.utils.get_index(optimization_problem.variables, name="mu", dual_index=["3a"])
    mu_index_b = mesmo.utils.get_index(optimization_problem.variables, name="mu", dual_index=["3b"])
    nu_index = mesmo.utils.get_index(optimization_problem.variables, name="nu", dual_index=["3"])
    gurobipy_problem.addConstr(x_vector[nu_index] >= 0)
    soc_constraints_2 = [
        f" soc_constraints_2[{soc_i}]:"
        f" [ x_vector[{mu_ia}] * x_vector[{mu_ia}] + x_vector[{mu_ib}] * x_vector[{mu_ib}]"
        f" - x_vector[{nu_i}] * x_vector[{nu_i}] ] <= 0"
        for soc_i, mu_ia, mu_ib, nu_i in tqdm(
            zip(range(len(nu_index)), mu_index_a, mu_index_b, nu_index), total=len(nu_index)
        )
    ]

    mesmo.utils.logger.info("Modify Gurobi problem.")
    gurobipy_problem.write(str(results_path / "gurobi.lp"))
    gurobi_lp_file = pd.read_csv((results_path / "gurobi.lp"), header=None).iloc[:, 0]
    bounds_line = gurobi_lp_file.index[gurobi_lp_file.str.contains("Bounds")][0]
    gurobi_lp_file = pd.concat(
        [
            gurobi_lp_file.iloc[:bounds_line],
            pd.Series(soc_constraints_1),
            pd.Series(soc_constraints_2),
            gurobi_lp_file.iloc[bounds_line:],
        ],
        ignore_index=True,
    )
    gurobi_lp_file.to_csv((results_path / "gurobi.lp"), header=False, index=False, quoting=csv.QUOTE_NONE)
    gurobipy_problem = gp.read((results_path / "gurobi.lp"))
    x_vector = gp.MVar(gurobipy_problem.getVars())
    x_vector = (
        # Sort x_vector entries, because they are unordered after modifying the Gurobi problem.
        x_vector[
            pd.Series(x_vector.getAttr("VarName"))
            .str.replace("]", "")
            .str.split("[", expand=True)
            .iloc[:, 1]
            .astype(int)
            .sort_values()
            .index
        ]
    )
    constraints = gp.MConstr(gurobipy_problem.getConstrs()[: constraints.shape[0]])
    objective = gurobipy_problem.getObjective()

    mesmo.utils.logger.info("Solve problem.")
    # gurobipy_problem.setParam('QCPDual', 1)  # Activate duals for QCP and SOCP.
    gurobipy_problem.setParam("BarQCPConvTol", 1e-6)
    gurobipy_problem.setParam("BarHomogeneous", 1)
    optimization_problem.solve_gurobi(gurobipy_problem, x_vector, constraints, objective)

    # Obtain results.
    x_vector_value = optimization_problem.x_vector
    stage_1_index = mesmo.utils.get_index(optimization_problem.variables, name="stage_1_vector")
    pd.DataFrame(x_vector_value[stage_1_index]).to_csv((results_path / f"s_1_vector_dro.csv"))
    stage_1_results = stage_1.optimization_problem.get_results(x_vector_value[stage_1_index])
    energy_stage_1 = stage_1_results["energy_stage_1"]
    up_reserve_stage_1 = stage_1_results["up_reserve_stage_1"]
    down_reserve_stage_1 = stage_1_results["down_reserve_stage_1"]

    #
    pd.DataFrame(energy_stage_1).to_csv((results_path / f"energy_dro.csv"))
    pd.DataFrame(up_reserve_stage_1).to_csv((results_path / f"up_reserve_dro.csv"))
    pd.DataFrame(down_reserve_stage_1).to_csv((results_path / f"down_reserve_dro.csv"))
    objective_dro = {"objective_value": [optimization_problem.objective]}
    objective_dro_df = pd.DataFrame(data=objective_dro)
    objective_dro_df.to_csv((results_path / f"objective_dro.csv"))

    # Plot some results.
    figure = go.Figure()
    figure.add_scatter(
        x=energy_stage_1.index,
        y=energy_stage_1.values,
        name="no_reserve",
        line=go.scatter.Line(shape="hv", width=5, dash="dot"),
    )
    figure.add_scatter(
        x=up_reserve_stage_1.index,
        y=up_reserve_stage_1.values,
        name="up_reserve",
        line=go.scatter.Line(shape="hv", width=4, dash="dot"),
    )
    figure.add_scatter(
        x=down_reserve_stage_1.index,
        y=down_reserve_stage_1.values,
        name="down_reserve",
        line=go.scatter.Line(shape="hv", width=3, dash="dot"),
    )
    figure.add_scatter(
        x=up_reserve_stage_1.index,
        y=(energy_stage_1 + up_reserve_stage_1).values,
        name="no_reserve + up_reserve",
        line=go.scatter.Line(shape="hv", width=2, dash="dot"),
    )
    figure.add_scatter(
        x=up_reserve_stage_1.index,
        y=(energy_stage_1 - down_reserve_stage_1).values,
        name="no_reserve - down_reserve",
        line=go.scatter.Line(shape="hv", width=1, dash="dot"),
    )
    figure.update_layout(
        title=f"Power balance",
        xaxis=go.layout.XAxis(tickformat="%H:%M"),
        legend=go.layout.Legend(x=0.01, xanchor="auto", y=0.99, yanchor="auto"),
    )
    # figure.show()
    mesmo.utils.write_figure_plotly(figure, (results_path / f"0_power_balance"))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
