"""MESMO tutorial: Example 3."""

import numpy as np

import mesmo


def main():

    # Obtain random parameter matrix.
    dimension = 1000
    parameter_matrix = np.random.rand(dimension, dimension)

    # Instantiate optimization problem.
    optimization_problem = mesmo.solutions.OptimizationProblem()

    # Define optimization parameters.
    optimization_problem.define_parameter("parameter_matrix", parameter_matrix)

    # Define optimization variables.
    optimization_problem.define_variable("a_vector", a_index=range(dimension))
    optimization_problem.define_variable("b_vector", b_index=range(dimension))

    # Define optimization constraints.
    optimization_problem.define_constraint(
        ("variable", 1.0, dict(name="b_vector")),
        "==",
        ("variable", "parameter_matrix", dict(name="a_vector")),
    )
    optimization_problem.define_constraint(
        ("constant", -10.0),
        "<=",
        ("variable", 1.0, dict(name="a_vector")),
    )
    optimization_problem.define_constraint(
        ("constant", +10.0),
        ">=",
        ("variable", 1.0, dict(name="a_vector")),
    )

    # Define optimization objective.
    optimization_problem.define_objective(("variable", 1.0, dict(name="b_vector")))

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = optimization_problem.get_results()
    a_vector = results["a_vector"]
    b_vector = results["b_vector"]


if __name__ == "__main__":
    main()
