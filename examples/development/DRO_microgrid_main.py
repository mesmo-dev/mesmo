"""Example script for DRO problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge
from mg_offer_stage_1_problem_standard_form import stage_1_problem_standard_form
from mg_offer_stage_2_problem_standard_form import stage_2_problem_standard_form
from mg_offer_stage_3_problem_standard_form import stage_3_problem_standard_form

def main():
    scenario_name = 'singapore_6node'
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Get results path.
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # initialize all three stage problem
    standard_form_stage_1, a_matrix, b_vector, f_vector, stochastic_scenarios, der_model_set \
        = stage_1_problem_standard_form()

    standard_form_stage_2, A2_matrix, B2_matrix, C2_matrix, M_Q2_delta, m_Q2_s2 \
        = stage_2_problem_standard_form()

    standard_form_stage_3, A3_matrix, B3_matrix, C3_matrix, D3_matrix, m_Q3_s2, m_Q3_s3 \
        = stage_3_problem_standard_form()

    # Instantiate optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define optimization problem.
    optimization_problem.x_vector = cp.Variable((len(standard_form_stage_1.variables), 1))
    optimization_problem.constraints.append(
        a_matrix.toarray() @ optimization_problem.x_vector <= b_vector
    )
    optimization_problem.objective += (
        (
                f_vector.T
                @ optimization_problem.x_vector
        )
    )
    # Define optimization objective

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = standard_form_stage_1.get_results(optimization_problem.x_vector)


if __name__ == '__main__':
    main()
