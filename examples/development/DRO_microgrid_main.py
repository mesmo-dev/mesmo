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
    scenario_name = 'singapore_6node_custom'
    # scenario_name = 'singapore_6node'

    price_categories = ['energy', 'up_reserve', 'down_reserve']
    ambiguity_set_dual_variables_categories = ['mu_1', 'nu_1', 'mu_2', 'nu_2', 'mu_3', 'nu_3', 'mu_4', 'nu_4']

    # Get results path.
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    fledge.data_interface.recreate_database()

    price_data = fledge.data_interface.PriceData(scenario_name)

    # initialize all three stage problem
    standard_form_stage_1, A1_matrix, b1_vector, f_vector, stochastic_scenarios, der_model_set \
        = stage_1_problem_standard_form(scenario_name)

    standard_form_stage_2, b2_vector, A2_matrix, B2_matrix, C2_matrix, M_Q2_delta, m_Q2_s2, s2_indices_stage2, \
        delta_indices_stage2, s1_indices = stage_2_problem_standard_form(scenario_name)

    standard_form_stage_3, b3_vector, A3_matrix, B3_matrix, C3_matrix, D3_matrix, m_Q3_s2, m_Q3_s3, \
        delta_indices_stage3, s1_indices_stage3, s2_indices_stage3, s3_indices_stage3 = stage_3_problem_standard_form(scenario_name)

    # Instantiate optimization problem.
    optimization_problem_dro = fledge.utils.OptimizationProblem()

    print('define DRO variable')

    # TODO: we need proper initialization of DRO data
    # constants
    optimization_problem_dro.gamma = 30*np.zeros((len(delta_indices_stage2), 1))

    optimization_problem_dro.delta_lower_bound = - 10 * np.ones((len(delta_indices_stage2), 1))

    optimization_problem_dro.delta_upper_bound = 10 * np.ones((len(delta_indices_stage2), 1))

    optimization_problem_dro.u_upper_bound = 1000 * np.ones((len(delta_indices_stage2), 1))

    # Define optimization problem variables
    optimization_problem_dro.s1_vector = cp.Variable((len(standard_form_stage_1.variables), 1))

    optimization_problem_dro.alpha = cp.Variable((len(delta_indices_stage2), 1))

    optimization_problem_dro.beta = cp.Variable((len(delta_indices_stage2), 1))

    optimization_problem_dro.sigma = cp.Variable((1, 1))

    optimization_problem_dro.k_0_s2 = cp.Variable((s2_indices_stage2.shape[0], 1))

    optimization_problem_dro.K_delta_s2 = cp.Variable((s2_indices_stage2.shape[0], delta_indices_stage2.shape[0]))

    optimization_problem_dro.K_u_s2 = cp.Variable((s2_indices_stage2.shape[0], delta_indices_stage2.shape[0]))

    optimization_problem_dro.k_0_s3 = cp.Variable((s3_indices_stage3.shape[0], 1))

    optimization_problem_dro.K_delta_s3 = cp.Variable((s3_indices_stage3.shape[0], delta_indices_stage2.shape[0]))

    optimization_problem_dro.K_u_s3 = cp.Variable((s3_indices_stage3.shape[0], delta_indices_stage2.shape[0]))

    # TODO: define mu nu with standard_form_stage_2, need a function to get the index and length of the variable...

    optimization_problem_dro.ambiguity_set_duals_uncertain_price = {}

    for price_category in price_categories:
        for time_step in der_model_set.timesteps:
            for dual_category in ambiguity_set_dual_variables_categories:
                if dual_category == 'mu_3':
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price[
                        price_category, time_step, dual_category
                    ] = cp.Variable((2, 1))
                else:
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price[
                        price_category, time_step, dual_category
                    ] = cp.Variable((1, 1))

    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance = {}

    for der_name, der_model in der_model_set.flexible_der_models.items():
        if not der_model.disturbances.empty:
            for time_step in der_model_set.timesteps:

                index_temp_der_disturbance = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                    timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                )

                size_der_disturbance = index_temp_der_disturbance.shape[0]

                for dual_category in ambiguity_set_dual_variables_categories:
                    if dual_category == 'mu_3':
                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                            der_name, time_step, dual_category
                        ] = cp.Variable((2 * size_der_disturbance, 1))

                    else:
                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                            der_name, time_step, dual_category
                        ] = cp.Variable((size_der_disturbance, 1))

    print('define constr 32b')
    # constr 32b)
    optimization_problem_dro.constraints.append(
        A1_matrix.toarray() @ optimization_problem_dro.s1_vector <= b1_vector
    )

    print('define constr 32d')
    # constr 32d)
    optimization_problem_dro.constraints.append(
        optimization_problem_dro.beta >= 0
    )

    # optimization_problem_dro.constraints.append(
    #     optimization_problem_dro.k_0_s2 == 0
    # )
    #
    # optimization_problem_dro.constraints.append(
    #     optimization_problem_dro.k_0_s3 == 0
    # )


    print('define constr 35 (a-d)')
    # constr 35a)
    temp_35a = np.transpose((m_Q2_s2 + m_Q3_s2)) @ optimization_problem_dro.k_0_s2 + \
        np.transpose((m_Q3_s3)) @ optimization_problem_dro.k_0_s3 + optimization_problem_dro.sigma

    for price_category in price_categories:
        for time_step in der_model_set.timesteps:
            if price_category == 'energy':
                index_temp_uncertain_prices = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_energy_price_deviation_s2', timestep=time_step
                )
            elif price_category == 'up_reserve':
                index_temp_uncertain_prices = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_up_reserve_price_deviation_s2', timestep=time_step
                )
            else:
                index_temp_uncertain_prices = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_down_reserve_price_deviation_s2', timestep=time_step
                )

            temp_35a -= (
                            optimization_problem_dro.delta_lower_bound[
                            np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                        ) @ \
                        (
                            optimization_problem_dro.ambiguity_set_duals_uncertain_price[
                            price_category, time_step, 'mu_1']
                        )

            temp_35a -= (
                             -optimization_problem_dro.delta_lower_bound[
                             np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                        ) @ \
                        (
                            optimization_problem_dro.ambiguity_set_duals_uncertain_price[
                            price_category, time_step, 'nu_1']
                        )

            temp_35a -= (
                                -optimization_problem_dro.delta_upper_bound[
                                np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                        ) @ \
                        (
                            optimization_problem_dro.ambiguity_set_duals_uncertain_price[
                                price_category, time_step, 'mu_2']
                        )

            temp_35a -= (
                            optimization_problem_dro.delta_upper_bound[
                                np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                        ) @ \
                        (
                            optimization_problem_dro.ambiguity_set_duals_uncertain_price[
                                price_category, time_step, 'nu_2']
                        )

            temp_35a -= np.array([[0.5, 0]]) @ optimization_problem_dro.ambiguity_set_duals_uncertain_price[
                                price_category, time_step, 'mu_3']

            temp_35a -= 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_price[
                                price_category, time_step, 'nu_3']

            temp_35a -= (
                            optimization_problem_dro.u_upper_bound[
                                np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0
                            ]
                        ) @ \
                        (
                            optimization_problem_dro.ambiguity_set_duals_uncertain_price[
                                price_category, time_step, 'nu_4']
                        )


    for der_name, der_model in der_model_set.flexible_der_models.items():
        if not der_model.disturbances.empty:
            for time_step in der_model_set.timesteps:

                index_temp_der_disturbance = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                    timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                )

                size_der_disturbance = index_temp_der_disturbance.shape[0]
                #  print(size_der_disturbance)

                if size_der_disturbance == 1:
                    temp_35a -= (
                                        optimization_problem_dro.delta_lower_bound[
                                            np.where(pd.Index(delta_indices_stage2).isin(
                                                index_temp_der_disturbance)), 0] + 1
                                ) @ \
                                (
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'mu_1'
                                    ]
                                )

                    temp_35a -= (
                                        -optimization_problem_dro.delta_lower_bound[
                                            np.where(pd.Index(delta_indices_stage2).isin(
                                                index_temp_der_disturbance)), 0] + 1
                                ) @ \
                                (
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'nu_1'
                                    ]
                                )

                    temp_35a -= (
                                        -optimization_problem_dro.delta_upper_bound[
                                            np.where(pd.Index(delta_indices_stage2).isin(
                                                index_temp_der_disturbance)), 0] + 1
                                ) @ \
                                (
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'mu_2'
                                    ]
                                )

                    temp_35a -= (
                                        optimization_problem_dro.delta_upper_bound[
                                            np.where(pd.Index(delta_indices_stage2).isin(
                                                index_temp_der_disturbance)), 0] + 1
                                ) @ \
                                (
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'nu_2'
                                    ]
                                )

                    temp_35a -= np.array([[0.5, 0]]) @ \
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'mu_3'
                                    ]

                    temp_35a -= 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'nu_3'
                                    ]

                    temp_35a -= (
                                    optimization_problem_dro.u_upper_bound[
                                        np.where(pd.Index(delta_indices_stage2).isin(index_temp_der_disturbance)), 0
                                    ]
                                ) @ \
                                (
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'nu_4'
                                    ]
                                )
                else:
                    for index_within_der_disturbance in index_temp_der_disturbance:
                        temp_35a -= (
                                            optimization_problem_dro.delta_lower_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    [index_within_der_disturbance])), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                            der_name, time_step, 'mu_1'
                                        ][
                                            np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))
                                         ]
                                    )

                        temp_35a -= (
                                            -optimization_problem_dro.delta_lower_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    [index_within_der_disturbance])), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                            der_name, time_step, 'nu_1'
                                        ][
                                            np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))
                                         ]
                                    )

                        temp_35a -= (
                                            -optimization_problem_dro.delta_upper_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    [index_within_der_disturbance])), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                            der_name, time_step, 'mu_2'
                                        ][
                                            np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))
                                         ]
                                    )

                        temp_35a -= (
                                            optimization_problem_dro.delta_upper_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    [index_within_der_disturbance])), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                            der_name, time_step, 'nu_2'
                                        ][
                                            np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))
                                         ]
                                    )

                        temp_35a -= np.array([[0.5, 0]]) @ \
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'mu_3'
                                    ][
                                        [
                                            2*np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))[0].item(),
                                            2*np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))[0].item()+1
                                        ], :
                                    ]

                        temp_35a -= 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                            der_name, time_step, 'nu_3'
                            ][
                                np.where(pd.Index(index_temp_der_disturbance).isin([index_within_der_disturbance]))
                             ]

                        temp_35a -= (
                                        optimization_problem_dro.u_upper_bound[
                                            np.where(pd.Index(delta_indices_stage2).isin([index_within_der_disturbance])),0
                                        ]
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                            der_name, time_step, 'nu_4'
                                        ][
                                            np.where(pd.Index(index_temp_der_disturbance).isin([index_within_der_disturbance]))
                                        ]

                                    )

    optimization_problem_dro.constraints.append(
        temp_35a >= 0
    )

#  constr 35b)
    temp_35b = optimization_problem_dro.s1_vector.T @ M_Q2_delta + \
               np.transpose(m_Q2_s2 + m_Q3_s2) @ optimization_problem_dro.K_delta_s2 + \
               np.transpose(m_Q3_s3) @ optimization_problem_dro.K_delta_s3 + \
               optimization_problem_dro.alpha.T

    for price_category in price_categories:
        for time_step in der_model_set.timesteps:
            if price_category == 'energy':
                index_temp_uncertain_prices = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_energy_price_deviation_s2', timestep=time_step
                )
            elif price_category == 'up_reserve':
                index_temp_uncertain_prices = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_up_reserve_price_deviation_s2',
                    timestep=time_step
                )
            else:
                index_temp_uncertain_prices = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_down_reserve_price_deviation_s2',
                    timestep=time_step
                )

            optimization_problem_dro.constraints.append(
                - optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'mu_1']
                + optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'nu_1']
                + optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'mu_2']
                - optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'nu_2']
                + np.array([[0.5, 0]]) @ optimization_problem_dro.ambiguity_set_duals_uncertain_price[
                    price_category, time_step, 'mu_3']
                ==
                temp_35b[0, np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices))]
            )

    for der_name, der_model in der_model_set.flexible_der_models.items():
        if not der_model.disturbances.empty:
            for time_step in der_model_set.timesteps:

                index_temp_der_disturbance = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                    timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                )

                size_der_disturbance = index_temp_der_disturbance.shape[0]

                if size_der_disturbance == 1:
                    optimization_problem_dro.constraints.append(
                        - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                            der_name, time_step, 'mu_1']
                        + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                            der_name, time_step, 'nu_1']
                        + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                            der_name, time_step, 'mu_2']
                        - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                            der_name, time_step, 'nu_2']
                        + np.array([[0.5, 0]]) @ optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                            der_name, time_step, 'mu_3']
                        ==
                        temp_35b[0, np.where(pd.Index(delta_indices_stage2).isin(index_temp_der_disturbance))]
                    )
                else:
                    for index_within_der_disturbance in index_temp_der_disturbance:
                        optimization_problem_dro.constraints.append(
                            - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                der_name, time_step, 'mu_1'][
                                            np.where(pd.Index(index_temp_der_disturbance).isin(
                                                [index_within_der_disturbance]))
                                        ]
                            + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                der_name, time_step, 'nu_1'][
                                            np.where(pd.Index(index_temp_der_disturbance).isin(
                                                [index_within_der_disturbance]))
                                        ]
                            + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                der_name, time_step, 'mu_2'][
                                            np.where(pd.Index(index_temp_der_disturbance).isin(
                                                [index_within_der_disturbance]))
                                        ]
                            - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                der_name, time_step, 'nu_2'][
                                            np.where(pd.Index(index_temp_der_disturbance).isin(
                                                [index_within_der_disturbance]))
                                        ]
                            + np.array([[0.0, 1.0]]) @
                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                der_name, time_step, 'mu_3'][
                                        [
                                            2*np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))[0].item(),
                                            2*np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))[0].item()+1
                                        ], :
                                    ]
                            ==
                            temp_35b[0, np.where(pd.Index(delta_indices_stage2).isin([index_within_der_disturbance]))]
                        )

    #  constr 35 c) & 35 d)
    temp_35c = np.transpose(m_Q2_s2 + m_Q3_s2) @ optimization_problem_dro.K_u_s2 + \
               np.transpose(m_Q3_s3) @ optimization_problem_dro.K_u_s3 + \
               optimization_problem_dro.beta.T

    for price_category in price_categories:
        for time_step in der_model_set.timesteps:
            if price_category == 'energy':
                index_temp_uncertain_prices = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_energy_price_deviation_s2',
                    timestep=time_step
                )
            elif price_category == 'up_reserve':
                index_temp_uncertain_prices = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_up_reserve_price_deviation_s2',
                    timestep=time_step
                )
            else:
                index_temp_uncertain_prices = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_down_reserve_price_deviation_s2',
                    timestep=time_step
                )
            #  constr 35 c)
            optimization_problem_dro.constraints.append(
                + np.array([[-0.5, 0]]) @ optimization_problem_dro.ambiguity_set_duals_uncertain_price[
                    price_category, time_step, 'mu_3']
                + 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'nu_3']
                + optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'mu_4']
                ==
                temp_35c[0, np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices))]
            )

            #  constr 35 d)
            optimization_problem_dro.constraints.append(
                cp.SOC(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'nu_1'][0],
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'mu_1'][0]
                )
            )

            optimization_problem_dro.constraints.append(
                cp.SOC(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'nu_2'][0],
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'mu_2'][0]
                )
            )

            optimization_problem_dro.constraints.append(
                cp.SOC(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'nu_3'][0],
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'mu_3'][:, 0]
                )
            )

            optimization_problem_dro.constraints.append(
                cp.SOC(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'nu_4'][0],
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'mu_4'][0]
                )
            )

            # constr 35e
            optimization_problem_dro.constraints.append(
                optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'nu_1'][0] >= 0
            )

            optimization_problem_dro.constraints.append(
                optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'nu_2'][0] >= 0
            )

            optimization_problem_dro.constraints.append(
                optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'nu_3'][0] >= 0
            )

            optimization_problem_dro.constraints.append(
                optimization_problem_dro.ambiguity_set_duals_uncertain_price[price_category, time_step, 'nu_4'][0] >= 0
            )

    for der_name, der_model in der_model_set.flexible_der_models.items():
        if not der_model.disturbances.empty:
            for time_step in der_model_set.timesteps:

                # constr 35e
                optimization_problem_dro.constraints.append(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[der_name, time_step, 'nu_1'][
                        0] >= 0
                )

                optimization_problem_dro.constraints.append(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[der_name, time_step, 'nu_2'][
                        0] >= 0
                )

                optimization_problem_dro.constraints.append(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[der_name, time_step, 'nu_3'][
                        0] >= 0
                )

                optimization_problem_dro.constraints.append(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[der_name, time_step, 'nu_4'][
                        0] >= 0
                )

                index_temp_der_disturbance = fledge.utils.get_index(
                    standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                    timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                )

                size_der_disturbance = index_temp_der_disturbance.shape[0]
                #  constr 35 c)
                if size_der_disturbance == 1:
                    optimization_problem_dro.constraints.append(
                        np.array([[-0.5, 0]]) @
                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                            der_name, time_step, 'mu_3']
                        + 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                            der_name, time_step, 'nu_3']
                        + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                            der_name, time_step, 'mu_4']
                        ==
                        temp_35c[0, np.where(pd.Index(delta_indices_stage2).isin(index_temp_der_disturbance))]
                    )

                    #  constr 35 d)
                    optimization_problem_dro.constraints.append(
                        cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'nu_1'][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'mu_1'][0]
                        )
                    )

                    optimization_problem_dro.constraints.append(
                        cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'nu_2'][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'mu_2'][0]
                        )
                    )

                    optimization_problem_dro.constraints.append(
                        cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'nu_3'][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'mu_3'][:, 0])
                    )

                    optimization_problem_dro.constraints.append(
                        cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'nu_4'][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                        der_name, time_step, 'mu_4'][0]
                        )
                    )

                else:
                    #  constr 35 c)
                    for index_within_der_disturbance in index_temp_der_disturbance:
                        optimization_problem_dro.constraints.append(
                            + np.array([[-0.5, 0]]) @
                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                der_name, time_step, 'mu_3'][
                                [
                                    2*np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))[0].item(),
                                    2*np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))[0].item() + 1
                                ], :
                            ]
                            + 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                der_name, time_step, 'nu_3'][
                                np.where(pd.Index(index_temp_der_disturbance).isin(
                                    [index_within_der_disturbance]))
                            ]
                            + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                der_name, time_step, 'mu_4'][
                                np.where(pd.Index(index_temp_der_disturbance).isin(
                                    [index_within_der_disturbance]))
                            ]
                            ==
                            temp_35c[
                                0, np.where(pd.Index(delta_indices_stage2).isin([index_within_der_disturbance]))]
                        )

                        #  constr 35 d)
                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                    der_name, time_step, 'nu_1'][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                    der_name, time_step, 'mu_1'][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ][0]
                            )
                        )

                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                    der_name, time_step, 'nu_2'][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                    der_name, time_step, 'mu_2'][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ][0]
                            )
                        )

                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                    der_name, time_step, 'nu_3'][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                    der_name, time_step, 'mu_3'][
                                        [
                                            2*np.where(pd.Index(index_temp_der_disturbance).isin(
                                                [index_within_der_disturbance]))[0].item(),
                                            2*np.where(pd.Index(index_temp_der_disturbance).isin(
                                                [index_within_der_disturbance]))[0].item() + 1
                                        ], :
                                    ][:, 0]
                            )
                        )

                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                    der_name, time_step, 'nu_4'][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance[
                                    der_name, time_step, 'mu_4'][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ][0]
                            )
                        )

    print('define constr 40 a')

    optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40 = {}
    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40 = {}

    for row_index in range(b2_vector.shape[0]):
        for price_category in price_categories:
            for time_step in der_model_set.timesteps:
                for dual_category in ambiguity_set_dual_variables_categories:
                    if dual_category == 'mu_3':
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, dual_category, row_index
                        ] = cp.Variable((2, 1))
                    else:
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, dual_category, row_index
                        ] = cp.Variable((1, 1))

        for der_name, der_model in der_model_set.flexible_der_models.items():
            if not der_model.disturbances.empty:
                for time_step in der_model_set.timesteps:

                    index_temp_der_disturbance = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                        timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                    )

                    size_der_disturbance = index_temp_der_disturbance.shape[0]

                    for dual_category in ambiguity_set_dual_variables_categories:
                        if dual_category == 'mu_3':
                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                der_name, time_step, dual_category, row_index
                            ] = cp.Variable((2 * size_der_disturbance, 1))

                        else:
                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                der_name, time_step, dual_category, row_index
                            ] = cp.Variable((size_der_disturbance, 1))


    # constr 40 a)

    temp_40a_orig = b2_vector - A2_matrix @ optimization_problem_dro.s1_vector - \
                    B2_matrix @ optimization_problem_dro.k_0_s2
    for row_index in range(b2_vector.shape[0]):

        temp_40a = temp_40a_orig[row_index, :]
        for price_category in price_categories:
            for time_step in der_model_set.timesteps:
                if price_category == 'energy':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_energy_price_deviation_s2', timestep=time_step
                    )
                elif price_category == 'up_reserve':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_up_reserve_price_deviation_s2', timestep=time_step
                    )
                else:
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_down_reserve_price_deviation_s2', timestep=time_step
                    )

                temp_40a -= (
                                optimization_problem_dro.delta_lower_bound[
                                np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                            ) @ \
                            (
                                optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                                price_category, time_step, 'mu_1', row_index]
                            )

                temp_40a -= (
                                 -optimization_problem_dro.delta_lower_bound[
                                 np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                            ) @ \
                            (
                                optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                                price_category, time_step, 'nu_1', row_index]
                            )

                temp_40a -= (
                                    -optimization_problem_dro.delta_upper_bound[
                                    np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                            ) @ \
                            (
                                optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                                    price_category, time_step, 'mu_2', row_index]
                            )

                temp_40a -= (
                                optimization_problem_dro.delta_upper_bound[
                                    np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                            ) @ \
                            (
                                optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                                    price_category, time_step, 'nu_2', row_index]
                            )

                temp_40a -= np.array([[0.5, 0]]) @ optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                                    price_category, time_step, 'mu_3', row_index]

                temp_40a -= 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                                    price_category, time_step, 'nu_3', row_index]

                temp_40a -= (
                                optimization_problem_dro.u_upper_bound[
                                    np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0
                                ]
                            ) @ \
                            (
                                optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                                    price_category, time_step, 'nu_4', row_index]
                            )

        for der_name, der_model in der_model_set.flexible_der_models.items():
            if not der_model.disturbances.empty:
                for time_step in der_model_set.timesteps:

                    index_temp_der_disturbance = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                        timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                    )

                    size_der_disturbance = index_temp_der_disturbance.shape[0]
                    #print(size_der_disturbance)

                    if size_der_disturbance == 1:
                        temp_40a -= (
                                            optimization_problem_dro.delta_lower_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    index_temp_der_disturbance)), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                            der_name, time_step, 'mu_1', row_index
                                        ]
                                    )

                        temp_40a -= (
                                            -optimization_problem_dro.delta_lower_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    index_temp_der_disturbance)), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                            der_name, time_step, 'nu_1', row_index
                                        ]
                                    )

                        temp_40a -= (
                                            -optimization_problem_dro.delta_upper_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    index_temp_der_disturbance)), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                            der_name, time_step, 'mu_2', row_index
                                        ]
                                    )

                        temp_40a -= (
                                            optimization_problem_dro.delta_upper_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    index_temp_der_disturbance)), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                            der_name, time_step, 'nu_2', row_index
                                        ]
                                    )

                        temp_40a -= np.array([[0.5, 0]]) @ \
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                            der_name, time_step, 'mu_3', row_index
                                        ]

                        temp_40a -= 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                            der_name, time_step, 'nu_3', row_index
                                        ]
                        # TODO check the np.where indices is correct: [np.where(pd.Index(s1_indices).isin(index_energy_stage_2))[0], :]
                        temp_40a -= (
                                        optimization_problem_dro.u_upper_bound[
                                            np.where(pd.Index(delta_indices_stage2).isin(index_temp_der_disturbance)), 0
                                        ]
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                            der_name, time_step, 'nu_4', row_index
                                        ]
                                    )
                    else:
                        for index_within_der_disturbance in index_temp_der_disturbance:
                            temp_40a -= (
                                                optimization_problem_dro.delta_lower_bound[
                                                    np.where(pd.Index(delta_indices_stage2).isin(
                                                        [index_within_der_disturbance])), 0] + 1
                                        ) @ \
                                        (
                                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                                der_name, time_step, 'mu_1', row_index
                                            ][
                                                np.where(pd.Index(index_temp_der_disturbance).isin(
                                                        [index_within_der_disturbance]))
                                             ]
                                        )

                            temp_40a -= (
                                                -optimization_problem_dro.delta_lower_bound[
                                                    np.where(pd.Index(delta_indices_stage2).isin(
                                                        [index_within_der_disturbance])), 0] + 1
                                        ) @ \
                                        (
                                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                                der_name, time_step, 'nu_1', row_index
                                            ][
                                                np.where(pd.Index(index_temp_der_disturbance).isin(
                                                        [index_within_der_disturbance]))
                                             ]
                                        )

                            temp_40a -= (
                                                -optimization_problem_dro.delta_upper_bound[
                                                    np.where(pd.Index(delta_indices_stage2).isin(
                                                        [index_within_der_disturbance])), 0] + 1
                                        ) @ \
                                        (
                                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                                der_name, time_step, 'mu_2', row_index
                                            ][
                                                np.where(pd.Index(index_temp_der_disturbance).isin(
                                                        [index_within_der_disturbance]))
                                             ]
                                        )

                            temp_40a -= (
                                                optimization_problem_dro.delta_upper_bound[
                                                    np.where(pd.Index(delta_indices_stage2).isin(
                                                        [index_within_der_disturbance])), 0] + 1
                                        ) @ \
                                        (
                                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                                der_name, time_step, 'nu_2', row_index
                                            ][
                                                np.where(pd.Index(index_temp_der_disturbance).isin(
                                                        [index_within_der_disturbance]))
                                             ]
                                        )

                            temp_40a -= np.array([[0.5, 0]]) @ \
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                            der_name, time_step, 'mu_3', row_index
                                        ][
                                            [
                                                2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))[0].item(),
                                                2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))[0].item() + 1
                                            ], :
                                        ]

                            temp_40a -= 0.5 * \
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                            der_name, time_step, 'nu_3', row_index
                                        ][
                                            np.where(pd.Index(index_temp_der_disturbance).isin(
                                                [index_within_der_disturbance]))
                                        ]

                            temp_40a -= (
                                            optimization_problem_dro.u_upper_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    [index_within_der_disturbance])), 0
                                            ]
                                        ) @ \
                                        (

                                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                                der_name, time_step, 'nu_4', row_index
                                            ][
                                                np.where(pd.Index(index_temp_der_disturbance).isin([index_within_der_disturbance]))
                                            ]
                                        )
        optimization_problem_dro.constraints.append(
            temp_40a >= 0
        )
    print('define constr 40 b')
    # constr 40 b)
    temp_40b = -C2_matrix - B2_matrix @ optimization_problem_dro.K_delta_s2

    for row_index in range(b2_vector.shape[0]):
        for price_category in price_categories:
            for time_step in der_model_set.timesteps:
                if price_category == 'energy':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_energy_price_deviation_s2',
                        timestep=time_step
                    )
                elif price_category == 'up_reserve':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_up_reserve_price_deviation_s2',
                        timestep=time_step
                    )
                else:
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_down_reserve_price_deviation_s2',
                        timestep=time_step
                    )

                optimization_problem_dro.constraints.append(
                    - optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                        price_category, time_step, 'mu_1', row_index]
                    + optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                        price_category, time_step, 'nu_1', row_index]
                    + optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                        price_category, time_step, 'mu_2', row_index]
                    - optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                        price_category, time_step, 'nu_2', row_index]
                    + np.array([[0.5, 0]]) @ optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                        price_category, time_step, 'mu_3', row_index]
                    ==
                    temp_40b[row_index, np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices))[0]]
                )

        for der_name, der_model in der_model_set.flexible_der_models.items():
            if not der_model.disturbances.empty:
                for time_step in der_model_set.timesteps:

                    index_temp_der_disturbance = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                        timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                    )

                    size_der_disturbance = index_temp_der_disturbance.shape[0]

                    if size_der_disturbance == 1:
                        optimization_problem_dro.constraints.append(
                            - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                der_name, time_step, 'mu_1', row_index]
                            + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                der_name, time_step, 'nu_1', row_index]
                            + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                der_name, time_step, 'mu_2', row_index]
                            - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                der_name, time_step, 'nu_2', row_index]
                            + np.array([[0.5, 0]])
                            @ optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                der_name, time_step, 'mu_3', row_index]
                            ==
                            temp_40b[row_index, np.where(pd.Index(delta_indices_stage2).isin(index_temp_der_disturbance))[0]
                            ]
                        )
                    else:
                        for index_within_der_disturbance in index_temp_der_disturbance:
                            optimization_problem_dro.constraints.append(
                                - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'mu_1', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'nu_1', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'mu_2', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'nu_2', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                + np.array([[0.0, 1.0]]) @
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'mu_3', row_index][
                                [
                                    2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))[0].item(),
                                    2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))[0].item() + 1
                                ], :
                                ]
                                ==
                                temp_40b[row_index, np.where(pd.Index(delta_indices_stage2).isin([index_within_der_disturbance]))[0]
                                ]
                            )
    print('define constr 40 c+d')
    #  constr 40 c)
    temp_40c = - B2_matrix @ optimization_problem_dro.K_u_s2

    for row_index in range(b2_vector.shape[0]):
        for price_category in price_categories:
            for time_step in der_model_set.timesteps:
                if price_category == 'energy':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_energy_price_deviation_s2',
                        timestep=time_step
                    )
                elif price_category == 'up_reserve':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_up_reserve_price_deviation_s2',
                        timestep=time_step
                    )
                else:
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_down_reserve_price_deviation_s2',
                        timestep=time_step
                    )
                #  constr 40 c)
                optimization_problem_dro.constraints.append(
                    + np.array([[-0.5, 0]]) @ optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                        price_category, time_step, 'mu_3', row_index]
                    + 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                        price_category, time_step, 'nu_3', row_index]
                    + optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                        price_category, time_step, 'mu_4', row_index]
                    ==
                    temp_40c[row_index, np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices))[0]
                    ]
                )

        for der_name, der_model in der_model_set.flexible_der_models.items():
            if not der_model.disturbances.empty:
                for time_step in der_model_set.timesteps:

                    index_temp_der_disturbance = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                        timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                    )

                    size_der_disturbance = index_temp_der_disturbance.shape[0]
                    #  constr 40 c)
                    if size_der_disturbance == 1:
                        optimization_problem_dro.constraints.append(
                            np.array([[-0.5, 0]]) @
                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                der_name, time_step, 'mu_3', row_index]
                            + 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                der_name, time_step, 'nu_3', row_index]
                            + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                der_name, time_step, 'mu_4', row_index]
                            ==
                            temp_40c[
                                row_index, np.where(pd.Index(delta_indices_stage2).isin(index_temp_der_disturbance))[0]
                            ]
                        )

                    else:
                        #  constr 40 c)
                        for index_within_der_disturbance in index_temp_der_disturbance:
                            optimization_problem_dro.constraints.append(
                                + np.array([[-0.5, 0]]) @
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'mu_3', row_index][
                                [
                                    2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))[0].item(),
                                    2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))[0].item() + 1
                                ], :
                                ]
                                + 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'nu_3', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'mu_4', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                ==
                                temp_40c[row_index,
                                         np.where(pd.Index(delta_indices_stage2).isin([index_within_der_disturbance]))[0]
                                ]
                            )

        #  constr 40 d)
        for price_category in price_categories:
            for time_step in der_model_set.timesteps:

                optimization_problem_dro.constraints.append(
                    cp.SOC(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'nu_1', row_index][0],
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'mu_1', row_index][0]
                    )
                )

                optimization_problem_dro.constraints.append(
                    cp.SOC(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'nu_2', row_index][0],
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'mu_2', row_index][0]
                    )
                )

                optimization_problem_dro.constraints.append(
                    cp.SOC(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'nu_3', row_index][0],
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'mu_3', row_index][:, 0]
                    )
                )

                optimization_problem_dro.constraints.append(
                    cp.SOC(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'nu_4', row_index][0],
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'mu_4', row_index][0]
                    )
                )

                # constr 40 e
                optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'nu_1', row_index] >= 0
                )

                optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'nu_2', row_index] >= 0
                )

                optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'nu_3', row_index] >= 0
                )

                optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_40[
                            price_category, time_step, 'nu_4', row_index] >= 0
                )

        for der_name, der_model in der_model_set.flexible_der_models.items():
            if not der_model.disturbances.empty:
                for time_step in der_model_set.timesteps:

                    # constr 40e
                    optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                            der_name, time_step, 'nu_1', row_index] >= 0
                    )

                    optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                            der_name, time_step, 'nu_2', row_index] >= 0
                    )

                    optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                            der_name, time_step, 'nu_3', row_index] >= 0
                    )

                    optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                            der_name, time_step, 'nu_4', row_index] >= 0
                    )

                    index_temp_der_disturbance = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                        timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                    )

                    size_der_disturbance = index_temp_der_disturbance.shape[0]

                    if size_der_disturbance == 1:
                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'nu_1', row_index][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'mu_1', row_index][0]
                            )
                        )

                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'nu_2', row_index][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'mu_2', row_index][0]
                            )
                        )

                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'nu_3', row_index][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'mu_3', row_index][:, 0]
                            )
                        )

                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'nu_4', row_index][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                    der_name, time_step, 'mu_4', row_index][0]
                            )
                        )

                    else:
                        for index_within_der_disturbance in index_temp_der_disturbance:
                            #  constr 40 d)
                            optimization_problem_dro.constraints.append(
                                cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                        der_name, time_step, 'nu_1', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                        der_name, time_step, 'mu_1', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0]
                                )
                            )

                            optimization_problem_dro.constraints.append(
                                cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                        der_name, time_step, 'nu_2', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                        der_name, time_step, 'mu_2', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0]
                                )
                            )

                            optimization_problem_dro.constraints.append(
                                cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                        der_name, time_step, 'nu_3', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                        der_name, time_step, 'mu_3', row_index][
                                    [
                                        2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))[0].item(),
                                        2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))[0].item() + 1
                                    ], :
                                    ][:, 0]
                                )
                            )

                            optimization_problem_dro.constraints.append(
                                cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                        der_name, time_step, 'nu_4', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_40[
                                        der_name, time_step, 'mu_4', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0]
                                )
                            )

    print('define constr 41 a')
    optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41 = {}
    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41 = {}

    for row_index in range(b3_vector.shape[0]):
        for price_category in price_categories:
            for time_step in der_model_set.timesteps:
                for dual_category in ambiguity_set_dual_variables_categories:
                    if dual_category == 'mu_3':
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                            price_category, time_step, dual_category, row_index
                        ] = cp.Variable((2, 1))
                    else:
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                            price_category, time_step, dual_category, row_index
                        ] = cp.Variable((1, 1))

        for der_name, der_model in der_model_set.flexible_der_models.items():
            if not der_model.disturbances.empty:
                for time_step in der_model_set.timesteps:

                    index_temp_der_disturbance = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                        timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                    )

                    size_der_disturbance = index_temp_der_disturbance.shape[0]

                    for dual_category in ambiguity_set_dual_variables_categories:
                        if dual_category == 'mu_3':
                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                der_name, time_step, dual_category, row_index
                            ] = cp.Variable((2 * size_der_disturbance, 1))

                        else:
                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                der_name, time_step, dual_category, row_index
                            ] = cp.Variable((size_der_disturbance, 1))

    # constr 41 a)
    temp_41a_orig = b3_vector - A3_matrix @ optimization_problem_dro.s1_vector - \
        B3_matrix @ optimization_problem_dro.k_0_s2 - D3_matrix @ optimization_problem_dro.k_0_s3

    for row_index in range(b3_vector.shape[0]):

        temp_41a = temp_41a_orig[row_index, :]

        for price_category in price_categories:
            for time_step in der_model_set.timesteps:
                if price_category == 'energy':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_energy_price_deviation_s2', timestep=time_step
                    )
                elif price_category == 'up_reserve':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_up_reserve_price_deviation_s2',
                        timestep=time_step
                    )
                else:
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_down_reserve_price_deviation_s2',
                        timestep=time_step
                    )

                temp_41a -= (
                                    optimization_problem_dro.delta_lower_bound[
                                        np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                            ) @ \
                            (
                                optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                                    price_category, time_step, 'mu_1', row_index]
                            )

                temp_41a -= (
                                    -optimization_problem_dro.delta_lower_bound[
                                        np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                            ) @ \
                            (
                                optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                                    price_category, time_step, 'nu_1', row_index]
                            )

                temp_41a -= (
                                    -optimization_problem_dro.delta_upper_bound[
                                        np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                            ) @ \
                            (
                                optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                                    price_category, time_step, 'mu_2', row_index]
                            )

                temp_41a -= (
                                    optimization_problem_dro.delta_upper_bound[
                                        np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0] + 1
                            ) @ \
                            (
                                optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                                    price_category, time_step, 'nu_2', row_index]
                            )

                temp_41a -= np.array([[0.5, 0]]) @ optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                    price_category, time_step, 'mu_3', row_index]

                temp_41a -= 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                    price_category, time_step, 'nu_3', row_index]

                temp_41a -= (
                                optimization_problem_dro.u_upper_bound[
                                    np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices)), 0
                                ]
                            ) @ \
                            (
                                optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                                    price_category, time_step, 'nu_4', row_index]
                            )

        for der_name, der_model in der_model_set.flexible_der_models.items():
            if not der_model.disturbances.empty:
                for time_step in der_model_set.timesteps:

                    index_temp_der_disturbance = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                        timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                    )

                    size_der_disturbance = index_temp_der_disturbance.shape[0]
                    #print(size_der_disturbance)

                    if size_der_disturbance == 1:
                        temp_41a -= (
                                            optimization_problem_dro.delta_lower_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    index_temp_der_disturbance)), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                            der_name, time_step, 'mu_1', row_index
                                        ]
                                    )

                        temp_41a -= (
                                            -optimization_problem_dro.delta_lower_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    index_temp_der_disturbance)), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                            der_name, time_step, 'nu_1', row_index
                                        ]
                                    )

                        temp_41a -= (
                                            -optimization_problem_dro.delta_upper_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    index_temp_der_disturbance)), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                            der_name, time_step, 'mu_2', row_index
                                        ]
                                    )

                        temp_41a -= (
                                            optimization_problem_dro.delta_upper_bound[
                                                np.where(pd.Index(delta_indices_stage2).isin(
                                                    index_temp_der_disturbance)), 0] + 1
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                            der_name, time_step, 'nu_2', row_index
                                        ]
                                    )

                        temp_41a -= np.array([[0.5, 0]]) @ \
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                        der_name, time_step, 'mu_3', row_index
                                    ]

                        temp_41a -= 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                            der_name, time_step, 'nu_3', row_index
                        ]

                        temp_41a -= (
                                        optimization_problem_dro.u_upper_bound[
                                            np.where(pd.Index(delta_indices_stage2).isin(index_temp_der_disturbance)), 0
                                        ]
                                    ) @ \
                                    (
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                            der_name, time_step, 'nu_4', row_index
                                        ]
                                    )
                    else:
                        for index_within_der_disturbance in index_temp_der_disturbance:
                            temp_41a -= (
                                                optimization_problem_dro.delta_lower_bound[
                                                    np.where(pd.Index(delta_indices_stage2).isin(
                                                        [index_within_der_disturbance])), 0] + 1
                                        ) @ \
                                        (
                                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                                der_name, time_step, 'mu_1', row_index
                                            ][
                                                np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))
                                            ]
                                        )

                            temp_41a -= (
                                                -optimization_problem_dro.delta_lower_bound[
                                                    np.where(pd.Index(delta_indices_stage2).isin(
                                                        [index_within_der_disturbance])), 0] + 1
                                        ) @ \
                                        (
                                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                                der_name, time_step, 'nu_1', row_index
                                            ][
                                                np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))
                                            ]
                                        )

                            temp_41a -= (
                                                -optimization_problem_dro.delta_upper_bound[
                                                    np.where(pd.Index(delta_indices_stage2).isin(
                                                        [index_within_der_disturbance])), 0] + 1
                                        ) @ \
                                        (
                                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                                der_name, time_step, 'mu_2', row_index
                                            ][
                                                np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))
                                            ]
                                        )

                            temp_41a -= (
                                                optimization_problem_dro.delta_upper_bound[
                                                    np.where(pd.Index(delta_indices_stage2).isin(
                                                        [index_within_der_disturbance])), 0] + 1
                                        ) @ \
                                        (
                                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                                der_name, time_step, 'nu_2', row_index
                                            ][
                                                np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))
                                            ]
                                        )

                            temp_41a -= np.array([[0.5, 0]]) @ \
                                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                            der_name, time_step, 'mu_3', row_index
                                        ][
                                        [
                                            2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                                [index_within_der_disturbance]))[0].item(),
                                            2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                                [index_within_der_disturbance]))[0].item() + 1
                                        ], :
                                        ]

                            temp_41a -= 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                der_name, time_step, 'nu_3', row_index
                            ][
                                np.where(pd.Index(index_temp_der_disturbance).isin([index_within_der_disturbance]))
                            ]

                            temp_41a -= (
                                            optimization_problem_dro.u_upper_bound[
                                                np.where(
                                                    pd.Index(delta_indices_stage2).isin([index_within_der_disturbance])), 0
                                            ]
                                        ) @ \
                                        (
                                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                                der_name, time_step, 'nu_4', row_index
                                            ][
                                                np.where(pd.Index(index_temp_der_disturbance).isin(
                                                    [index_within_der_disturbance]))
                                            ]

                                        )

        optimization_problem_dro.constraints.append(
            temp_41a >= 0
        )
    print('define constr 41 b')
    # constr 41 b)
    temp_41b = -C3_matrix - B3_matrix @ optimization_problem_dro.K_delta_s2 - \
        D3_matrix @ optimization_problem_dro.K_delta_s3

    for row_index in range(b3_vector.shape[0]):
        for price_category in price_categories:
            for time_step in der_model_set.timesteps:
                if price_category == 'energy':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_energy_price_deviation_s2',
                        timestep=time_step
                    )
                elif price_category == 'up_reserve':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_up_reserve_price_deviation_s2',
                        timestep=time_step
                    )
                else:
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_down_reserve_price_deviation_s2',
                        timestep=time_step
                    )

                optimization_problem_dro.constraints.append(
                    - optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'mu_1', row_index]
                    + optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'nu_1', row_index]
                    + optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'mu_2', row_index]
                    - optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'nu_2', row_index]
                    + np.array([[0.5, 0]]) @ optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'mu_3', row_index]
                    ==
                    temp_41b[row_index, np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices))[0]]
                )

        for der_name, der_model in der_model_set.flexible_der_models.items():
            if not der_model.disturbances.empty:
                for time_step in der_model_set.timesteps:

                    index_temp_der_disturbance = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                        timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                    )

                    size_der_disturbance = index_temp_der_disturbance.shape[0]

                    if size_der_disturbance == 1:
                        optimization_problem_dro.constraints.append(
                            - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                der_name, time_step, 'mu_1', row_index]
                            + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                der_name, time_step, 'nu_1', row_index]
                            + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                der_name, time_step, 'mu_2', row_index]
                            - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                der_name, time_step, 'nu_2', row_index]
                            + np.array([[0.5, 0]]) @
                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                der_name, time_step, 'mu_3', row_index]
                            ==
                            temp_41b[
                                row_index, np.where(pd.Index(delta_indices_stage2).isin(index_temp_der_disturbance))[0]]
                        )
                    else:
                        for index_within_der_disturbance in index_temp_der_disturbance:
                            optimization_problem_dro.constraints.append(
                                - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'mu_1', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'nu_1', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'mu_2', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                - optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'nu_2', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                + np.array([[0.0, 1.0]]) @
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'mu_3', row_index][
                                [
                                    2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))[0].item(),
                                    2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))[0].item() + 1
                                ], :
                                ]
                                ==
                                temp_41b[row_index, np.where(
                                    pd.Index(delta_indices_stage2).isin([index_within_der_disturbance]))[0]]
                            )
    print('define constr 41 c + d')
    #  constr 41 c)
    temp_41c = - B3_matrix @ optimization_problem_dro.K_u_s2 - D3_matrix @ optimization_problem_dro.K_u_s3

    for row_index in range(b3_vector.shape[0]):
        for price_category in price_categories:
            for time_step in der_model_set.timesteps:
                if price_category == 'energy':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_energy_price_deviation_s2',
                        timestep=time_step
                    )
                elif price_category == 'up_reserve':
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_up_reserve_price_deviation_s2',
                        timestep=time_step
                    )
                else:
                    index_temp_uncertain_prices = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_down_reserve_price_deviation_s2',
                        timestep=time_step
                    )
                #  constr 41 c)
                optimization_problem_dro.constraints.append(
                    + np.array([[-0.5, 0]]) @ optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'mu_3', row_index
                    ]
                    + 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'nu_3', row_index
                    ]
                    + optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'mu_4', row_index
                    ]
                    ==
                    temp_41c[row_index, np.where(pd.Index(delta_indices_stage2).isin(index_temp_uncertain_prices))[0]]
                )

        for der_name, der_model in der_model_set.flexible_der_models.items():
            if not der_model.disturbances.empty:
                for time_step in der_model_set.timesteps:

                    index_temp_der_disturbance = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                        timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                    )

                    size_der_disturbance = index_temp_der_disturbance.shape[0]
                    #  constr 41 c)
                    if size_der_disturbance == 1:
                        optimization_problem_dro.constraints.append(
                            np.array([[-0.5, 0]]) @
                            optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                der_name, time_step, 'mu_3', row_index]
                            + 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                der_name, time_step, 'nu_3', row_index]
                            + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                der_name, time_step, 'mu_4', row_index]
                            ==
                            temp_41c[
                                 row_index, np.where(pd.Index(delta_indices_stage2).isin(index_temp_der_disturbance))[0]]
                        )

                    else:
                        #  constr 41 c)
                        for index_within_der_disturbance in index_temp_der_disturbance:
                            optimization_problem_dro.constraints.append(
                                + np.array([[-0.5, 0]]) @
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'mu_3', row_index][
                                [
                                    2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))[0].item(),
                                    2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))[0].item() + 1
                                ], :
                                ]
                                + 0.5 * optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'nu_3', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                + optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'mu_4', row_index][
                                    np.where(pd.Index(index_temp_der_disturbance).isin(
                                        [index_within_der_disturbance]))
                                ]
                                ==
                                temp_41c[row_index,
                                         np.where(pd.Index(delta_indices_stage2).isin([index_within_der_disturbance]))[0]]
                            )

        #  constr 41 d)
        for price_category in price_categories:
            for time_step in der_model_set.timesteps:

                optimization_problem_dro.constraints.append(
                    cp.SOC(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                            price_category, time_step, 'nu_1', row_index][0],
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                            price_category, time_step, 'mu_1', row_index][0]
                    )
                )

                optimization_problem_dro.constraints.append(
                    cp.SOC(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                            price_category, time_step, 'nu_2', row_index][0],
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                            price_category, time_step, 'mu_2', row_index][0]
                    )
                )

                optimization_problem_dro.constraints.append(
                    cp.SOC(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                            price_category, time_step, 'nu_3', row_index][0],
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                            price_category, time_step, 'mu_3', row_index][:, 0]
                    )
                )

                optimization_problem_dro.constraints.append(
                    cp.SOC(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                            price_category, time_step, 'nu_4', row_index][0],
                        optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                            price_category, time_step, 'mu_4', row_index][0]
                    )
                )

                # constr 41e
                optimization_problem_dro.constraints.append(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'nu_1', row_index] >= 0
                )

                optimization_problem_dro.constraints.append(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'nu_2', row_index] >= 0
                )

                optimization_problem_dro.constraints.append(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'nu_3', row_index] >= 0
                )

                optimization_problem_dro.constraints.append(
                    optimization_problem_dro.ambiguity_set_duals_uncertain_price_constr_41[
                        price_category, time_step, 'nu_4', row_index] >= 0
                )

        for der_name, der_model in der_model_set.flexible_der_models.items():
            if not der_model.disturbances.empty:
                for time_step in der_model_set.timesteps:

                    # constr 41e
                    optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                            der_name, time_step, 'nu_1', row_index] >= 0
                    )

                    optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                            der_name, time_step, 'nu_2', row_index] >= 0
                    )

                    optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                            der_name, time_step, 'nu_3', row_index] >= 0
                    )

                    optimization_problem_dro.constraints.append(
                        optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                            der_name, time_step, 'nu_4', row_index] >= 0
                    )

                    index_temp_der_disturbance = fledge.utils.get_index(
                        standard_form_stage_2.variables, name='uncertainty_disturbances_vector_s2',
                        timestep=time_step, der_name=[der_name], disturbance=der_model.disturbances,
                    )

                    size_der_disturbance = index_temp_der_disturbance.shape[0]

                    if size_der_disturbance == 1:
                        #  constr 41 d)
                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'nu_1', row_index][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'mu_1', row_index][0]
                            )
                        )

                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'nu_2', row_index][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'mu_2', row_index][0]
                            )
                        )

                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'nu_3', row_index][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'mu_3', row_index][:, 0])
                        )

                        optimization_problem_dro.constraints.append(
                            cp.SOC(
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'nu_4', row_index][0],
                                optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                    der_name, time_step, 'mu_4', row_index][0]
                            )
                        )

                    else:
                        for index_within_der_disturbance in index_temp_der_disturbance:
                            optimization_problem_dro.constraints.append(
                                cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                        der_name, time_step, 'nu_1', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                        der_name, time_step, 'mu_1', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0]
                                )
                            )

                            optimization_problem_dro.constraints.append(
                                cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                        der_name, time_step, 'nu_2', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                        der_name, time_step, 'mu_2', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0]
                                )
                            )

                            optimization_problem_dro.constraints.append(
                                cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                        der_name, time_step, 'nu_3', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                        der_name, time_step, 'mu_3', row_index][
                                    [
                                        2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))[0].item(),
                                        2 * np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))[0].item() + 1
                                    ], :
                                    ][:, 0]
                                )
                            )

                            optimization_problem_dro.constraints.append(
                                cp.SOC(
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                        der_name, time_step, 'nu_4', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0],
                                    optimization_problem_dro.ambiguity_set_duals_uncertain_der_disturbance_constr_41[
                                        der_name, time_step, 'mu_4', row_index][
                                        np.where(pd.Index(index_temp_der_disturbance).isin(
                                            [index_within_der_disturbance]))
                                    ][0]
                                )
                            )

    optimization_problem_dro.objective -= (
        (
            f_vector.T
            @ optimization_problem_dro.s1_vector
        ) -
        (
            optimization_problem_dro.gamma.T
            @ optimization_problem_dro.beta
        ) -
        (
            optimization_problem_dro.sigma
        )

    )
    # Define optimization objective

    # Solve optimization problem.
    optimization_problem_dro.solve()

    # Obtain results.
    results_dro = standard_form_stage_1.get_results(optimization_problem_dro.s1_vector)

    # obtain results for dual variables
    beta_res = pd.Series(optimization_problem_dro.beta.value.ravel())
    sigma_res = pd.Series(optimization_problem_dro.sigma.value.ravel())

    # Obtain reserve results.
    no_reserve = pd.Series(results_dro['energy'].values.ravel(), index=der_model_set.timesteps)
    up_reserve = pd.Series(results_dro['up_reserve'].values.ravel(), index=der_model_set.timesteps)
    down_reserve = pd.Series(results_dro['down_reserve'].values.ravel(), index=der_model_set.timesteps)

    # Instantiate DER results variables.
    state_vector = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.states)
        for stochastic_scenario in stochastic_scenarios
    }
    control_vector = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.controls)
        for stochastic_scenario in stochastic_scenarios
    }
    output_vector = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
        for stochastic_scenario in stochastic_scenarios
    }
    der_active_power_vector = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.electric_ders)
        for stochastic_scenario in stochastic_scenarios
    }
    der_active_power_vector_per_unit = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.electric_ders)
        for stochastic_scenario in stochastic_scenarios
    }
    der_reactive_power_vector = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.electric_ders)
        for stochastic_scenario in stochastic_scenarios
    }
    der_reactive_power_vector_per_unit = {
        stochastic_scenario: pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.electric_ders)
        for stochastic_scenario in stochastic_scenarios
    }

    # Obtain DER results.
    for stochastic_scenario in stochastic_scenarios:
        for der_name in der_model_set.flexible_der_names:
            state_vector[stochastic_scenario].loc[:, (der_name, slice(None))] = (
                results_dro['state_vector'].loc[:, (stochastic_scenario, der_name, slice(None))].values
            )
            control_vector[stochastic_scenario].loc[:, (der_name, slice(None))] = (
                results_dro['control_vector'].loc[:, (stochastic_scenario, der_name, slice(None))].values
            )
            output_vector[stochastic_scenario].loc[:, (der_name, slice(None))] = (
                results_dro['output_vector'].loc[:, (stochastic_scenario, der_name, slice(None))].values
            )
        for der_name, der_model in der_model_set.der_models.items():
            if der_model.is_electric_grid_connected:
                der_active_power_vector_per_unit[stochastic_scenario].loc[:, (der_model.der_type, der_name)] = (
                    results_dro['der_active_power_vector'].loc[:,
                    [(stochastic_scenario, (der_model.der_type, der_name))]].values
                )
                der_active_power_vector[stochastic_scenario].loc[:, (slice(None), der_name)] = (
                        der_active_power_vector_per_unit[stochastic_scenario].loc[:, (slice(None), der_name)].values
                        * der_model.active_power_nominal
                )
                der_reactive_power_vector_per_unit[stochastic_scenario].loc[:, (slice(None), der_name)] = (
                    results_dro['der_reactive_power_vector'].loc[:,
                    [(stochastic_scenario, (der_model.der_type, der_name))]].values
                )
                der_reactive_power_vector[stochastic_scenario].loc[:, (slice(None), der_name)] = (
                        der_reactive_power_vector_per_unit[stochastic_scenario].loc[:, (slice(None), der_name)].values
                        * der_model.reactive_power_nominal
                )

    # Instantiate optimization problem.
    optimization_problem_deterministic = fledge.utils.OptimizationProblem()

    # Define optimization problem.
    optimization_problem_deterministic.x_vector = cp.Variable((len(standard_form_stage_1.variables), 1))
    optimization_problem_deterministic.constraints.append(
        A1_matrix.toarray() @ optimization_problem_deterministic.x_vector <= b1_vector
    )
    optimization_problem_deterministic.objective -= (
            (
                f_vector.T
                @ optimization_problem_deterministic.x_vector
            )
    )
    # Define optimization objective

    # Solve optimization problem.
    optimization_problem_deterministic.solve()
    # Obtain results.
    results_determinstic = standard_form_stage_1.get_results(optimization_problem_deterministic.x_vector)

    # Obtain reserve results.
    no_reserve_det = pd.Series(results_determinstic['energy'].values.ravel(), index=der_model_set.timesteps)
    up_reserve_det = pd.Series(results_determinstic['up_reserve'].values.ravel(), index=der_model_set.timesteps)
    down_reserve_det = pd.Series(results_determinstic['down_reserve'].values.ravel(), index=der_model_set.timesteps)


    # Plot some results.
    figure = go.Figure()
    figure.add_scatter(
        x=no_reserve.index,
        y=no_reserve.values,
        name='no_reserve',
        line=go.scatter.Line(shape='hv', width=5, dash='dot')
    )
    figure.add_scatter(
        x=up_reserve.index,
        y=up_reserve.values,
        name='up_reserve',
        line=go.scatter.Line(shape='hv', width=4, dash='dot')
    )
    figure.add_scatter(
        x=down_reserve.index,
        y=down_reserve.values,
        name='down_reserve',
        line=go.scatter.Line(shape='hv', width=3, dash='dot')
    )
    figure.add_scatter(
        x=up_reserve.index,
        y=(no_reserve + up_reserve).values,
        name='no_reserve + up_reserve',
        line=go.scatter.Line(shape='hv', width=2, dash='dot')
    )
    figure.add_scatter(
        x=up_reserve.index,
        y=(no_reserve - down_reserve).values,
        name='no_reserve - down_reserve',
        line=go.scatter.Line(shape='hv', width=1, dash='dot')
    )
    figure.update_layout(
        title=f'Power balance',
        xaxis=go.layout.XAxis(tickformat='%H:%M'),
        legend=go.layout.Legend(x=0.01, xanchor='auto', y=0.99, yanchor='auto')
    )
    # figure.show()
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'0_power_balance'))

    for der_name, der_model in der_model_set.flexible_der_models.items():

        for output in der_model.outputs:
            figure = go.Figure()
            figure.add_scatter(
                x=der_model.output_maximum_timeseries.index,
                y=der_model.output_maximum_timeseries.loc[:, output].values,
                name='Maximum',
                line=go.scatter.Line(shape='hv')
            )
            figure.add_scatter(
                x=der_model.output_minimum_timeseries.index,
                y=der_model.output_minimum_timeseries.loc[:, output].values,
                name='Minimum',
                line=go.scatter.Line(shape='hv')
            )
            for number, stochastic_scenario in enumerate(stochastic_scenarios):
                figure.add_scatter(
                    x=output_vector[stochastic_scenario].index,
                    y=output_vector[stochastic_scenario].loc[:, (der_name, output)].values,
                    name=f'Optimal: {stochastic_scenario}',
                    line=go.scatter.Line(shape='hv', width=number + 3, dash='dot')
                )
            figure.update_layout(
                title=f'DER: ({der_model.der_type}, {der_name}) / Output: {output}',
                xaxis=go.layout.XAxis(tickformat='%H:%M'),
                legend=go.layout.Legend(x=0.01, xanchor='auto', y=0.99, yanchor='auto')
            )
            # figure.show()
            fledge.utils.write_figure_plotly(figure, os.path.join(
                results_path, f'der_{der_model.der_type}_{der_name}_output_{output}'
            ))

        # for control in der_model.controls:
        #     figure = go.Figure()
        #     for number, stochastic_scenario in enumerate(stochastic_scenarios):
        #         figure.add_scatter(
        #             x=output_vector[stochastic_scenario].index,
        #             y=output_vector[stochastic_scenario].loc[:, (der_name, control)].values,
        #             name=f'Optimal: {stochastic_scenario}',
        #             line=go.scatter.Line(shape='hv', width=number+3, dash='dot')
        #         )
        #     figure.update_layout(
        #         title=f'DER: ({der_model.der_type}, {der_name}) / Control: {control}',
        #         xaxis=go.layout.XAxis(tickformat='%H:%M'),
        #         legend=go.layout.Legend(x=0.01, xanchor='auto', y=0.99, yanchor='auto')
        #     )
        #     # figure.show()
        #     fledge.utils.write_figure_plotly(figure, os.path.join(
        #         results_path, f'der_{der_model.der_type}_{der_name}_control_{control}'
        #     ))

        # for disturbance in der_model.disturbances:
        #     figure = go.Figure()
        #     figure.add_scatter(
        #         x=der_model.disturbance_timeseries.index,
        #         y=der_model.disturbance_timeseries.loc[:, disturbance].values,
        #         line=go.scatter.Line(shape='hv')
        #     )
        #     figure.update_layout(
        #         title=f'DER: ({der_model.der_type}, {der_name}) / Disturbance: {disturbance}',
        #         xaxis=go.layout.XAxis(tickformat='%H:%M'),
        #         showlegend=False
        #     )
        #     # figure.show()
        #     fledge.utils.write_figure_plotly(figure, os.path.join(
        #         results_path, f'der_{der_model.der_type}_{der_name}_disturbance_{disturbance}'
        #     ))

    for commodity_type in ['active_power', 'reactive_power']:

        if commodity_type in price_data.price_timeseries.columns.get_level_values('commodity_type'):
            figure = go.Figure()
            figure.add_scatter(
                x=price_data.price_timeseries.index,
                y=price_data.price_timeseries.loc[:, (commodity_type, 'source', 'source')].values,
                line=go.scatter.Line(shape='hv')
            )
            figure.update_layout(
                title=f'Price: {commodity_type}',
                xaxis=go.layout.XAxis(tickformat='%H:%M')
            )
            # figure.show()
            fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'price_{commodity_type}'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")



if __name__ == '__main__':
    main()
