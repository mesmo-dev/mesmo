"""Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem."""

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.sparse as sp
import matplotlib.pyplot as plt
# plt.style.use(['science','ieee'])
from future.backports.datetime import time

plt.style.use(['science'])
from matplotlib.ticker import FormatStrFormatter
pd.options.plotting.backend = "matplotlib"

from mesmo.kkt_conditions_with_state_space import StrategicMarket
import mesmo


def main():
    # TODO: Currently not working. Review limits below.

    # scenarios = [None]
    # scenario_name = "strategic_dso_market"
    # global strategic_der_model_set
    # scenario_name = 'strategic_market_19_node'
    scenario_name = 'ieee_34node'
    strategic_scenario = True
    kkt_conditions = False
    strategic_der_name = 'pv_860_strategic'
    # strategic_der_name = 'pv_890'
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    scenario_profiles = mesmo.data_interface.DERData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name)
    # Obtain models.
    electric_grid_model = mesmo.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = mesmo.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model_set = (
        mesmo.electric_grid_models.LinearElectricGridModelSet(
            electric_grid_model,
            power_flow_solution
        )
    )
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)

    # Instantiate centralized optimization problem.
    optimization_non_strategic = mesmo.utils.OptimizationProblem()

    # max_branch_power = np.array([0.808, 0.784, 0.726, 0.532,
    #                              0.926, 0.803, 0.810, 0.708, 0.708,
    #                              0.789, 0.789, 0.789, 0.789, 0.789, 0.789,
    #                              0.538, 0.538, 0.538, 0.538])
    max_branch_power = 0.8
    max_branch_power = pd.Series(1.0, index=electric_grid_model.branches)
    max_branch_power['transformer'] = 10

    # Define electric grid problem.
    # TODO: Review limits.
    node_voltage_magnitude_vector_minimum = 0.9 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.1 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = max_branch_power.values * electric_grid_model.branch_power_vector_magnitude_reference

    jump_here = True
    grid_cost_coefficient = 1
    report_time = '2019-07-17 14:00:00'
    flexible_der_type = ['flexible_generator', 'flexible_load']
    time_index = [scenario_data.timesteps]
    price_correction = 1e3 / scenario_data.scenario.at['base_apparent_power']
    admm_max_iteration = 40

    # """
    # Set seller and buyer ders for P2P market
    seller_der_name = ['pv_806', 'pv_812']
    buyer_der_name = ['fx_818_1_y', 'fx_856_2_y', 'fl_840_y', 'fx_810_2_y', 'fx_824_3_y', 'fx_842_1_y']
    seller_ders = pd.Index(
        [der_name for der_type, der_name in electric_grid_model.ders if der_name in seller_der_name])
    buyer_ders = pd.Index([der_name for der_type, der_name in electric_grid_model.ders if der_name in buyer_der_name])
    peers = seller_ders.append(buyer_ders)


    der_model_set.define_optimization_problem(optimization_non_strategic,
                                              price_data,
                                              grid_cost_coefficient=grid_cost_coefficient
                                              # kkt_conditions=False,
                                              # state_space_model=True
                                              )

    linear_electric_grid_model_set.define_optimization_problem(
        optimization_non_strategic,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
        grid_cost_coefficient=grid_cost_coefficient
        # kkt_conditions=False
    )
    optimization_non_strategic.solve()
    results_non_strategic = mesmo.problems.Results()
    results_non_strategic.update(linear_electric_grid_model_set.get_optimization_results(optimization_non_strategic))
    results_non_strategic.update(der_model_set.get_optimization_results(optimization_non_strategic))
    dlmps_non_strategic = linear_electric_grid_model_set.get_optimization_dlmps(optimization_non_strategic, price_data)
    flexible_der_active_power_non_strategic = results_non_strategic.der_active_power_vector_per_unit[flexible_der_type]
    flexible_der_reactive_power_non_strategic = results_non_strategic.der_reactive_power_vector_per_unit[
        flexible_der_type]

    seller_dlmp_non_strategic = dlmps_non_strategic.electric_grid_total_dlmp_der_active_power.loc[:, (slice(None), seller_ders)]
    buyer_dlmp_non_strategic = dlmps_non_strategic.electric_grid_total_dlmp_der_active_power.loc[:, (slice(None), buyer_ders)]

    grid_using_price_non_strategic = pd.DataFrame(0, index=seller_dlmp_non_strategic.index,
                                                  columns=pd.MultiIndex.from_product([seller_ders, buyer_ders]))
    for x, b in buyer_dlmp_non_strategic.columns:
        for y, s in seller_dlmp_non_strategic.columns:  # for y, b in grid_using_price_strategic.columns:
            grid_using_price_non_strategic.at[:, (s, b)] = buyer_dlmp_non_strategic.loc[:, (x, b)].values - seller_dlmp_non_strategic.loc[:,
                                                                                          (y, s)].values

    # grid_using_price_strategic *=-1

    seller_optimization_problem_sets_non_strategic = pd.Series(data=None, index=seller_ders, dtype=object)
    for seller in seller_optimization_problem_sets_non_strategic.index:
        seller_optimization_problem_sets_non_strategic.at[seller] = mesmo.utils.OptimizationProblem()

        # Define seller's ADMM variable
        seller_optimization_problem_sets_non_strategic.loc[seller].define_variable(
            f'energy_transacted_from_seller_{seller}_to_buyers', buyer=buyer_ders, timestep=scenario_data.timesteps
        )
        seller_optimization_problem_sets_non_strategic.loc[seller].define_variable(
            f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers', buyer=buyer_ders,
            timestep=scenario_data.timesteps
        )
        seller_optimization_problem_sets_non_strategic.loc[seller].define_variable(
            f'seller_{seller}_active_power_vector', timestep=scenario_data.timesteps
        )
        # Define seller's ADMMM parameter
        seller_optimization_problem_sets_non_strategic.loc[seller].define_parameter(
            f'admm_lambda_seller_{seller}_to_buyers_active_power',
            np.zeros(len(scenario_data.timesteps) * len(buyer_ders))
        )
        seller_optimization_problem_sets_non_strategic.loc[seller].define_parameter(
            f'energy_transacted_from_seller_{seller}_to_buyers_local_copy',
            np.zeros((len(scenario_data.timesteps) * len(buyer_ders), 1))
        )
        seller_optimization_problem_sets_non_strategic.loc[seller].define_parameter(
            f'energy_transacted_from_seller_{seller}_to_buyers_zeros',
            np.zeros((len(scenario_data.timesteps) * len(buyer_ders), 1))
        )
        seller_optimization_problem_sets_non_strategic.loc[seller].define_parameter(
            f'seller_{seller}_max_power',
            # np.array([1.0] * len(scenario_data.timesteps))
            np.transpose([der_model_set.der_models[seller].active_power_nominal_timeseries.values])
        )
        seller_optimization_problem_sets_non_strategic.loc[seller].define_parameter(
            f'seller_{seller}_min_power',
            np.array([0.0] * len(scenario_data.timesteps))
        )
        seller_optimization_problem_sets_non_strategic.loc[seller].define_parameter(
            f'half_of_grid_using_price_for_seller_{seller}',
            0.5 * pd.concat([grid_using_price_non_strategic.loc[:, (seller, buyer)] for buyer in buyer_ders]).values
        )
        seller_optimization_problem_sets_non_strategic.loc[seller].define_parameter(
            f'buyer_sized_ones_for_{seller}_energy_transaction',
            np.tile(np.diag(np.ones(len(scenario_data.timesteps))), len(buyer_ders))
        )

    buyer_optimization_problem_sets_non_strategic = pd.Series(data=None, index=buyer_ders, dtype=object)
    for buyer in buyer_optimization_problem_sets_non_strategic.index:
        buyer_optimization_problem_sets_non_strategic.loc[buyer] = mesmo.utils.OptimizationProblem()

        # Define seller's ADMM variable
        buyer_optimization_problem_sets_non_strategic.loc[buyer].define_variable(
            f'energy_transacted_from_sellers_to_buyer_{buyer}', seller=seller_ders, timestep=scenario_data.timesteps
        )
        buyer_optimization_problem_sets_non_strategic.loc[buyer].define_variable(
            f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}', seller=seller_ders,
            timestep=scenario_data.timesteps
        )
        buyer_optimization_problem_sets_non_strategic.loc[buyer].define_variable(
            f'buyer_{buyer}_active_power_vector', timestep=scenario_data.timesteps
        )
        # Define seller's ADMMM parameter
        buyer_optimization_problem_sets_non_strategic.loc[buyer].define_parameter(
            f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy',
            np.zeros((len(scenario_data.timesteps) * len(seller_ders), 1))
        )
        buyer_optimization_problem_sets_non_strategic.loc[buyer].define_parameter(
            f'admm_lambda_from_sellers_to_buyer_{buyer}_active_power',
            np.zeros(len(scenario_data.timesteps) * len(seller_ders))
        )
        buyer_optimization_problem_sets_non_strategic.loc[buyer].define_parameter(
            f'energy_transacted_from_sellers_to_buyer_{buyer}_zeros',
            np.zeros((len(scenario_data.timesteps) * len(seller_ders), 1))
        )
        buyer_optimization_problem_sets_non_strategic.loc[buyer].define_parameter(
            f'buyer_{buyer}_min_power',
            # np.array([1.0] * len(scenario_data.timesteps))
            np.transpose([der_model_set.der_models[buyer].active_power_nominal_timeseries.values])
        )
        buyer_optimization_problem_sets_non_strategic.loc[buyer].define_parameter(
            f'buyer_{buyer}_max_power',
            np.array([0.0] * len(scenario_data.timesteps))
        )
        buyer_optimization_problem_sets_non_strategic.loc[buyer].define_parameter(
            f'half_of_grid_using_price_for_buyer_{buyer}',
            0.5 * pd.concat([grid_using_price_non_strategic.loc[:, (seller, buyer)] for seller in seller_ders]).values
        )
        buyer_optimization_problem_sets_non_strategic.loc[buyer].define_parameter(
            f'seller_sized_ones_for_{buyer}_energy_transaction',
            np.tile(np.diag(np.ones(len(scenario_data.timesteps))), len(seller_ders))
        )

    admm_iteration_non_strategic = 0
    admm_rho = 800
    radius_non_strategic = 1
    radius_non_strategic_save = []
    peers_active_power_non_strategic = pd.DataFrame(index=scenario_data.timesteps, columns=peers)

    while radius_non_strategic >= 0.0001:

        # Defining optimization constraints and objectives for sellers:
        for seller in seller_optimization_problem_sets_non_strategic.index:
            seller_optimization_problem_sets_non_strategic.loc[seller].define_constraint(
                ('variable', f'buyer_sized_ones_for_{seller}_energy_transaction',
                 dict(name=f'energy_transacted_from_seller_{seller}_to_buyers',
                      buyer=buyer_ders, timestep=scenario_data.timesteps)),
                '==',
                ('variable', 1.0,
                 dict(name=f'seller_{seller}_active_power_vector'))
            )
            seller_optimization_problem_sets_non_strategic.loc[seller].define_constraint(
                ('variable', 1.0, dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers',
                                       buyer=buyer_ders, timestep=scenario_data.timesteps)),
                ('constant', f'energy_transacted_from_seller_{seller}_to_buyers_local_copy'),
                '==',
                ('variable', 1.0, dict(name=f'energy_transacted_from_seller_{seller}_to_buyers',
                                       buyer=buyer_ders, timestep=scenario_data.timesteps))
            )
            seller_optimization_problem_sets_non_strategic.loc[seller].define_constraint(
                ('variable', 1.0, dict(name=f'seller_{seller}_active_power_vector')),
                '>=',
                ('constant', f'seller_{seller}_min_power')
            )
            seller_optimization_problem_sets_non_strategic.loc[seller].define_constraint(
                ('variable', 1.0, dict(name=f'seller_{seller}_active_power_vector')),
                '<=',
                ('constant', f'seller_{seller}_max_power')
            )
            seller_optimization_problem_sets_non_strategic.loc[seller].define_constraint(
                ('variable', 1.0, dict(
                    name=f'energy_transacted_from_seller_{seller}_to_buyers')),
                '>=',
                ('constant', f'energy_transacted_from_seller_{seller}_to_buyers_zeros')
            )
            seller_optimization_problem_sets_non_strategic.loc[seller].define_objective(
                ('variable', f'half_of_grid_using_price_for_seller_{seller}',
                 dict(name=f'energy_transacted_from_seller_{seller}_to_buyers')),
                ('variable', f'admm_lambda_seller_{seller}_to_buyers_active_power',
                 dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers')),
                ('variable', 0.5 * admm_rho,
                 dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers'),
                 dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers'))
            )
            seller_optimization_problem_sets_non_strategic.loc[seller].solve()
            peers_active_power_non_strategic.at[:, seller] = \
            seller_optimization_problem_sets_non_strategic[seller].results[
                f'seller_{seller}_active_power_vector'].values / \
            der_model_set.fixed_der_models[seller].active_power_nominal

        # Defining optimization constraints and objectives for sellers:
        for buyer in buyer_optimization_problem_sets_non_strategic.index:
            buyer_optimization_problem_sets_non_strategic.loc[buyer].define_constraint(
                ('variable', f'seller_sized_ones_for_{buyer}_energy_transaction',
                 dict(name=f'energy_transacted_from_sellers_to_buyer_{buyer}')),
                '==',
                ('variable', -1.0,
                 dict(name=f'buyer_{buyer}_active_power_vector'))
            )
            buyer_optimization_problem_sets_non_strategic.loc[buyer].define_constraint(
                ('variable', 1.0, dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}')),
                ('constant', f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy'),
                '==',
                ('variable', 1.0, dict(name=f'energy_transacted_from_sellers_to_buyer_{buyer}'))
            )
            buyer_optimization_problem_sets_non_strategic.loc[buyer].define_constraint(
                ('variable', 1.0, dict(name=f'buyer_{buyer}_active_power_vector')),
                '>=',
                ('constant', f'buyer_{buyer}_min_power')
            )
            buyer_optimization_problem_sets_non_strategic.loc[buyer].define_constraint(
                ('variable', 1.0, dict(name=f'buyer_{buyer}_active_power_vector')),
                '<=',
                ('constant', f'buyer_{buyer}_max_power')
            )
            buyer_optimization_problem_sets_non_strategic.loc[buyer].define_constraint(
                ('variable', 1.0, dict(
                    name=f'energy_transacted_from_sellers_to_buyer_{buyer}')),
                '>=',
                ('constant', f'energy_transacted_from_sellers_to_buyer_{buyer}_zeros')
            )
            buyer_optimization_problem_sets_non_strategic.loc[buyer].define_objective(
                ('variable', f'half_of_grid_using_price_for_buyer_{buyer}',
                 dict(name=f'energy_transacted_from_sellers_to_buyer_{buyer}')),
                ('variable', f'admm_lambda_from_sellers_to_buyer_{buyer}_active_power',
                 dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}')),
                ('variable', 0.5 * admm_rho,
                 dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}'),
                 dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}'))
            )
            buyer_optimization_problem_sets_non_strategic.loc[buyer].solve()

            peers_active_power_non_strategic.at[:, buyer] = \
            buyer_optimization_problem_sets_non_strategic[buyer].results[
                f'buyer_{buyer}_active_power_vector'].values / \
            der_model_set.fixed_der_models[buyer].active_power_nominal

        # ====================================
        optimization_non_strategic.define_parameter(
            'active_power_constant',
            np.concatenate([
                np.transpose([
                    der_model_set.fixed_der_models[der_name].active_power_nominal_timeseries.values
                    / (
                        der_model_set.fixed_der_models[der_name].active_power_nominal
                        if der_model_set.fixed_der_models[der_name].active_power_nominal != 0.0
                        else 1.0
                    )
                    if der_model_set.fixed_der_models[der_name].is_electric_grid_connected
                    else 0.0 * der_model_set.fixed_der_models[der_name].active_power_nominal_timeseries.values
                ])
                if der_name in der_model_set.fixed_der_names.drop(peers)
                else np.zeros((len(der_model_set.timesteps), 1))
                for der_type, der_name in der_model_set.electric_ders
            ], axis=1).ravel() + np.concatenate([
                np.transpose([
                    peers_active_power_non_strategic[der_name].values
                    if der_model_set.fixed_der_models[der_name].is_electric_grid_connected
                    else 0.0 * der_model_set.fixed_der_models[der_name].active_power_nominal_timeseries.values
                ])
                if der_name in peers
                else np.zeros((len(der_model_set.timesteps), 1))
                for der_type, der_name in der_model_set.electric_ders
            ], axis=1).ravel()
        )
        optimization_non_strategic.solve()

        dlmps_non_strategic = linear_electric_grid_model_set.get_optimization_dlmps(optimization_non_strategic,
                                                                                    price_data)

        seller_dlmp_non_strategic = dlmps_non_strategic.electric_grid_total_dlmp_der_active_power.loc[:,
                                    (slice(None), seller_ders)]
        buyer_dlmp_non_strategic = dlmps_non_strategic.electric_grid_total_dlmp_der_active_power.loc[:,
                                   (slice(None), buyer_ders)]

        for x, b in buyer_dlmp_non_strategic.columns:
            for y, s in seller_dlmp_non_strategic.columns:  # for y, b in grid_using_price_strategic.columns:
                grid_using_price_non_strategic.at[:, (s, b)] = buyer_dlmp_non_strategic.loc[:,
                                                               (x, b)].values - seller_dlmp_non_strategic.loc[:,
                                                                                (y, s)].values
        # =====================================

        # Update admm parameters for seller optimization:
        for seller in seller_optimization_problem_sets_non_strategic.index:
            seller_optimization_problem_sets_non_strategic.loc[seller].parameters[
                f'energy_transacted_from_seller_{seller}_to_buyers_local_copy'] = 0.5 * np.transpose([
                pd.concat([
                    seller_optimization_problem_sets_non_strategic.loc[seller].results[
                        f'energy_transacted_from_seller_{seller}_to_buyers'][buyer] for buyer in buyer_ders
                ]).values + pd.concat([
                    buyer_optimization_problem_sets_non_strategic.loc[buyer].results[
                        f'energy_transacted_from_sellers_to_buyer_{buyer}'][seller] for buyer in buyer_ders]).values
            ])

            seller_optimization_problem_sets_non_strategic.loc[seller].parameters[
                f'admm_lambda_seller_{seller}_to_buyers_active_power'] += admm_rho * (pd.concat([
                seller_optimization_problem_sets_non_strategic.loc[seller].results[
                    f'energy_transacted_from_seller_{seller}_to_buyers'][buyer] for buyer in buyer_ders
            ]).values - seller_optimization_problem_sets_non_strategic.loc[seller].parameters[
                                    f'energy_transacted_from_seller_{seller}_to_buyers_local_copy'].ravel())
            seller_optimization_problem_sets_non_strategic.loc[seller].define_parameter(
                f'half_of_grid_using_price_for_seller_{seller}',
                0.5 * pd.concat([grid_using_price_non_strategic.loc[:, (seller, buyer)] for buyer in buyer_ders]).values
            )

        # Update admm parameters for buyer optimization:
        for buyer in buyer_optimization_problem_sets_non_strategic.index:
            buyer_optimization_problem_sets_non_strategic.loc[buyer].parameters[
                f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy'] = 0.5 * np.transpose([
                pd.concat([
                    buyer_optimization_problem_sets_non_strategic.loc[buyer].results[
                        f'energy_transacted_from_sellers_to_buyer_{buyer}'][seller] for seller in seller_ders
                ]).values + pd.concat([
                    seller_optimization_problem_sets_non_strategic.loc[seller].results[
                        f'energy_transacted_from_seller_{seller}_to_buyers'][buyer] for seller in
                    seller_ders]).values
            ])

            buyer_optimization_problem_sets_non_strategic.loc[buyer].parameters[
                f'admm_lambda_from_sellers_to_buyer_{buyer}_active_power'] += admm_rho * (pd.concat([
                buyer_optimization_problem_sets_non_strategic.loc[buyer].results[
                    f'energy_transacted_from_sellers_to_buyer_{buyer}'][seller] for seller in seller_ders
            ]).values - buyer_optimization_problem_sets_non_strategic.loc[buyer].parameters[
                                        f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy'].ravel())
            buyer_optimization_problem_sets_non_strategic.loc[buyer].define_parameter(
                f'half_of_grid_using_price_for_buyer_{buyer}',
                0.5 * pd.concat([grid_using_price_non_strategic.loc[:, (seller, buyer)] for seller in seller_ders]).values
            )

        radius_non_strategic = np.linalg.norm(
            np.concatenate(
            [seller_optimization_problem_sets_non_strategic.loc[seller].results[
                 f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers']
             for seller in seller_ders]).ravel().__abs__()
             + np.concatenate(
            [buyer_optimization_problem_sets_non_strategic.loc[buyer].results[
                 f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}']
             for buyer in buyer_ders]).ravel().__abs__()
        )

        admm_iteration_non_strategic += 1

        radius_non_strategic_save.append(radius_non_strategic)
        print(radius_non_strategic)
        print(admm_iteration_non_strategic)
        if admm_iteration_non_strategic >= admm_max_iteration:
            break


    # if strategic_scenario:
    optimization_strategic = mesmo.utils.OptimizationProblem()
    der_model_set.define_optimization_problem(optimization_strategic,
                                              price_data,
                                              kkt_conditions=kkt_conditions,
                                              grid_cost_coefficient=grid_cost_coefficient
                                              # state_space_model=True
                                              )

    linear_electric_grid_model_set.define_optimization_problem(
        optimization_strategic,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
        kkt_conditions=kkt_conditions,
        grid_cost_coefficient=grid_cost_coefficient
    )

    strategic_der_model_set = StrategicMarket(scenario_name, strategic_der=strategic_der_name)
    strategic_der_model_set.strategic_optimization_problem(
        optimization_strategic,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
        big_m=1e3,
        kkt_conditions=kkt_conditions,
        grid_cost_coefficient=grid_cost_coefficient
    )

    optimization_strategic.solve()


    results_strategic = mesmo.problems.Results()
    results_strategic.update(linear_electric_grid_model_set.get_optimization_results(optimization_strategic))
    results_strategic.update(der_model_set.get_optimization_results(optimization_strategic))

    # Obtain DLMPs.

    dlmps_strategic = strategic_der_model_set.get_optimization_dlmps(optimization_strategic, price_data)
    dlmp_difference = dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power - \
                      dlmps_non_strategic.electric_grid_total_dlmp_node_active_power


    flexible_der_active_power_strategic = results_strategic.der_active_power_vector_per_unit[flexible_der_type]
    flexible_der_reactive_power_strategic = results_strategic.der_reactive_power_vector_per_unit[flexible_der_type]

    # =======================================================================================

    seller_dlmp_strategic = dlmps_strategic.strategic_electric_grid_total_dlmp_der_active_power.loc[:, (slice(None), seller_ders)]
    buyer_dlmp_strategic = dlmps_strategic.strategic_electric_grid_total_dlmp_der_active_power.loc[:, (slice(None), buyer_ders)]

    grid_using_price_strategic = pd.DataFrame(0, index=seller_dlmp_strategic.index,
                                    columns=pd.MultiIndex.from_product([seller_ders, buyer_ders]))
    for x, b in buyer_dlmp_strategic.columns:
        for y, s in seller_dlmp_strategic.columns:  # for y, b in grid_using_price_strategic.columns:
            grid_using_price_strategic.at[:, (s, b)] = buyer_dlmp_strategic.loc[:, (x, b)].values - seller_dlmp_strategic.loc[:,
                                                                                          (y, s)].values

    # grid_using_price_strategic *=-1

    seller_optimization_problem_sets_strategic = pd.Series(data=None, index=seller_ders, dtype=object)
    for seller in seller_optimization_problem_sets_strategic.index:
        seller_optimization_problem_sets_strategic.at[seller] = mesmo.utils.OptimizationProblem()

        # Define seller's ADMM variable
        seller_optimization_problem_sets_strategic.loc[seller].define_variable(
            f'energy_transacted_from_seller_{seller}_to_buyers', buyer=buyer_ders, timestep=scenario_data.timesteps
        )
        seller_optimization_problem_sets_strategic.loc[seller].define_variable(
            f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers', buyer=buyer_ders,
            timestep=scenario_data.timesteps
        )
        seller_optimization_problem_sets_strategic.loc[seller].define_variable(
            f'seller_{seller}_active_power_vector', timestep=scenario_data.timesteps
        )
        # Define seller's ADMMM parameter
        seller_optimization_problem_sets_strategic.loc[seller].define_parameter(
            f'admm_lambda_seller_{seller}_to_buyers_active_power',
            np.zeros(len(scenario_data.timesteps) * len(buyer_ders))
        )
        seller_optimization_problem_sets_strategic.loc[seller].define_parameter(
            f'energy_transacted_from_seller_{seller}_to_buyers_local_copy',
            np.zeros((len(scenario_data.timesteps) * len(buyer_ders), 1))
        )
        seller_optimization_problem_sets_strategic.loc[seller].define_parameter(
            f'energy_transacted_from_seller_{seller}_to_buyers_zeros',
            np.zeros((len(scenario_data.timesteps) * len(buyer_ders), 1))
        )
        seller_optimization_problem_sets_strategic.loc[seller].define_parameter(
            f'seller_{seller}_max_power',
            # np.array([1.0] * len(scenario_data.timesteps))
            np.transpose([der_model_set.der_models[seller].active_power_nominal_timeseries.values])
        )
        seller_optimization_problem_sets_strategic.loc[seller].define_parameter(
            f'seller_{seller}_min_power',
            np.array([0.0] * len(scenario_data.timesteps))
        )
        seller_optimization_problem_sets_strategic.loc[seller].define_parameter(
            f'half_of_grid_using_price_for_seller_{seller}',
            0.5 * pd.concat([grid_using_price_strategic.loc[:, (seller, buyer)] for buyer in buyer_ders]).values
        )
        seller_optimization_problem_sets_strategic.loc[seller].define_parameter(
            f'buyer_sized_ones_for_{seller}_energy_transaction',
            np.tile(np.diag(np.ones(len(scenario_data.timesteps))), len(buyer_ders))
        )

    buyer_optimization_problem_sets_strategic = pd.Series(data=None, index=buyer_ders, dtype=object)
    for buyer in buyer_optimization_problem_sets_strategic.index:
        buyer_optimization_problem_sets_strategic.loc[buyer] = mesmo.utils.OptimizationProblem()

        # Define seller's ADMM variable
        buyer_optimization_problem_sets_strategic.loc[buyer].define_variable(
            f'energy_transacted_from_sellers_to_buyer_{buyer}', seller=seller_ders, timestep=scenario_data.timesteps
        )
        buyer_optimization_problem_sets_strategic.loc[buyer].define_variable(
            f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}', seller=seller_ders,
            timestep=scenario_data.timesteps
        )
        buyer_optimization_problem_sets_strategic.loc[buyer].define_variable(
            f'buyer_{buyer}_active_power_vector', timestep=scenario_data.timesteps
        )
        # Define seller's ADMMM parameter
        buyer_optimization_problem_sets_strategic.loc[buyer].define_parameter(
            f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy',
            np.zeros((len(scenario_data.timesteps) * len(seller_ders), 1))
        )
        buyer_optimization_problem_sets_strategic.loc[buyer].define_parameter(
            f'admm_lambda_from_sellers_to_buyer_{buyer}_active_power',
            np.zeros(len(scenario_data.timesteps) * len(seller_ders))
        )
        buyer_optimization_problem_sets_strategic.loc[buyer].define_parameter(
            f'energy_transacted_from_sellers_to_buyer_{buyer}_zeros',
            np.zeros((len(scenario_data.timesteps) * len(seller_ders), 1))
        )
        buyer_optimization_problem_sets_strategic.loc[buyer].define_parameter(
            f'buyer_{buyer}_min_power',
            # np.array([1.0] * len(scenario_data.timesteps))
            np.transpose([der_model_set.der_models[buyer].active_power_nominal_timeseries.values])
        )
        buyer_optimization_problem_sets_strategic.loc[buyer].define_parameter(
            f'buyer_{buyer}_max_power',
            np.array([0.0] * len(scenario_data.timesteps))
        )
        buyer_optimization_problem_sets_strategic.loc[buyer].define_parameter(
            f'half_of_grid_using_price_for_buyer_{buyer}',
            0.5 * pd.concat([grid_using_price_strategic.loc[:, (seller, buyer)] for seller in seller_ders]).values
        )
        buyer_optimization_problem_sets_strategic.loc[buyer].define_parameter(
            f'seller_sized_ones_for_{buyer}_energy_transaction',
            np.tile(np.diag(np.ones(len(scenario_data.timesteps))), len(seller_ders))
        )

    admm_iteration_strategic = 0
    admm_rho = 800
    radius_strategic = 1
    radius_strategic_save = []
    peers_active_power_strategic = pd.DataFrame(index=scenario_data.timesteps, columns=peers)

    while radius_strategic >= 0.00005:

        # Defining optimization constraints and objectives for sellers:
        for seller in seller_optimization_problem_sets_strategic.index:
            seller_optimization_problem_sets_strategic.loc[seller].define_constraint(
                ('variable', f'buyer_sized_ones_for_{seller}_energy_transaction',
                 dict(name=f'energy_transacted_from_seller_{seller}_to_buyers',
                      buyer=buyer_ders, timestep=scenario_data.timesteps)),
                '==',
                ('variable', 1.0,
                 dict(name=f'seller_{seller}_active_power_vector'))
            )
            seller_optimization_problem_sets_strategic.loc[seller].define_constraint(
                ('variable', 1.0, dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers',
                                       buyer=buyer_ders, timestep=scenario_data.timesteps)),
                ('constant', f'energy_transacted_from_seller_{seller}_to_buyers_local_copy'),
                '==',
                ('variable', 1.0, dict(name=f'energy_transacted_from_seller_{seller}_to_buyers',
                                       buyer=buyer_ders, timestep=scenario_data.timesteps))
            )
            seller_optimization_problem_sets_strategic.loc[seller].define_constraint(
                ('variable', 1.0, dict(name=f'seller_{seller}_active_power_vector')),
                '>=',
                ('constant', f'seller_{seller}_min_power')
            )
            seller_optimization_problem_sets_strategic.loc[seller].define_constraint(
                ('variable', 1.0, dict(name=f'seller_{seller}_active_power_vector')),
                '<=',
                ('constant', f'seller_{seller}_max_power')
            )
            seller_optimization_problem_sets_strategic.loc[seller].define_constraint(
                ('variable', 1.0, dict(
                    name=f'energy_transacted_from_seller_{seller}_to_buyers')),
                '>=',
                ('constant', f'energy_transacted_from_seller_{seller}_to_buyers_zeros')
            )
            seller_optimization_problem_sets_strategic.loc[seller].define_objective(
                ('variable', f'half_of_grid_using_price_for_seller_{seller}',
                 dict(name=f'energy_transacted_from_seller_{seller}_to_buyers')),
                ('variable', f'admm_lambda_seller_{seller}_to_buyers_active_power',
                 dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers')),
                ('variable', 0.5 * admm_rho,
                 dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers'),
                 dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers'))
            )
            seller_optimization_problem_sets_strategic.loc[seller].solve()
            peers_active_power_strategic.at[:, seller] = seller_optimization_problem_sets_strategic[seller].results[
                                                             f'seller_{seller}_active_power_vector'].values / \
                                                         der_model_set.fixed_der_models[seller].active_power_nominal

        # Defining optimization constraints and objectives for sellers:
        for buyer in buyer_optimization_problem_sets_strategic.index:
            buyer_optimization_problem_sets_strategic.loc[buyer].define_constraint(
                ('variable', f'seller_sized_ones_for_{buyer}_energy_transaction',
                 dict(name=f'energy_transacted_from_sellers_to_buyer_{buyer}')),
                '==',
                ('variable', -1.0,
                 dict(name=f'buyer_{buyer}_active_power_vector'))
            )
            buyer_optimization_problem_sets_strategic.loc[buyer].define_constraint(
                ('variable', 1.0, dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}')),
                ('constant', f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy'),
                '==',
                ('variable', 1.0, dict(name=f'energy_transacted_from_sellers_to_buyer_{buyer}'))
            )
            buyer_optimization_problem_sets_strategic.loc[buyer].define_constraint(
                ('variable', 1.0, dict(name=f'buyer_{buyer}_active_power_vector')),
                '>=',
                ('constant', f'buyer_{buyer}_min_power')
            )
            buyer_optimization_problem_sets_strategic.loc[buyer].define_constraint(
                ('variable', 1.0, dict(name=f'buyer_{buyer}_active_power_vector')),
                '<=',
                ('constant', f'buyer_{buyer}_max_power')
            )
            buyer_optimization_problem_sets_strategic.loc[buyer].define_constraint(
                ('variable', 1.0, dict(
                    name=f'energy_transacted_from_sellers_to_buyer_{buyer}')),
                '>=',
                ('constant', f'energy_transacted_from_sellers_to_buyer_{buyer}_zeros')
            )
            buyer_optimization_problem_sets_strategic.loc[buyer].define_objective(
                ('variable', f'half_of_grid_using_price_for_buyer_{buyer}',
                 dict(name=f'energy_transacted_from_sellers_to_buyer_{buyer}')),
                ('variable', f'admm_lambda_from_sellers_to_buyer_{buyer}_active_power',
                 dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}')),
                ('variable', 0.5 * admm_rho,
                 dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}'),
                 dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}'))
            )
            buyer_optimization_problem_sets_strategic.loc[buyer].solve()
            peers_active_power_strategic.at[:, buyer] = buyer_optimization_problem_sets_strategic[buyer].results[
                                                            f'buyer_{buyer}_active_power_vector'].values / \
                                                        der_model_set.fixed_der_models[buyer].active_power_nominal
        optimization_strategic.define_parameter(
            'active_power_constant',
            np.concatenate([
                np.transpose([
                    der_model_set.fixed_der_models[der_name].active_power_nominal_timeseries.values
                    / (
                        der_model_set.fixed_der_models[der_name].active_power_nominal
                        if der_model_set.fixed_der_models[der_name].active_power_nominal != 0.0
                        else 1.0
                    )
                    if der_model_set.fixed_der_models[der_name].is_electric_grid_connected
                    else 0.0 * der_model_set.fixed_der_models[der_name].active_power_nominal_timeseries.values
                ])
                if der_name in der_model_set.fixed_der_names.drop(peers)
                else np.zeros((len(der_model_set.timesteps), 1))
                for der_type, der_name in der_model_set.electric_ders
            ], axis=1).ravel() + np.concatenate([
                np.transpose([
                    peers_active_power_strategic[der_name].values
                    if der_model_set.fixed_der_models[der_name].is_electric_grid_connected
                    else 0.0 * der_model_set.fixed_der_models[der_name].active_power_nominal_timeseries.values
                ])
                if der_name in peers
                else np.zeros((len(der_model_set.timesteps), 1))
                for der_type, der_name in der_model_set.electric_ders
            ], axis=1).ravel()
        )
        optimization_strategic.solve()

        dlmps_strategic = strategic_der_model_set.get_optimization_dlmps(optimization_strategic, price_data)

        seller_dlmp_strategic = dlmps_strategic.strategic_electric_grid_total_dlmp_der_active_power.loc[:,
                                (slice(None), seller_ders)]
        buyer_dlmp_strategic = dlmps_strategic.strategic_electric_grid_total_dlmp_der_active_power.loc[:,
                               (slice(None), buyer_ders)]

        for x, b in buyer_dlmp_strategic.columns:
            for y, s in seller_dlmp_strategic.columns:  # for y, b in grid_using_price_strategic.columns:
                grid_using_price_strategic.at[:, (s, b)] = buyer_dlmp_strategic.loc[:,
                                                           (x, b)].values - seller_dlmp_strategic.loc[:,
                                                                            (y, s)].values

        # Update admm parameters for seller optimization:
        for seller in seller_optimization_problem_sets_strategic.index:
            seller_optimization_problem_sets_strategic.loc[seller].parameters[
                f'energy_transacted_from_seller_{seller}_to_buyers_local_copy'] = 0.5 * np.transpose([
                pd.concat([
                    seller_optimization_problem_sets_strategic.loc[seller].results[
                        f'energy_transacted_from_seller_{seller}_to_buyers'][buyer] for buyer in buyer_ders
                ]).values + pd.concat([
                    buyer_optimization_problem_sets_strategic.loc[buyer].results[
                        f'energy_transacted_from_sellers_to_buyer_{buyer}'][seller] for buyer in buyer_ders]).values
            ])

            seller_optimization_problem_sets_strategic.loc[seller].parameters[
                f'admm_lambda_seller_{seller}_to_buyers_active_power'] += admm_rho * (pd.concat([
                seller_optimization_problem_sets_strategic.loc[seller].results[
                    f'energy_transacted_from_seller_{seller}_to_buyers'][buyer] for buyer in buyer_ders
            ]).values - seller_optimization_problem_sets_strategic.loc[seller].parameters[
                                    f'energy_transacted_from_seller_{seller}_to_buyers_local_copy'].ravel())
            seller_optimization_problem_sets_strategic.loc[seller].define_parameter(
                f'half_of_grid_using_price_for_seller_{seller}',
                0.5 * pd.concat([grid_using_price_strategic.loc[:, (seller, buyer)] for buyer in buyer_ders]).values
            )

        # Update admm parameters for buyer optimization:
        for buyer in buyer_optimization_problem_sets_strategic.index:
            buyer_optimization_problem_sets_strategic.loc[buyer].parameters[
                f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy'] = 0.5 * np.transpose([
                pd.concat([
                    buyer_optimization_problem_sets_strategic.loc[buyer].results[
                        f'energy_transacted_from_sellers_to_buyer_{buyer}'][seller] for seller in seller_ders
                ]).values + pd.concat([
                    seller_optimization_problem_sets_strategic.loc[seller].results[
                        f'energy_transacted_from_seller_{seller}_to_buyers'][buyer] for seller in
                    seller_ders]).values
            ])

            buyer_optimization_problem_sets_strategic.loc[buyer].parameters[
                f'admm_lambda_from_sellers_to_buyer_{buyer}_active_power'] += admm_rho * (pd.concat([
                buyer_optimization_problem_sets_strategic.loc[buyer].results[
                    f'energy_transacted_from_sellers_to_buyer_{buyer}'][seller] for seller in seller_ders
            ]).values - buyer_optimization_problem_sets_strategic.loc[buyer].parameters[
                                        f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy'].ravel())
            buyer_optimization_problem_sets_strategic.loc[buyer].define_parameter(
                f'half_of_grid_using_price_for_buyer_{buyer}',
                0.5 * pd.concat([grid_using_price_strategic.loc[:, (seller, buyer)] for seller in seller_ders]).values
            )


        radius_strategic = np.linalg.norm(
            np.concatenate(
            [seller_optimization_problem_sets_strategic.loc[seller].results[
                 f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers']
             for seller in seller_ders]).ravel().__abs__()
                                          + np.concatenate(
            [buyer_optimization_problem_sets_strategic.loc[buyer].results[
                 f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}']
             for buyer in buyer_ders]).ravel().__abs__()
        )

        admm_iteration_strategic += 1
        radius_strategic_save.append(radius_strategic)
        print(radius_strategic)
        print(admm_iteration_strategic)
        if admm_iteration_strategic >= admm_max_iteration:
            break

    print('Done!')

    """
    # =========================================================
    load_profile = 100 * scenario_profiles.der_definitions[('schedule_per_unit', 'mixed_commercial_residential')]
    pv_profile = 100 * scenario_profiles.der_definitions[('schedule_per_unit', 'photovoltaic_generic')]

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(8, 8))
    load_profile.plot.area(
        ax=axes[0],
        label='Mixed-commercial residential load profile',
        color='b'
        # marker='s'
    )
    pv_profile.plot.area(
        ax=axes[1],
        label='Generic PV profile',
        color='g'
        # marker='^'
    )
    # x = np.arange(len(load_profile.index))
    # axes[0].set_xticks(x)
    # axes[0].set_xticklabels(load_profile.index,
    #                             rotation=-30, fontsize=8, minor=False)
    axes[0].title.set_text('Mixed-commercial residential load profile')
    axes[0].title.set_fontsize(18)
    axes[1].title.set_text('Generic PV profile')
    axes[1].title.set_fontsize(18)
    axes[1].set_xlabel('Time [h]', fontsize=18)
    # fig.suptitle(f'Nodal Voltage Profile at {sample_time}')
    # axes[i - 1].set_ylim([0.5, 1.05])
    axes[0].set_ylabel('%', fontsize=18)
    axes[1].set_ylabel('%', fontsize=18)
    # axes[0].legend()
    axes[0].grid(axis='y')
    axes[0].grid(axis='x')
    axes[1].grid(axis='y')
    axes[1].grid(axis='x')
    axes[0].tick_params(axis='both', which='minor', labelsize=14)
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[1].tick_params(axis='both', which='minor', labelsize=14)
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    fig.set_tight_layout(True)
    # fig.suptitle('Mixed-commercial residential load and Generic PV profile', fontsize=18)
    # fig.show()
    # fig.savefig('strategic results/load_and_pv_profiles.pdf')

    gsp_price_timeseries = price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')] \
                           * 1e3/ scenario_data.scenario.at['base_apparent_power']

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 3))
    gsp_price_timeseries.plot.area(
        ax=axes,
        # label='Energy price at GSP',
        color='g'
        # marker='s'
    )
    # x = np.arange(len(load_profile.index))
    # axes[0].set_xticks(x)
    # axes[0].set_xticklabels(load_profile.index,
    #                             rotation=-30, fontsize=8, minor=False)
    axes.title.set_text('Energy price at GSP')
    axes.title.set_fontsize(14)
    axes.set_xlabel('Time [h]', fontsize=12)
    # fiuptitle(f'Nodal Voltage Profile at {sample_time}')
    # axi - 1].set_ylim([0.5, 1.05])
    axes.set_ylabel(r"$c_t^{0,p}$  [\$/kWh]", fontsize=12)
    # ax0].legend()
    axes.grid(axis='y')
    axes.grid(axis='x')
    axes.tick_params(axis='both', which='minor', labelsize=10)
    axes.tick_params(axis='both', which='major', labelsize=10)
    fig.set_tight_layout(True)
    # fig.suptitle('Mixed-commercial residential load and Generic PV profile', fontsize=18)
    # fig.show()
    # fig.savefig('strategic results/energy_price_at_gsp.pdf')
    """
    results_non_strategic = mesmo.problems.Results()
    results_non_strategic.update(linear_electric_grid_model_set.get_optimization_results(optimization_non_strategic))
    results_non_strategic.update(der_model_set.get_optimization_results(optimization_non_strategic))
    dlmps_non_strategic = linear_electric_grid_model_set.get_optimization_dlmps(optimization_non_strategic, price_data)
    flexible_der_active_power_non_strategic = results_non_strategic.der_active_power_vector_per_unit[flexible_der_type]

    results_strategic = mesmo.problems.Results()
    results_strategic.update(linear_electric_grid_model_set.get_optimization_results(optimization_strategic))
    results_strategic.update(der_model_set.get_optimization_results(optimization_strategic))
    dlmps_strategic = strategic_der_model_set.get_optimization_dlmps(optimization_strategic, price_data)
    dlmp_difference = dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power - \
                      dlmps_non_strategic.electric_grid_total_dlmp_node_active_power

    flexible_der_active_power_strategic = results_strategic.der_active_power_vector_per_unit[flexible_der_type]

    primal_active_losses = results_non_strategic.loss_active
    primal_active_losses.index = scenario_data.timesteps
    strategic_scenario_active_losses = results_strategic.loss_active
    strategic_scenario_active_losses.index = scenario_data.timesteps

    flat_time_index = np.array(['13:00', '14:00', '15:00'])
    fig, ax = plt.subplots(figsize=(5,2.3))
    ax.plot(
        primal_active_losses.values,
        label='Non-strategic scenario',
        color='b',
        marker='s'
    )
    ax.plot(
        strategic_scenario_active_losses.values,
        label='Strategic_scenario',
        color='r',
        marker='^'
    )
    ax.set_xticks(np.arange(len(flat_time_index)))
    ax.set_xticklabels(flat_time_index)
    ax.set_ylabel('Losses [p.u.]'
                  # fontsize=12
                  )
    ax.set_xlabel('Time [h]'
                  # , fontsize=12
                  )
    # ax.tick_params(which='major')
    # axes.tick_params(axis='both', which='minor', labelsize=14)
    plt.legend(ncol=2, loc='lower right')
    plt.grid(axis='y')
    plt.grid(axis='x')
    ax.title.set_text(f"Active power losses timeseries")
    fig.set_tight_layout(True)
    fig.show()
    fig.savefig('strategic results/strategic_scenario_active_losses.pdf')


    x = np.arange(len(flexible_der_active_power_non_strategic.columns))
    width = 0.35
    fig, axes = plt.subplots(1
                             # , figsize=(12, 6)
                             )
    axes.bar(x + width / 3,
             flexible_der_active_power_non_strategic.loc[report_time],
             width=width,
             color='b',
             label='Non-strategic scenario')
    axes.bar(x - width / 3,
             flexible_der_active_power_strategic.loc[report_time],
             width=width,
             color='r',
             label='Strategic scenario')
    axes.set_xticks(x, flexible_der_active_power_strategic.columns)
    plt.xlabel('DER name'
               # , fontsize=18
               )
    axes.set_ylabel('Power dispatch [p.u]'
                    # , fontsize=18
                    )
    axes.title.set_text(f"Flexible DER's active power dispatch at {report_time}")
    # axes.title.set_fontsize(18)
    fig.set_tight_layout(True)
    plt.xticks(rotation=-90
               # , fontsize=8
               )
    # plt.yticks(fontsize=10)
    axes.legend()
    axes.grid()
    fig.show()
    fig.savefig('strategic results/strategic_flexible_der_active_power_dispatch_report_time.pdf')

    # DLMPs in non-strategic scenario Timeseries for Node 10 for three phases
    # Energy portion of DLMP:

    node_860_energy_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_energy_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))]

    # Loss portion of DLMP:
    node_860_loss_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_loss_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))]

    # Voltage portion of DLMP:
    node_860_voltage_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_voltage_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))]

    # Congestion portion of DLMP:
    node_860_congestion_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_congestion_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))]

    # TODO plotting the results
    # Total DLMP:
    node_860_total_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_total_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    # ------------------DLMPs in Strategic scenario Timeseries for Node 10 for three phases------------------------:
    # Energy portion of DLMP:
    node_860_energy_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_energy_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    # Loss portion of DLMP:
    node_860_loss_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_loss_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    # Voltage portion of DLMP:
    node_860_voltage_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_voltage_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    # Congestion portion of DLMP:
    node_860_congestion_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_congestion_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    # TODO plotting the results
    # Total DLMP:
    node_860_total_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    phases = [1, 2, 3]
    fig, axes = plt.subplots(3
                             , sharex=True, sharey=True
                             , figsize=(5, 4)
                             )
    for i in phases:
        axes[i-1].plot(
            node_860_total_dlmps_non_strategic_active_power[('no_source', '860', i)].values,
            label=f'Non-strategic phase {i}',
            color='b',
            marker='s'
        )
        axes[i-1].plot(
            node_860_total_dlmps_strategic_active_power[('no_source', '860', i)].values,
            label=f'Strategic_phase_{i}',
            color='r',
            marker='^'
        )
        x = np.arange(len(node_860_total_dlmps_strategic_active_power.index))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(flat_time_index)
        #                             rotation=-30, fontsize=8, minor=False)
        axes[i - 1].title.set_text(f'Node 860 phase {i} total DLMP')
        # axes[i - 1].title.set_fontsize(18)
        axes[i - 1].set_xlabel('Time [h]'
                               # , fontsize=18
                               )
        # fig.suptitle(f'Nodal Voltage Profile at {sample_time}')
        # axes[i - 1].set_ylim([0.5, 1.05])
        axes[i - 1].set_ylabel(f'DLMP [\$/kWh]'
                               # , fontsize=18
                               )
        axes[i - 1].legend()
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
        fig.set_tight_layout(True)
    # fig.suptitle('DLMP timeseries at strategic node 860'
                 # , fontsize=20
                 # )
    fig.show()
    fig.savefig('strategic results/strategic_DLMP_time_series_at_strategic_node_860.pdf')



    # ______________________Nodal DLMPs at time 15:00:00 for non strategic scenario:_________________________________
    nodal_energy_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_energy_dlmp_node_active_power.loc[report_time] * price_correction

    # Loss portion of DLMPs:
    nodal_loss_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_loss_dlmp_node_active_power.loc[report_time] * price_correction

    # Voltage portion of DLMPs:
    nodal_voltage_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_voltage_dlmp_node_active_power.loc[report_time] * price_correction

    # Congestion portion of DLMPs:
    nodal_congestion_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_congestion_dlmp_node_active_power.loc[report_time] * price_correction

    # Total DLMPs:
    nodal_total_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_total_dlmp_node_active_power.loc[report_time] * price_correction

    # ______________________Nodal DLMPs at time 15:00:00 for non strategic scenario:_________________________________
    nodal_energy_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_energy_dlmp_node_active_power.loc[report_time] * price_correction

    # Loss portion of DLMP:
    nodal_loss_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_loss_dlmp_node_active_power.loc[report_time] * price_correction

    # Voltage portion of DLMP:
    nodal_voltage_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_voltage_dlmp_node_active_power.loc[report_time] * price_correction

    # Congestion portion of DLMP:
    nodal_congestion_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_congestion_dlmp_node_active_power.loc[report_time] * price_correction

    # Total DLMP:
    nodal_total_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power.loc[report_time] * price_correction

    for i in phases:
        x = np.arange(len(nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)].index))
        fig, axes = plt.subplots(2, sharex=True, sharey=True
                                 # , figsize=(12, 6)
                                 )
        axes[0].bar(x, nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)]
                    , label='Energy', color='g')
        axes[1].bar(x, nodal_energy_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)]
                    , label='Energy', color='g')
        axes[0].bar(x, nodal_loss_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Loss',
                    color='b')
        axes[1].bar(x, nodal_loss_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Loss',
                    color='b')
        axes[0].bar(x, nodal_voltage_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_loss_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Voltage',
                    color='r')
        axes[1].bar(x, nodal_voltage_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_loss_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Voltage',
                    color='r')
        axes[0].bar(x, nodal_congestion_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_loss_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_voltage_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Congestion',
                    color='y')
        axes[1].bar(x, nodal_congestion_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_loss_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_voltage_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Congestion',
                    color='y')
        axes[0].plot(x, nodal_total_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                     color='k', label='Total', marker='*')
        axes[1].plot(x, nodal_total_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                     color='k', label='Total', marker='*')
        axes[1].set_xticks(x, nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)].index)
        plt.xlabel('Node name')
        fig.set_tight_layout(True)
        axes[0].set_ylabel('DLMP [\$/MW]'
                           # , fontsize=18
                           )
        axes[1].set_ylabel('DLMP [\$/MW]'
                           # , fontsize=18
                           )
        # axes[0].set_ylim([0., 0.2])
        # axes[1].set_ylim([0., 0.2])
        axes[0].title.set_text(f"DSO non-strategic nodal DLMPs at {report_time} for phase {i}")
        # axes[0].title.set_fontsize(18)
        axes[1].title.set_text(f"DSO strategic nodal DLMPs at {report_time} for phase {i}")
        # axes[1].title.set_fontsize(18)
        plt.xticks(rotation=-90
                   # , fontsize=10
                   )
        # plt.yticks(fontsize=10)

        axes[0].legend(ncol=5, loc="upper left")
        axes[1].legend(ncol=5, loc="upper left")
        axes[0].grid()
        axes[1].grid()
        axes[0].set_ylim([0.09, 0.13])
        axes[1].set_ylim([0.09, 0.13])
        fig.show()
        fig.savefig(f'strategic results/strategic_contributions_to_DLMP_{i}.pdf')


    fig, axes = plt.subplots(3, sharex=False, sharey=True
                             , figsize=(7, 7)
                             )
    for i in phases:
        nodal_total_dlmps_non_strategic_active_power[(slice(None), slice(None), i)].plot(
            ax=axes[i - 1],
            label=f'Non-strategic phase {i}',
            color='b',
            marker='s'
        )
        nodal_total_dlmps_strategic_active_power[(slice(None), slice(None), i)].plot(
            ax=axes[i - 1],
            label=f'Strategic phase {i}',
            color='r',
            marker='^'
        )
        x = np.arange(len(nodal_total_dlmps_non_strategic_active_power[(slice(None), slice(None), i)].index))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(nodal_total_dlmps_non_strategic_active_power[(slice(None), slice(None), i)].index
                                    ,rotation=-90
                                    # , fontsize=10, minor=False
                                    )
        # plt.yticks(fontsize=10)
        # axes[i - 1].title.set_text(f'Nodal DLMPs at time {report_time} for phase {i} non-strategic vs strategic scenarios ')
        axes[i - 1].set_xlabel(None
                               # , fontsize=18
                               )
        axes[2].set_xlabel('Node name'
                           # , fontsize=18
                           )
        # fig.suptitle(f'Nodal Voltage Profile at {sample_time}')
        # axes[i - 1].set_ylim([0.5, 1.05])
        axes[i - 1].set_ylabel(f'Phase {i} DLMP [\$/kWh]')
        axes[i - 1].legend(ncol=2, loc="lower left")
        axes[2].legend(ncol=2, loc="upper right")
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
        fig.set_tight_layout(True)
    fig.suptitle(f'Nodal DLMPs at {report_time}')
    fig.show()
    fig.savefig('strategic results/strategic_nodal_dlmp_at_18_pm.pdf')


    # Figures for strategic DER offers:

    strategic_der_active_power_vector_strategic_scenario = results_strategic.der_active_power_vector_per_unit[
        ('flexible_generator', strategic_der_name)]
    strategic_der_active_power_vector_non_strategic_scenario = results_non_strategic.der_active_power_vector_per_unit[
        ('flexible_generator', strategic_der_name)]

    der_active_power_marginal_offers_timeseries = \
        results_non_strategic.der_active_power_marginal_offers_timeseries.loc[
        ('flexible_generator', strategic_der_name), :] / der_model_set.der_models[
            strategic_der_name].active_power_nominal * price_correction
    der_active_power_strategic_marginal_offers_timeseries = \
        dlmps_strategic.strategic_der_marginal_price_offers/ der_model_set.der_models[
            strategic_der_name].active_power_nominal * price_correction


    fig, ax = plt.subplots(figsize=(5,2.3))
    ax.plot(
        der_active_power_marginal_offers_timeseries.values,
        label='DER active power marginal cost',
        color='b',
        marker='s'
    )
    ax.plot(
        der_active_power_strategic_marginal_offers_timeseries.values,
        label='DER offered marginal cost',
        color='r',
        marker='^'
    )
    ax.set_xticks(np.arange(len(flat_time_index)))
    ax.set_xticklabels(flat_time_index)
    fig.suptitle('Offer comparison for the strategic generator')
    ax.set_ylabel('Offer price [\$/MWh]')
    ax.set_xlabel('Time [h]')
    # ax.set_ylim([0.0, 1.0])
    plt.legend()
    plt.grid(axis='y')
    plt.grid(axis='x')
    fig.set_tight_layout(True)
    fig.show()
    fig.savefig('strategic results/strategic_offers.pdf')

    fig, ax = plt.subplots(figsize=(5,2.3))
    ax.plot(
        strategic_der_active_power_vector_non_strategic_scenario.values,
        label="Non-strategic scenario ",
        color='b',
        marker='s'
    )
    ax.plot(
        strategic_der_active_power_vector_strategic_scenario.values,
        label='Strategic scenario',
        color='r',
        marker='^'
    )
    ax.set_xticks(np.arange(len(flat_time_index)))
    ax.set_xticklabels(flat_time_index)
    fig.suptitle("Strategic der's active power generation")
    ax.set_ylabel('Active power dispatched [p.u.]')
    ax.set_xlabel('Time [h]')
    # ax.set_ylim([0.0, 1.0])
    plt.legend()
    plt.grid(axis='y')
    plt.grid(axis='x')
    fig.set_tight_layout(True)
    # plt.xticks(rotation=-90
    #            # , fontsize=8
    #            )
    fig.show()
    fig.savefig('strategic results/strategic_der_active_power.pdf')

    # Plot Offer and power dispatch together:
    fig, ax = plt.subplots(
        figsize=(5,2.3)
    )
    ax.plot(
        strategic_der_active_power_vector_strategic_scenario.values,
        label="Strategic DER power dispatch",
        color='b',
        marker='s'
    )
    ax2 = ax.twinx()
    ax2.plot(
        der_active_power_strategic_marginal_offers_timeseries.values,
        label='DER marginal offer',
        color='r',
        marker='^'
    )
    fig.suptitle("Strategic der's active power generation"
                 # , fontsize=18
                 )
    ax.set_ylabel('Active power dispatched [p.u.]'
                  # , fontsize=18
                  )
    ax2.set_ylabel('Marginal cost [\$/MWh]'
                   # , fontsize=18
                   )
    ax.set_xlabel('Time [h]'
                  # , fontsize=18
                  )
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')
    plt.grid(axis='y')
    plt.grid(axis='x')
    ax.set_xticks(np.arange(len(flat_time_index)))
    ax.set_xticklabels(flat_time_index)
    # ax2.tick_params(axis='both', which='major', labelsize=14)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    fig.set_tight_layout(True)
    # plt.xticks(rotation=-90
    #            # , fontsize=12
    #            )
    fig.show()
    fig.savefig('strategic results/strategic_DER_890_active_power_offer.pdf')

    voltage_profile_non_strategic = results_non_strategic.node_voltage_magnitude_vector_per_unit.loc[report_time]
    voltage_profile_strategic = results_strategic.node_voltage_magnitude_vector_per_unit.loc[report_time]

    fig, axes = plt.subplots(3, sharey=True
                             # , sharex=True
                             , figsize=(7, 7)
                             )
    for i in [1, 2, 3]:
        voltage_profile_non_strategic[:, :, i].plot(
            ax=axes[i - 1],
            label=f'Phase {i} non-strategic',
            # y=(slice(None), slice(None), 3),
            color='b',
            marker='s'
        )
        voltage_profile_strategic[:, :, i].plot(
            ax=axes[i - 1],
            label=f'Phase {i} strategic',
            # y=(slice(None), slice(None), 3),
            color='r',
            marker='^'
        )
        x = np.arange(len(voltage_profile_non_strategic[:, :, i].index))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(voltage_profile_non_strategic[:, :, i].index
                                    , rotation=-90
                                    # , fontsize=10, minor=False
                                    )
        # plt.yticks(fontsize=10)
        # axes[i - 1].set_ylim([0.75, 1.1])
        axes[i - 1].set_ylabel('Voltage [p.u]'
                               # , fontsize=18
                               )
        axes[i - 1].set_xlabel(None
                               # , fontsize=18
                               )
        axes[2].set_xlabel('Node name'
                               # , fontsize=18
                               )
        axes[i - 1].legend(ncol=2, loc="lower center")
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
    fig.set_tight_layout(True)
    # fig.set_tight_layout(True)
    fig.suptitle(f'Nodal Voltage Profile at {report_time} non-strategic vs strategic scenario'
                 # , fontsize=20
                 )
    fig.show()
    fig.savefig('strategic results/strategic_min_voltage_profile.pdf')

    line_loading_non_strategic1 = results_non_strategic.branch_power_magnitude_vector_1_per_unit.loc[report_time]
    line_loading_strategic1 = results_strategic.branch_power_magnitude_vector_1_per_unit.loc[report_time]
    fig, axes = plt.subplots(3, sharey=True
                             ,figsize=(7, 7)
                             )
    for i in [1, 2, 3]:
        line_loading_non_strategic1.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Line loading phase {i} non-strategic',
            color='b',
            marker='s'
        )
        line_loading_strategic1.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Line loading phase {i} strategic',
            color='r',
            marker='^'
        )
        x = np.arange(len(line_loading_non_strategic1[:, :, i].index[:-2]))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(line_loading_non_strategic1[:, :, i].index[:-2]
                                    , rotation=-90
                                    # , fontsize=10, minor=False
                                    )
        # axes[i-1].set_ylim([0, 7])
        # plt.yticks(fontsize=10)
        axes[i - 1].set_ylabel('Loading [p.u]'
                               # , fontsize=18
                               )
        axes[i - 1].set_xlabel(None
                               # , fontsize=18
                               )
        axes[2].set_xlabel("Line name")
        axes[i - 1].legend(ncol=2, loc="upper left")
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
    fig.set_tight_layout(True)
    fig.suptitle(f'Line  loading in "From" direction at {report_time} non-strategic vs strategic'
                 # , fontsize=20
                 )
    fig.set_tight_layout(True)
    fig.show()
    fig.savefig('strategic results/strategic_line_loading_from.pdf')

    line_loading_non_strategic2 = results_non_strategic.branch_power_magnitude_vector_2_per_unit.loc[report_time]
    line_loading_strategic2 = results_strategic.branch_power_magnitude_vector_2_per_unit.loc[report_time]
    fig, axes = plt.subplots(3, sharey=True
                             ,figsize=(7, 7))
    for i in [1, 2, 3]:
        line_loading_non_strategic2.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Line loading phase {i} non-strategic',
            color='b',
            marker='s'
        )
        line_loading_strategic2.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Line loading phase {i} strategic',
            color='r',
            marker='^'
        )
        x = np.arange(len(line_loading_non_strategic2[:, :, i].index[:-2]))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(line_loading_non_strategic2[:, :, i].index[:-2]
                                    , rotation=-90
                                    # , fontsize=10,minor=False
                                    )
        # axes[i-1].set_ylim([0, 7])
        # plt.yticks(fontsize=10)
        axes[i - 1].set_ylabel('Loading [p.u]'
                               # , fontsize=18
                               )
        axes[i - 1].set_xlabel(None
                               # , fontsize=18
                               )
        axes[2].set_xlabel("Line name")
        axes[i - 1].legend(ncol=2, loc="upper left")
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
    fig.set_tight_layout(True)
    fig.suptitle(f'Line  loading in "To" direction at {report_time} non-strategic vs strategic scenario'
                 # , fontsize=20
                 )
    fig.set_tight_layout(True)
    fig.show()
    fig.savefig('strategic results/strategic_line_loading_to.pdf')

    # ******************************************
    # dlmps_non_strategic.electric_grid_total_dlmp_node_active_power.index = np.arange(len(scenario_data.timesteps))
    # dlmps_non_strategic.electric_grid_total_dlmp_node_active_power = \
    #     dlmps_non_strategic.electric_grid_total_dlmp_node_active_power * 1e3 / scenario_data.scenario.at['base_apparent_power']
    # dlmps_strategic.electric_grid_total_dlmp_node_active_power.index = np.arange(25)
    # dlmps_strategic.electric_grid_total_dlmp_node_active_power = \
    #     dlmps_strategic.electric_grid_total_dlmp_node_active_power * 1e3/ scenario_data.scenario.at['base_apparent_power']
    #
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
    # axes = sns.heatmap(dlmps_non_strategic.electric_grid_total_dlmp_node_active_power, annot=False, cmap="YlGnBu")
    # axes.set_ylabel('Time [h]', fontsize=20)
    # axes.set_xlabel('Node', fontsize=20)
    # axes.collections[0].colorbar.set_label("DLMP level [$/kWh]", fontsize=20)
    # fig.suptitle('Nodal DLMPs over the time horizon in non-strategic scenario', fontsize=30)
    # axes.tick_params(axis='both', which='minor', labelsize=14)
    # axes.tick_params(axis='both', which='major', labelsize=14)
    # fig.set_tight_layout(True)
    # fig.show()
    # # fig.savefig('strategic results/heatmap_DLMP nodal_timeseries_non_strategic.pdf')
    #
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
    # axes = sns.heatmap(dlmps_strategic.electric_grid_total_dlmp_node_active_power, annot=False, cmap="YlGnBu")
    # axes.set_ylabel('Time [h]', fontsize=20)
    # axes.set_xlabel('Node', fontsize=20)
    # axes.collections[0].colorbar.set_label("DLMP level [$/kWh]", fontsize=20)
    # fig.suptitle('Nodal DLMPs over the time horizon in non-strategic scenario', fontsize=30)
    # axes.tick_params(axis='both', which='minor', labelsize=14)
    # axes.tick_params(axis='both', which='major', labelsize=14)
    # fig.set_tight_layout(True)
    # fig.show()
    # fig.savefig('strategic results/heatmap_DLMP_nodal_timeseries_strategic.pdf')

    x = np.arange(len(buyer_ders))
    width = 0.2
    fig, axes = plt.subplots(ncols=1, nrows=2, sharey=True, sharex=True
                             , figsize=(5, 2.5)
                             )
    for seller in seller_ders:
        axes[seller_ders.get_loc(seller)].bar(x + width / 2,
                 seller_optimization_problem_sets_non_strategic[seller].results[
                     f'energy_transacted_from_seller_{seller}_to_buyers'].loc[report_time],
                 width=width,
                 color='b',
                 label='Non-strategic scenario')
        axes[seller_ders.get_loc(seller)].bar(x - width / 2,
                 seller_optimization_problem_sets_strategic[seller].results[
                     f'energy_transacted_from_seller_{seller}_to_buyers'].loc[report_time],
                 width=width,
                 color='r',
                 label='Strategic scenario')
        axes[seller_ders.get_loc(seller)].set_xticks(x, seller_optimization_problem_sets_non_strategic[seller].results[
                     f'energy_transacted_from_seller_{seller}_to_buyers'].columns)
        axes[seller_ders.get_loc(seller)].title.set_text(f"From seller {seller} to the byers at {report_time}")
        # axes.title.set_fontsize(18)
        axes[seller_ders.get_loc(seller)].grid()
        plt.xticks(rotation=0
                   # , fontsize=8
                   )
        # plt.yticks(fontsize=10)
        axes[seller_ders.get_loc(seller)].legend()
    plt.xlabel('Buyer name'
               # , fontsize=18
               )
    fig.supylabel('Power transacted [p.u]')
    fig.set_tight_layout(True)
    fig.show()
    fig.savefig('strategic results/energy_transacted_from_sellers_to_buyers.pdf')


    x = np.arange(len(seller_ders))
    width = 0.2
    fig, axes = plt.subplots(nrows=2, ncols=3
                             , sharey=True, sharex=True
                             , figsize=(5.5, 2.5)
                             )
    for buyer in buyer_ders:
        axes.ravel()[buyer_ders.get_loc(buyer)].bar(x + width / 2,
                 buyer_optimization_problem_sets_non_strategic[buyer].results[
                     f'energy_transacted_from_sellers_to_buyer_{buyer}'].loc[report_time],
                 width=width,
                 color='b',
                 label='Non-strategic scenario')
        axes.ravel()[buyer_ders.get_loc(buyer)].bar(x - width / 2,
                 buyer_optimization_problem_sets_strategic[buyer].results[
                     f'energy_transacted_from_sellers_to_buyer_{buyer}'].loc[report_time],
                 width=width,
                 color='r',
                 label='Strategic scenario')
        axes.ravel()[buyer_ders.get_loc(buyer)].set_xticks(x, buyer_optimization_problem_sets_non_strategic[buyer].results[
                     f'energy_transacted_from_sellers_to_buyer_{buyer}'].columns)
        axes.ravel()[buyer_ders.get_loc(buyer)].title.set_text(f"To buyer {buyer}")
        axes.ravel()[buyer_ders.get_loc(buyer)].title.set_fontsize(8)
        axes.ravel()[buyer_ders.get_loc(buyer)].grid()
    fig.set_tight_layout(True)
    plt.xticks(rotation=0
               # , fontsize=8
               )
    axes.ravel()[1].legend(loc="upper left")
    fig.supylabel('Power transacted [p.u]')
    fig.supxlabel('Buyer name')
    # fig.show()
    # fig.savefig('strategic results/strategic_scenario_active_losses.pdf')


    fig,axes = plt.subplots(2
                            ,sharex=True, sharey=True
                            ,figsize=(5,3)
                            )
    for seller in seller_ders:
        axes[seller_ders.get_loc(seller)].plot(
            price_correction * grid_using_price_non_strategic.loc[report_time, (seller, slice(None))].values,
            label=f'GUP non-strategic',
            color='b',
            marker='s'
        )
        axes[seller_ders.get_loc(seller)].plot(
            price_correction * grid_using_price_strategic.loc[report_time, (seller, slice(None))].values,
            label=f'GUP strategic',
            color='r',
            marker='^'
        )
        x = np.arange(len(buyer_ders))
        axes[seller_ders.get_loc(seller)].set_xticks(x)
        axes[seller_ders.get_loc(seller)].set_xticklabels(buyer_ders
                                    , rotation=0
                                    # , fontsize=10, minor=False
                                    )
        axes[seller_ders.get_loc(seller)].title.set_text(f"GUP from seller {seller} to the buyers")
        axes[seller_ders.get_loc(seller)].title.set_fontsize(10)
        axes[seller_ders.get_loc(seller)].grid()
    axes[0].legend(ncol=2, loc='upper right')
    axes[1].legend(ncol=2, loc='lower right')
    fig.supylabel('GUP [\$/kWh] [p.u]')
    fig.supxlabel('Buyer name')
    fig.set_tight_layout(True)
    fig.show()
    fig.savefig('strategic results/gup_from_seller_to_buyers.pdf')

    fig, axes = plt.subplots(1)
    axes.plot(
        np.array(radius_non_strategic_save),
        label=f'Non-strategic',
        color='b',
        marker='*')
    axes.plot(
        np.array(radius_strategic_save),
        label=f'Non-strategic',
        color='r',
        marker='o')
    axes.legend()
    axes.set_ylabel('Residue')
    axes.set_xlabel('Iteration')
    axes.legend(loc="upper right")
    axes.grid(axis='y')
    axes.grid(axis='x')
    fig.set_tight_layout(True)
    fig.suptitle(f'Convergence rate of ADMM in non-strategic vs strategic problems')
    fig.show()
    fig.savefig('strategic results/convergence_rate_of_ADMM.pdf')

    print(1)


if __name__ == '__main__':
    main()
