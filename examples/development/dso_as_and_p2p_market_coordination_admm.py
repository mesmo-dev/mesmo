import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

pd.options.plotting.backend = "matplotlib"
import mesmo


def main():
    # TODO: Currently not working. Review limits below.

    scenario_name = 'polimi_test_case'
    # strategic_scenario = True
    admm_rho = 1e-1
    radius = 1
    admm_iteration = 0
    # results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name
                                                # , price_type='singapore_wholesale'
                                                )
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
    optimization_baseline = mesmo.utils.OptimizationProblem()

    max_branch_power = 1

    # Define electric grid problem.
    # TODO: Review limits.
    node_voltage_magnitude_vector_minimum = 0.95 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.05 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = max_branch_power * electric_grid_model.branch_power_vector_magnitude_reference

    grid_cost_coefficient = 1

    der_model_set.define_optimization_problem(optimization_baseline,
                                              price_data,
                                              state_space_model=True,
                                              kkt_conditions=False,
                                              grid_cost_coefficient=grid_cost_coefficient
                                              )

    linear_electric_grid_model_set.define_optimization_problem(
        optimization_baseline,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
        kkt_conditions=False,
        grid_cost_coefficient=grid_cost_coefficient
    )

    # Define DER problem.

    # Solve centralized optimization problem.
    optimization_baseline.solve()

    results_non_strategic = mesmo.problems.Results()
    results_non_strategic.update(linear_electric_grid_model_set.get_optimization_results(optimization_baseline))
    results_non_strategic.update(
        linear_electric_grid_model_set.get_optimization_dlmps(optimization_baseline, price_data))
    results_non_strategic.update(der_model_set.get_optimization_results(optimization_baseline))

    # Obtain results.
    flexible_der_type = ['flexible_generator', 'flexible_load']
    seller_ders = pd.Index(
        [der_name for der_type, der_name in electric_grid_model.ders if 'fixed_generator' in der_type])
    buyer_ders = pd.Index([der_name for der_type, der_name in electric_grid_model.ders if 'fixed_load' in der_type])
    seller_dlmp = results_non_strategic.electric_grid_total_dlmp_der_active_power.loc[:, (slice(None), seller_ders)]
    buyer_dlmp = results_non_strategic.electric_grid_total_dlmp_der_active_power.loc[:, (slice(None), buyer_ders)]
    der_active_power_reference = pd.Series(der_model_set.der_active_power_vector_reference,
                                           index=der_model_set.electric_ders)
    seller_der_active_power_referecne = der_active_power_reference.loc[:, seller_ders]
    buyer_der_active_power_reference = der_active_power_reference.loc[:, buyer_ders]

    grid_using_price = pd.DataFrame(0, index=seller_dlmp.index,
                                    columns=pd.MultiIndex.from_product([seller_ders, buyer_ders]))
    for x, b in buyer_dlmp.columns:
        for y, s in seller_dlmp.columns:  # for y, b in grid_using_price.columns:
            grid_using_price.at[:, (s, b)] = buyer_dlmp.loc[:, (slice(None), b)].values - seller_dlmp.loc[:,
                                                                                          (slice(None), s)].values

    # buyer_ones = sp.block_diag([[np.ones(len(buyer_ders))]] * len(scenario_data.timesteps))
    # seller_ones = sp.block_diag([[np.ones(len(seller_ders))]] * len(scenario_data.timesteps)).transpose()
    # grid_using_price = -1.0 * seller_dlmp.transpose().values @ buyer_ones + seller_ones @ buyer_dlmp.values

    optimization_centralized = mesmo.utils.OptimizationProblem()
    optimization_centralized.define_variable(
     'energy_transacted_from_sellers_to_buyers', seller=seller_ders, buyer=buyer_ders, timestep=scenario_data.timesteps
    )
    optimization_centralized.define_variable(
        'sellers_active_power_vector', seller=seller_ders, timestep=scenario_data.timesteps
    )
    optimization_centralized.define_variable(
        'buyers_active_power_vector', buyer=buyer_ders, timestep=scenario_data.timesteps
    )
    optimization_centralized.define_parameter(
        'grid_using_price',
        pd.concat([pd.concat([grid_using_price.loc[:, (s, b)] for b in buyer_ders]) for s in seller_ders]).values
    )
    optimization_centralized.define_parameter(
        'sellers_max_active_power_vector',
        np.transpose([[1] * len(seller_ders) * len(scenario_data.timesteps)])
    )
    optimization_centralized.define_parameter(
        'sellers_min_active_power_vector',
        np.transpose([[0] * len(seller_ders) * len(scenario_data.timesteps)])
    )
    optimization_centralized.define_parameter(
        'buyers_max_active_power_vector',
        np.transpose([[1] * len(buyer_ders) * len(scenario_data.timesteps)])
    )
    optimization_centralized.define_parameter(
        'buyers_min_active_power_vector',
        np.transpose([[0] * len(buyer_ders) * len(scenario_data.timesteps)])
    )
    optimization_centralized.define_parameter(
        'minimum_energy_transaction',
        np.transpose([[0] * len(seller_ders)*len(buyer_ders) * len(scenario_data.timesteps)])
    )

    for seller in seller_ders:
        optimization_centralized.define_constraint(
            ('variable', np.tile(np.diag(np.ones(len(scenario_data.timesteps))), len(buyer_ders)), dict(
                name='energy_transacted_from_sellers_to_buyers', seller=seller, buyer=buyer_ders,
                timestep=scenario_data.timesteps
            )),
            '==',
            ('variable', sp.diags(der_model_set.fixed_der_models[seller].active_power_nominal_timeseries), dict(
                name='sellers_active_power_vector', seller=seller, timestep=scenario_data.timesteps))
        )

    for buyer in buyer_ders:
        optimization_centralized.define_constraint(
            ('variable', np.tile(np.diag(np.ones(len(scenario_data.timesteps))), len(seller_ders)), dict(
                name='energy_transacted_from_sellers_to_buyers', seller=seller_ders, buyer=buyer,
                timestep=scenario_data.timesteps
            )),
            '==',
            ('variable', - 1.0 * sp.diags(der_model_set.fixed_der_models[buyer].active_power_nominal_timeseries), dict(
                name='buyers_active_power_vector', buyer=buyer, timestep=scenario_data.timesteps))
        )

    optimization_centralized.define_constraint(
        ('variable', 1.0, dict(name='sellers_active_power_vector')),
        '>=',
        ('constant', 'sellers_min_active_power_vector')
    )
    optimization_centralized.define_constraint(
        ('variable', 1.0, dict(name='sellers_active_power_vector')),
        '<=',
        ('constant', 'sellers_max_active_power_vector')
    )
    optimization_centralized.define_constraint(
        ('variable', 1.0, dict(name='buyers_active_power_vector')),
        '>=',
        ('constant', 'buyers_min_active_power_vector')
    )
    optimization_centralized.define_constraint(
        ('variable', 1.0, dict(name='buyers_active_power_vector')),
        '<=',
        ('constant', 'buyers_max_active_power_vector')
    )
    optimization_centralized.define_constraint(
        ('variable', 1.0, dict(name='energy_transacted_from_sellers_to_buyers')),
        '>=',
        ('constant', 'minimum_energy_transaction')
    )

    optimization_centralized.define_objective(
        ('variable', 'grid_using_price', dict(name='energy_transacted_from_sellers_to_buyers'))
    )
    optimization_centralized.solve()



    seller_optimization_problem_sets = pd.Series(data=None, index=seller_ders, dtype=object)
    for seller in seller_optimization_problem_sets.index:
        seller_optimization_problem_sets.at[seller] = mesmo.utils.OptimizationProblem()

        # Define seller's ADMM variable
        seller_optimization_problem_sets.loc[seller].define_variable(
            f'energy_transacted_from_seller_{seller}_to_buyers', buyer=buyer_ders, timestep=scenario_data.timesteps
        )
        seller_optimization_problem_sets.loc[seller].define_variable(
            f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers', buyer=buyer_ders,
            timestep=scenario_data.timesteps
        )
        seller_optimization_problem_sets.loc[seller].define_variable(
            f'seller_{seller}_active_power_vector', timestep=scenario_data.timesteps
        )
        # Define seller's ADMMM parameter
        seller_optimization_problem_sets.loc[seller].define_parameter(
            f'admm_lambda_seller_{seller}_to_buyers_active_power',
            np.zeros(len(scenario_data.timesteps) * len(buyer_ders))
        )
        seller_optimization_problem_sets.loc[seller].define_parameter(
            f'energy_transacted_from_seller_{seller}_to_buyers_local_copy',
            np.zeros((len(scenario_data.timesteps) * len(buyer_ders), 1))
        )
        seller_optimization_problem_sets.loc[seller].define_parameter(
            f'energy_transacted_from_seller_{seller}_to_buyers_zeros',
            np.zeros((len(scenario_data.timesteps) * len(buyer_ders), 1))
        )
        seller_optimization_problem_sets.loc[seller].define_parameter(
            f'seller_{seller}_max_power',
            np.array([1.0] * len(scenario_data.timesteps))
        )
        seller_optimization_problem_sets.loc[seller].define_parameter(
            f'seller_{seller}_min_power',
            np.array([0.0] * len(scenario_data.timesteps))
        )
        seller_optimization_problem_sets.loc[seller].define_parameter(
            f'half_of_grid_using_price_for_seller_{seller}',
            0.5 * pd.concat([grid_using_price.loc[:, (seller, buyer)] for buyer in buyer_ders]).values
        )
        seller_optimization_problem_sets.loc[seller].define_parameter(
            f'buyer_sized_ones_for_{seller}_energy_transaction',
            np.tile(np.diag(np.ones(len(scenario_data.timesteps))), len(buyer_ders))
        )

    buyer_optimization_problem_sets = pd.Series(data=None, index=buyer_ders, dtype=object)
    for buyer in buyer_optimization_problem_sets.index:
        buyer_optimization_problem_sets.loc[buyer] = mesmo.utils.OptimizationProblem()

        # Define seller's ADMM variable
        buyer_optimization_problem_sets.loc[buyer].define_variable(
            f'energy_transacted_from_sellers_to_buyer_{buyer}', seller=seller_ders, timestep=scenario_data.timesteps
        )
        buyer_optimization_problem_sets.loc[buyer].define_variable(
            f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}', seller=seller_ders,
            timestep=scenario_data.timesteps
        )
        buyer_optimization_problem_sets.loc[buyer].define_variable(
            f'buyer_{buyer}_active_power_vector', timestep=scenario_data.timesteps
        )
        # Define seller's ADMMM parameter
        buyer_optimization_problem_sets.loc[buyer].define_parameter(
            f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy',
            np.zeros((len(scenario_data.timesteps) * len(seller_ders), 1))
        )
        buyer_optimization_problem_sets.loc[buyer].define_parameter(
            f'admm_lambda_from_sellers_to_buyer_{buyer}_active_power',
            np.zeros(len(scenario_data.timesteps) * len(seller_ders))
        )
        buyer_optimization_problem_sets.loc[buyer].define_parameter(
            f'energy_transacted_from_sellers_to_buyer_{buyer}_zeros',
            np.zeros((len(scenario_data.timesteps) * len(seller_ders), 1))
        )
        buyer_optimization_problem_sets.loc[buyer].define_parameter(
            f'buyer_{buyer}_max_power',
            np.array([1.0] * len(scenario_data.timesteps))
        )
        buyer_optimization_problem_sets.loc[buyer].define_parameter(
            f'buyer_{buyer}_min_power',
            np.array([0.0] * len(scenario_data.timesteps))
        )
        buyer_optimization_problem_sets.loc[buyer].define_parameter(
            f'half_of_grid_using_price_for_buyer_{buyer}',
           0.5 * pd.concat([grid_using_price.loc[:, (seller, buyer)] for seller in seller_ders]).values
        )
        buyer_optimization_problem_sets.loc[buyer].define_parameter(
            f'seller_sized_ones_for_{buyer}_energy_transaction',
            np.tile(np.diag(np.ones(len(scenario_data.timesteps))), len(seller_ders))
        )
    while radius >= 1:
        # Defining optimization constraints and objectives for sellers:
        for seller in seller_optimization_problem_sets.index:
            seller_optimization_problem_sets.loc[seller].define_constraint(
                ('variable', f'buyer_sized_ones_for_{seller}_energy_transaction',
                 dict(name=f'energy_transacted_from_seller_{seller}_to_buyers',
                      buyer=buyer_ders, timestep=scenario_data.timesteps)),
                '==',
                ('variable', sp.diags(der_model_set.fixed_der_models[seller].active_power_nominal_timeseries),
                 dict(name=f'seller_{seller}_active_power_vector'))
            )
            seller_optimization_problem_sets.loc[seller].define_constraint(
                ('variable', 1.0, dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers',
                                       buyer=buyer_ders, timestep=scenario_data.timesteps)),
                ('constant', f'energy_transacted_from_seller_{seller}_to_buyers_local_copy'),
                '==',
                ('variable', 1.0, dict(name=f'energy_transacted_from_seller_{seller}_to_buyers',
                                       buyer=buyer_ders, timestep=scenario_data.timesteps))
            )
            seller_optimization_problem_sets.loc[seller].define_constraint(
                ('variable', 1.0, dict(name=f'seller_{seller}_active_power_vector')),
                '>=',
                ('constant', f'seller_{seller}_min_power')
            )
            seller_optimization_problem_sets.loc[seller].define_constraint(
                ('variable', 1.0, dict(name=f'seller_{seller}_active_power_vector')),
                '<=',
                ('constant', f'seller_{seller}_max_power')
            )
            seller_optimization_problem_sets.loc[seller].define_constraint(
                ('variable', 1.0, dict(
                    name=f'energy_transacted_from_seller_{seller}_to_buyers')),
                '>=',
                ('constant', f'energy_transacted_from_seller_{seller}_to_buyers_zeros')
            )
            seller_optimization_problem_sets.loc[seller].define_objective(
                ('variable', f'half_of_grid_using_price_for_seller_{seller}',
                 dict(name=f'energy_transacted_from_seller_{seller}_to_buyers')),
                ('variable', f'admm_lambda_seller_{seller}_to_buyers_active_power',
                 dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers')),
                ('variable', 0.5 * admm_rho, dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers'),
                 dict(name=f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers'))
            )
            seller_optimization_problem_sets.loc[seller].solve()

        # Defining optimization constraints and objectives for sellers:
        for buyer in buyer_optimization_problem_sets.index:
            buyer_optimization_problem_sets.loc[buyer].define_constraint(
                ('variable', f'seller_sized_ones_for_{buyer}_energy_transaction',
                 dict(name=f'energy_transacted_from_sellers_to_buyer_{buyer}')),
                '==',
                ('variable', -1.0 * sp.diags(der_model_set.fixed_der_models[buyer].active_power_nominal_timeseries),
                 dict(name=f'buyer_{buyer}_active_power_vector'))
            )
            buyer_optimization_problem_sets.loc[buyer].define_constraint(
                ('variable', 1.0, dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}')),
                ('constant', f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy'),
                '==',
                ('variable', 1.0, dict(name=f'energy_transacted_from_sellers_to_buyer_{buyer}'))
            )
            buyer_optimization_problem_sets.loc[buyer].define_constraint(
                ('variable', 1.0, dict(name=f'buyer_{buyer}_active_power_vector')),
                '>=',
                ('constant', f'buyer_{buyer}_min_power')
            )
            buyer_optimization_problem_sets.loc[buyer].define_constraint(
                ('variable', 1.0, dict(name=f'buyer_{buyer}_active_power_vector')),
                '<=',
                ('constant', f'buyer_{buyer}_max_power')
            )
            buyer_optimization_problem_sets.loc[buyer].define_constraint(
                ('variable', 1.0, dict(
                    name=f'energy_transacted_from_sellers_to_buyer_{buyer}')),
                '>=',
                ('constant', f'energy_transacted_from_sellers_to_buyer_{buyer}_zeros')
            )
            buyer_optimization_problem_sets.loc[buyer].define_objective(
                ('variable', f'half_of_grid_using_price_for_buyer_{buyer}',
                 dict(name=f'energy_transacted_from_sellers_to_buyer_{buyer}')),
                ('variable', f'admm_lambda_from_sellers_to_buyer_{buyer}_active_power',
                 dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}')),
                ('variable', 0.5 * admm_rho, dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}'),
                 dict(name=f'deviation_of_energy_transacted_from_sellers_to_buyer_{buyer}'))
            )
            buyer_optimization_problem_sets.loc[buyer].solve()

        # Update admm parameters for seller optimization:
        for seller in seller_optimization_problem_sets.index:
            seller_optimization_problem_sets.loc[seller].parameters[
                f'energy_transacted_from_seller_{seller}_to_buyers_local_copy'] = 0.5 * np.transpose([
                pd.concat([
                    seller_optimization_problem_sets.loc[seller].results[
                        f'energy_transacted_from_seller_{seller}_to_buyers'][buyer] for buyer in buyer_ders
                ]).values + pd.concat([
                    buyer_optimization_problem_sets.loc[buyer].results[
                        f'energy_transacted_from_sellers_to_buyer_{buyer}'][seller] for buyer in buyer_ders]).values
            ])

            seller_optimization_problem_sets.loc[seller].parameters[
                f'admm_lambda_seller_{seller}_to_buyers_active_power'] += admm_rho * (pd.concat([
                    seller_optimization_problem_sets.loc[seller].results[
                        f'energy_transacted_from_seller_{seller}_to_buyers'][buyer] for buyer in buyer_ders
                ]).values - seller_optimization_problem_sets.loc[seller].parameters[
                            f'energy_transacted_from_seller_{seller}_to_buyers_local_copy'].ravel())

        # Update admm parameters for buyer optimization:
        for buyer in buyer_optimization_problem_sets.index:
            buyer_optimization_problem_sets.loc[buyer].parameters[
                f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy'] = 0.5 * np.transpose([
                pd.concat([
                    buyer_optimization_problem_sets.loc[buyer].results[
                        f'energy_transacted_from_sellers_to_buyer_{buyer}'][seller] for seller in seller_ders
                ]).values + pd.concat([
                    seller_optimization_problem_sets.loc[seller].results[
                        f'energy_transacted_from_seller_{seller}_to_buyers'][buyer] for seller in seller_ders]).values
            ])

            buyer_optimization_problem_sets.loc[buyer].parameters[
                f'admm_lambda_from_sellers_to_buyer_{buyer}_active_power'] += admm_rho * (pd.concat([
                    buyer_optimization_problem_sets.loc[buyer].results[
                        f'energy_transacted_from_sellers_to_buyer_{buyer}'][seller] for seller in seller_ders
                ]).values - buyer_optimization_problem_sets.loc[buyer].parameters[
                            f'energy_transacted_from_sellers_to_buyer_{buyer}_local_copy'].ravel())

        radius = np.linalg.norm(np.concatenate(a=[seller_optimization_problem_sets.loc[seller].results[
                f'deviation_of_energy_transacted_from_seller_{seller}_to_buyers'] for seller in seller_ders]).ravel())

        admm_iteration += 1
        print(radius)
        print(admm_iteration)



    print(2)


if __name__ == '__main__':
    main()
