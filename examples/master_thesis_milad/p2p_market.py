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
        if admm_iteration_non_strategic >=50:
            break

    fig, ax = plt.subplots()
    pd.Series(radius_non_strategic_save).plot(ax=ax)
    fig.show()