"""
This script generates input data, e.g. load profiles for the scnenarios. It is only used once, which is why it is not
part of the analysis script.
"""

import analysis.input

path_to_der_schedules_data = 'data/kerber_dorfnetz/der_schedules.csv'
path_to_der_data = 'analysis/res/additional_electric_grid_ders.csv'

scenario_names = [
    # 'nl_zuidermeer_manual',
    # 'NL_residential',
    # 'kerber_landnetz_freileitung_1',
    # 'kerber_landnetz_freileitung_2',
    # 'kerber_landnetz_kabel_1',
    # 'kerber_landnetz_kabel_2',
    'kerber_dorfnetz',
    # 'kerber_vorstadtnetz_kabel_1',
    # 'kerber_vorstadtnetz_kabel_2',
]

der_penetration_levels = {
    # 'no_penetration': 0.0,
    'low_penetration': 0.5,
    # 'high_penetration': 1.0,
}

analysis.input.generate_fixed_load_der_input_data(
    scenario_names_list=scenario_names,
    path_to_der_schedules_data=path_to_der_schedules_data,
    generate_der_models_csv=False
)

# generate scenario data for DER penetration scenarios
for scenario_name in scenario_names:
    for der_penetration in der_penetration_levels:
        analysis.input.increase_der_penetration_of_scenario_on_lv_level(
            scenario_name=scenario_name,
            path_to_der_data=path_to_der_data,
            penetration_ratio=der_penetration_levels[der_penetration],
            new_scenario_name=scenario_name + '_' + der_penetration
        )
