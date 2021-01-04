"""
This script generates input data, e.g. load profiles for the scnenarios. It is only used once, which is why it is not
part of the analysis script.
"""

import analysis.input

path_to_der_schedules_data = 'data/kerber_dorfnetz/der_schedules.csv'

scenario_names = [
    # 'NL_residential'
    # 'kerber_landnetz_freileitung_1',
    'kerber_landnetz_freileitung_2',
    # 'kerber_landnetz_kabel_1',
    # 'kerber_landnetz_kabel_2',
    # 'kerber_dorfnetz',
    # 'kerber_vorstadtnetz_kabel_1',
    # 'kerber_vorstadtnetz_kabel_2',
]

analysis.input.generate_fixed_load_der_input_data(
    scenario_names_list=scenario_names,
    path_to_der_schedules_data=path_to_der_schedules_data
)
