# Database Reference

``` warning::
    This reference is work in progress.
```

## `electric_grid_ders`

Distributed energy resources (DERs) in the electric grid. Can define both loads (negative power) and generators (positive power).

| Column | Unit | Description |
| --- |:---:| --- |
| `der_name` | | |
| `electric_grid_name` | | |
| `der_type` | | |
| `model_name` | | |
| `node_name` | | |
| `is_phase_1_connected` | | |
| `is_phase_2_connected` | | |
| `is_phase_3_connected` | | |
| `connection` | | |
| `active_power` | | |
| `reactive_power` | | |

## `electric_grid_line_types`

| Column | Unit | Description |
| --- |:---:| --- |
| `line_type` | | |
| `n_phases` | | |
| `maximum_current` | | |

## `electric_grid_line_types_matrices`

| Column | Unit | Description |
| --- |:---:| --- |
| `line_type` | | |
| `row` | | |
| `col` | | |
| `resistance` | | |
| `reactance` | | |
| `capacitance` | | |

## `electric_grid_lines`

| Column | Unit | Description |
| --- |:---:| --- |
| `line_name` | | |
| `electric_grid_name` | | |
| `line_type` | | |
| `node_1_name` | | |
| `node_2_name` | | |
| `is_phase_1_connected` | | |
| `is_phase_2_connected` | | |
| `is_phase_3_connected` | | |
| `length` | | |

## `electric_grid_nodes`

| Column | Unit | Description |
| --- |:---:| --- |
| `node_name` | | |
| `electric_grid_name` | | |
| `is_phase_1_connected` | | |
| `is_phase_2_connected` | | |
| `is_phase_3_connected` | | |
| `voltage` | | |
| `latitude` | | |
| `longitude` | | |

## `electric_grid_transformer_types`

| Column | Unit | Description |
| --- |:---:| --- |
| `transformer_type` | | |
| `resistance_percentage` | | |
| `reactance_percentage` | | |
| `tap_maximum_voltage_per_unit` | | |
| `tap_minimum_voltage_per_unit` | | |

## `electric_grid_transformers`

- Transformers are assumed to have same number of phases on each winding.
- Transformer can only have two windings.
- Wye-connected windings are assumed to be grounded.

| Column | Unit | Description |
| --- |:---:| --- |
| `transformer_name` | | |
| `electric_grid_name` | | |
| `transformer_type` | | |
| `node_1_name` | | |
| `node_2_name` | | |
| `is_phase_1_connected` | | |
| `is_phase_2_connected` | | |
| `is_phase_3_connected` | | |
| `connection` | | |
| `apparent_power` | | |

## `electric_grids`

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | |
| `source_node_name` | | |
| `base_frequency` | | |

## `ev_charger_timeseries`

| Column | Unit | Description |
| --- |:---:| --- |
| `timeseries_name` | | |
| `time` | | |
| `apparent_power_absolute` | | |
| `apparent_power_per_unit` | | |

## `ev_chargers`

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | |
| `timeseries_name` | | |
| `definition_type` | | |

## `fixed_load_timeseries`

| Column | Unit | Description |
| --- |:---:| --- |
| `timeseries_name` | | |
| `time` | | |
| `apparent_power_absolute` | | |
| `apparent_power_per_unit` | | |

## `fixed_loads`

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | |
| `timeseries_name` | | |
| `definition_type` | | |

## `flexible_load_timeseries`

| Column | Unit | Description |
| --- |:---:| --- |
| `timeseries_name` | | |
| `time` | | |
| `apparent_power_absolute` | | |
| `apparent_power_per_unit` | | |

## `flexible_loads`

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | |
| `timeseries_name` | | |
| `definition_type` | | |
| `power_increase_percentage_maximum` | | |
| `power_decrease_percentage_maximum` | | |
| `time_period_power_shift_maximum` | | |

## `price_timeseries`

| Column | Unit | Description |
| --- |:---:| --- |
| `price_type` | | |
| `time` | | |
| `price_value` | | |

## `scenarios`

| Column | Unit | Description |
| --- |:---:| --- |
| `scenario_name` | | |
| `electric_grid_name` | | |
| `thermal_grid_name` | | |
| `timestep_start` | | |
| `timestep_end` | | |
| `timestep_interval` | | |

## `thermal_grid_cooling_plant_types`

| Column | Unit | Description |
| --- |:---:| --- |
| `cooling_plant_type` | | |
| `pumping_total_efficiency` | | |
| `pump_head_cooling_water` | | |
| `pump_head_evaporators` | | |
| `chiller_set_beta` | | |
| `chiller_set_delta_temperature_cnd_min` | | |
| `chiller_set_evaporation_temperature` | | |
| `chiller_set_cooling_capacity` | | |
| `cooling_water_delta_temperature` | | |
| `cooling_tower_set_reference_temperature_cooling_water_supply` | | |
| `cooling_tower_set_reference_temperature_wet_bulb` | | |
| `cooling_tower_set_reference_temperature_slope` | | |
| `cooling_tower_set_ventilation_factor` | | |

## `thermal_grid_ders`

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | |
| `der_name` | | |
| `node_name` | | |
| `der_type` | | |
| `model_name` | | |
| `thermal_power_nominal` | | |

## `thermal_grid_lines`

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | |
| `line_name` | | |
| `node_1_name` | | |
| `node_2_name` | | |
| `length` | | |
| `diameter` | | |
| `absolute_roughness` | | |

## `thermal_grid_nodes`

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | |
| `node_name` | | |
| `node_type` | | |
| `latitude` | | |
| `longitude` | | |

## `thermal_grids`

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | |
| `enthalpy_difference_distribution_water` | | |
| `enthalpy_difference_cooling_water` | | |
| `water_density` | | |
| `water_kinematic_viscosity` | | |
| `pump_efficiency_secondary_pump` | | |
| `ets_head_loss` | | |
| `pipe_velocity_maximum` | | |
| `cooling_plant_type` | | |
