# Data Reference

``` warning::
    This reference is work in progress.
```

FLEDGE scenarios are defined through CSV files, where each CSV file represents a table as defined below (the file name is interpreted as the table name). Internally, FLEDGE loads all CSV files into a local SQLITE database for more convenient processing. The default location for FLEDGE scenario definitions is in the `data` directory in the repository and all CSV files in the `data` directory are automatically loaded into the database. The CSV files may be structured into sub-directories, but all files are eventually combined into the same database. Hence, all type / element identifiers must be unique across all scenario definitions.

## `electric_grid_ders`

Distributed energy resources (DERs) in the electric grid. Can define both loads (negative power) and generations (positive power). The selection of DER types will be extended in the future.

| Column | Unit | Description |
| --- |:---:| --- |
| `der_name` | | Unique DER identifier (must only be unique within the associated electric grid). |
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids`. |
| `der_type` | | DER type, which determines the type of DER model to be used. Choices: `fixed_load`, `flexible_load`, `ev_charger`, `flexible_building`. |
| `model_name` | | DER model identifier depending on the DER type, defined in `fixed_loads`, `flexible_loads`, `ev_chargers` or in CoBMo, for flexible buildings. |
| `node_name` | | Node identifier as defined in `electric_grid_nodes`. |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `connection` | | Selector for Wye / Delta connection. Choices: `wye`, `delta`. |
| `active_power` | W | Nominal active power, where loads are negative and generations are positive. |
| `reactive_power` | W | Nominal reactive power, where loads are negative and generations are positive. |

## `electric_grid_line_types`

Electric line type definitions are split into `electric_grid_line_types` for the general type definition and `electric_grid_line_types_matrices` for the definition of electric characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `line_type` | | Unique type identifier. |
| `n_phases` | - | Number of phases. |
| `maximum_current` | A | Maximum permissible current (thermal line limit). |

## `electric_grid_line_types_matrices`

Electric line characteristics are defined in terms of element property matrices. Note that the matrices are expected to be symmetric and therefore only the lower triangular matrix should be defined. The matrices are defined element-wise (indexed by row / column pairs), to allow definition of single-phase line types alongside multi-phase line types.

| Column | Unit | Description |
| --- |:---:| --- |
| `line_type` | | Line type identifier as defined in `electric_grid_line_types`. |
| `row` | | Element matrix row number (first row is `1`). |
| `col` | | Element matrix column number (first column is `1`). |
| `resistance` | Ω/km | Series resistance matrix entry. |
| `reactance` | Ω/km | Series reactance matrix entry. |
| `capacitance` | nF/km | Shunt capacitance matrix entry. |

## `electric_grid_lines`

Electric grid lines.

| Column | Unit | Description |
| --- |:---:| --- |
| `line_name` | | Unique line identifier (must only be unique within the associated electric grid). |
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids` |
| `line_type` | | Line type identifier as defined in `electric_grid_line_types`. |
| `node_1_name` | | Start node identifier as defined in `electric_grid_nodes` |
| `node_2_name` | | End node identifier as defined in `electric_grid_nodes` |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `length` | km | Line length. |

## `electric_grid_nodes`

Electric grid nodes.

| Column | Unit | Description |
| --- |:---:| --- |
| `node_name` | | Unique node identifier (must only be unique within the associated electric grid). |
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids` |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `voltage` | V | Nominal voltage. |
| `latitude` | | Latitude. |
| `longitude` | | Longitude. |

## `electric_grid_operation_limit_types`

Operation limit type definition for the electric grid. This information is utilized for the definition of the operational constraints in an optimal operation problem. The per unit definition is currently based on the nominal power flow, but may be changed in future.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_operation_limit_type` | | Unique type identifier. |
| `voltage_per_unit_minimum` | - | Minimum voltage in per unit of the nominal voltage. |
| `voltage_per_unit_maximum` | - | Maximum voltage in per unit of the nominal voltage. |
| `branch_flow_per_unit_maximum` | - | Maximum branch flow in per unit of the branch flow at nominal loading conditions. |

## `electric_grid_transformer_types`

Transformer type characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `transformer_type` | | Unique type identifier. |
| `resistance_percentage` | - | Resistance percentage. |
| `reactance_percentage` | - | Reactance percentage. |
| `tap_maximum_voltage_per_unit` | - | Maximum secondary side tap position. (Currently not used.) |
| `tap_minimum_voltage_per_unit` | - | Minimum secondary side tap position. (Currently not used.) |

## `electric_grid_transformers`

Electric grid transformers, which are limited to transformers with two windings, where the same number of phases is connected at each winding.

| Column | Unit | Description |
| --- |:---:| --- |
| `transformer_name` | | Unique transformer identifier (must only be unique within the associated electric grid).|
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids` |
| `transformer_type` | | Transformer type identifier as defined in `electric_grid_transformer_types` |
| `node_1_name` | | Primary side node name as defined in `electric_grid_nodes` |
| `node_2_name` | | Secondary side node name as defined in `electric_grid_nodes` |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `connection` | | Selector for Wye / Delta connection. Choices: `wye`, `delta`. Note that Wye-connected windings are assumed to be grounded. |
| `apparent_power` | VA | Nominal apparent power loading. |

## `electric_grids`

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | Unique electric grid identifier. |
| `source_node_name` | | Source node name as defined in `electric_grid_nodes` |
| `base_frequency` | Hz | Nominal grid frequency. |

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

Price timeseries.

| Column | Unit | Description |
| --- |:---:| --- |
| `price_type` | | Unique type identifier.|
| `time` | | Timestamp according to ISO 8601. |
| `price_value` | S$/kWh | Price value. Currently, prices are assumed to be in SGD. |

## `scenarios`

Scenario definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `scenario_name` | | Unique scenario identifier.|
| `electric_grid_name` | | Electric grid identifier as defined `electric grids` |
| `thermal_grid_name` | | Thermal grid identifier as defined `thermal grids` |
| `price_type` | | Type identifier as defined in `price_timeseries` |
| `electric_grid_operation_limit_type` | | Operation limit type as defined in `electric_grid_operation_limit_types` |
| `thermal_grid_operation_limit_type` | | Type identifier as defined in `thermal_grid_operation_limit_types` |
| `timestep_start` | | Start timestep in timestamp format according to ISO 8601. |
| `timestep_end` | | End timestep in timestamp format according to ISO 8601. |
| `timestep_interval` | | Time interval in format `HH:MM:SS` |

## `thermal_grid_cooling_plant_types`

Thermal grid cooling plant types defining the technical characteristic of the district cooling plant. Parameter names to be revised.

| Column | Unit | Description |
| --- |:---:| --- |
| `cooling_plant_type` | | Unique type identifier. |
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

Distributed energy resources (DERs) in the thermal grid. Can define both loads (negative power) and generations (positive power). The selection of DER types will be extended in the future.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Thermal grid identifier as defined in `thermal_grids`. |
| `der_name` | | Unique DER identifier (must only be unique within the associated thermal grid). |
| `node_name` | | Node identifier as defined in `thermal_grid_nodes`. |
| `der_type` | | DER type, which determines the type of DER model to be used. Choices: `flexible_building`.  |
| `model_name` | | DER model identifier depending on the DER type, defined in CoBMo for flexible buildings. |
| `thermal_power_nominal` | W | Nominal thermal power, where loads are negative and generations are positive. |

## `thermal_grid_lines`

Thermal grid pipe / line definitions.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Thermal grid identifier as defined in `thermal_grids`. |
| `line_name` | | Unique line identifier (must only be unique within the associated thermal grid). |
| `node_1_name` | | Start node identifier as defined in `thermal_grid_nodes` |
| `node_2_name` | | End node identifier as defined in `thermal_grid_nodes`. |
| `length` | km | Line length. |
| `diameter` | m | Pipe diameter. |
| `absolute_roughness` | mm | Absolute roughness of the pipe. |

## `thermal_grid_nodes`

Thermal grid nodes.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Thermal grid identifier as defined in `thermal_grids`. |
| `node_name` | | Unique node identifier (must only be unique within the associated thermal grid). |
| `node_type` | | Node type definition. Choices `source`, `no_source`. |
| `latitude` | | Latitude. |
| `longitude` | | Longitude. |

## `thermal_grid_operation_limit_types`

Thermal line limits are currently defined in per unit of the nominal thermal power solution, i.e., the thermal power flow solution for nominal loading conditions as defined in `thermal_grid_ders`.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_operation_limit_type` | | Unique type identifier. |
| `node_head_per_unit_maximum` | - | Maximum node head, in per unit of the nominal thermal power solution. |
| `pipe_flow_per_unit_maximum` | - | Maximum pipe / branch flow, in per unit of the nominal thermal power solution. |

## `thermal_grids`

Thermal grid definition. Parameters to be revised.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Unique thermal grid identifier. |
| `enthalpy_difference_distribution_water` | | |
| `enthalpy_difference_cooling_water` | | |
| `water_density` | | |
| `water_kinematic_viscosity` | | |
| `pump_efficiency_secondary_pump` | | |
| `ets_head_loss` | | |
| `pipe_velocity_maximum` | | |
| `cooling_plant_type` | | Cooling plant type identifier as defined in `thermal_grid_cooling_plant_types` |
