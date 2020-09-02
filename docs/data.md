# Data reference

``` warning::
    This reference is work in progress.
```

FLEDGE scenarios are defined through CSV files, where each CSV file represents a table as defined below (the file name is interpreted as the table name). Internally, FLEDGE loads all CSV files into a local SQLITE database for more convenient processing. The default location for FLEDGE scenario definitions is in the `data` directory in the repository and all CSV files in the `data` directory are automatically loaded into the database. The CSV files may be structured into sub-directories, but all files are eventually combined into the same database. Hence, all type / element identifiers must be unique across all scenario definitions.

## Scenario data

### `scenarios`

Scenario definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `scenario_name` | | Unique scenario identifier.|
| `electric_grid_name` | | Electric grid identifier as defined `electric grids` |
| `thermal_grid_name` | | Thermal grid identifier as defined `thermal grids` |
| `parameter_set` | | Parameter set identifier as defined in `parameters` |
| `price_type` | | Type identifier as defined in `price_timeseries` |
| `electric_grid_operation_limit_type` | | Operation limit type as defined in `electric_grid_operation_limit_types` |
| `thermal_grid_operation_limit_type` | | Type identifier as defined in `thermal_grid_operation_limit_types` |
| `timestep_start` | | Start timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_end` | | End timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_interval` | | Time interval in format `HH:MM:SS` |

### `parameters`

In all tables, a `parameter_name` string can be used to define numerical parameters in place of numerical values (identifiers or string values cannot be replaced with parameters). During model setup, those strings will be parsed from the `parameters` table to obtain the corresponding numerical values.

| Column | Unit | Description |
| --- |:---:| --- |
| `parameter_set` | | Parameter set identifier. |
| `parameter_name` | | Unique parameter identifier (must only be unique within the associated parameter set).|
| `parameter_value` | - | Parameter value. |

### `price_timeseries`

Price timeseries.

| Column | Unit | Description |
| --- |:---:| --- |
| `price_type` | | Unique type identifier.|
| `time` | | Timestamp according to ISO 8601. |
| `price_value` | $/kWh | Price value. *Currently, prices / costs are assumed to be in SGD.* |

## Electric grid data

### `electric_grids`

Electric grid definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | Unique electric grid identifier. |
| `source_node_name` | | Source node name as defined in `electric_grid_nodes` |
| `base_frequency` | Hz | Nominal grid frequency. |
| `is_single_phase_equivalent` | | Single-phase-equivalent modelling flag¹. If `0`, electric grid is modelled as multi-phase system. If `1`, electric grid is modelled as single-phase-equivalent of a three-phase balanced system. Optional column, which defaults to `0` if not explicitly defined. |

¹ If single-phase-equivalent modelling is used, all nodes, lines, transformers and DERs must be defined as single-phase elements, i.e., these elements should be connected only to phase 1. However, all power values (DER active / reactive power, transformer apparent power) must be defined as total three-phase power.

### `electric_grid_ders`

Distributed energy resources (DERs) in the electric grid. Can define both loads (negative power) and generations (positive power). The selection of DER types will be extended in the future.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids`. |
| `der_name` | | Unique DER identifier (must only be unique within the associated electric grid). |
| `der_type` | | DER type, which determines the type of DER model to be used. Choices: `fixed_load`, `flexible_load`, `fixed_ev_charger`, `flexible_building`, `fixed_generator`, `flexible_generator`, `cooling_plant`. |
| `model_name` | | DER model identifier depending on the DER type, defined in `fixed_loads`, `flexible_loads`, `fixed_ev_chargers`, `fixed_generators`, `flexible_generators` or in CoBMo, for flexible buildings. |
| `node_name` | | Node identifier as defined in `electric_grid_nodes`. |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `connection` | | Selector for Wye / Delta connection. Choices: `wye`, `delta`. |
| `active_power_nominal` | W | Nominal active power, where loads are negative and generations are positive. |
| `reactive_power_nominal` | VAr | Nominal reactive power, where loads are negative and generations are positive. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `electric_grid_line_types`

Electric line type definitions are split into `electric_grid_line_types` for the general type definition and `electric_grid_line_types_matrices` for the definition of electric characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `line_type` | | Unique type identifier. |
| `n_phases` | - | Number of phases. |
| `maximum_current` | A | Maximum permissible current (thermal line limit). |

### `electric_grid_line_types_matrices`

Electric line characteristics are defined in terms of element property matrices. Note that the matrices are expected to be symmetric and therefore only the lower triangular matrix should be defined. The matrices are defined element-wise (indexed by row / column pairs), to allow definition of single-phase line types alongside multi-phase line types.

| Column | Unit | Description |
| --- |:---:| --- |
| `line_type` | | Line type identifier as defined in `electric_grid_line_types`. |
| `row` | | Element matrix row number (first row is `1`). |
| `col` | | Element matrix column number (first column is `1`). |
| `resistance` | Ω/km | Series resistance matrix entry. |
| `reactance` | Ω/km | Series reactance matrix entry. |
| `capacitance` | nF/km | Shunt capacitance matrix entry. |

### `electric_grid_lines`

Electric grid lines.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids` |
| `line_name` | | Unique line identifier (must only be unique within the associated electric grid). |
| `line_type` | | Line type identifier as defined in `electric_grid_line_types`. |
| `node_1_name` | | Start node identifier as defined in `electric_grid_nodes` |
| `node_2_name` | | End node identifier as defined in `electric_grid_nodes` |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `length` | km | Line length. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `electric_grid_nodes`

Electric grid nodes.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids` |
| `node_name` | | Unique node identifier (must only be unique within the associated electric grid). |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `voltage` | V | Nominal voltage. |
| `latitude` | | Latitude. |
| `longitude` | | Longitude. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `electric_grid_operation_limit_types`

Operation limit type definition for the electric grid. This information is utilized for the definition of the operational constraints in an optimal operation problem. The per unit definition is currently based on the nominal power flow, but may be changed in future.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_operation_limit_type` | | Unique type identifier. |
| `voltage_per_unit_minimum` | - | Minimum voltage in per unit of the nominal voltage. |
| `voltage_per_unit_maximum` | - | Maximum voltage in per unit of the nominal voltage. |
| `branch_flow_per_unit_maximum` | - | Maximum branch flow in per unit of the branch flow at nominal loading conditions. |

### `electric_grid_transformer_types`

Transformer type characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `transformer_type` | | Unique type identifier. |
| `resistance_percentage` | - | Resistance percentage. |
| `reactance_percentage` | - | Reactance percentage. |
| `tap_maximum_voltage_per_unit` | - | Maximum secondary side tap position. (Currently not used.) |
| `tap_minimum_voltage_per_unit` | - | Minimum secondary side tap position. (Currently not used.) |

### `electric_grid_transformers`

Electric grid transformers, which are limited to transformers with two windings, where the same number of phases is connected at each winding.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids` |
| `transformer_name` | | Unique transformer identifier (must only be unique within the associated electric grid).|
| `transformer_type` | | Transformer type identifier as defined in `electric_grid_transformer_types` |
| `node_1_name` | | Primary side node name as defined in `electric_grid_nodes` |
| `node_2_name` | | Secondary side node name as defined in `electric_grid_nodes` |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `connection` | | Selector for Wye / Delta connection. Choices: `wye`, `delta`. Note that Wye-connected windings are assumed to be grounded. |
| `apparent_power` | VA | Nominal apparent power loading. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

## Thermal grid data

### `thermal_grids`

Thermal grid definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Unique thermal grid identifier. |
| `source_node_name` | | Source node name as defined in `thermal_grid_nodes`. |
| `distribution_pump_efficiency` | - | Pump efficiency (pump power / electric power) of the secondary side pumps, i.e. the pumps in the distribution system / thermal grid. |
| `energy_transfer_station_head_loss` | m | Pump pressure head loss in the energy transfer station at each DER. |
| `enthalpy_difference_distribution_water` | J/kg | Enthalpy difference between supply and return side of the distribution water, i.e. the water flowing to the thermal grid. |
| `water_density` | kg/m³ | Density of the distribution water. |
| `water_kinematic_viscosity` | m²/s | Kinematic viscosity of the distribution water. |
| `plant_type` | | Thermal supply plant type. Currently only `cooling_plant` is supported. |
| `plant_model_name` | | Plant model identifier. If plant type `cooling_plant`, as defined in `cooling_plants`. |

### `thermal_grid_ders`

Distributed energy resources (DERs) in the thermal grid. Can define both loads (negative power) and generations (positive power). The selection of DER types will be extended in the future.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Thermal grid identifier as defined in `thermal_grids`. |
| `der_name` | | Unique DER identifier (must only be unique within the associated thermal grid). |
| `node_name` | | Node identifier as defined in `thermal_grid_nodes`. |
| `der_type` | | DER type, which determines the type of DER model to be used. Choices: `flexible_building`, `fixed_generator`, `flexible_generator`, `cooling_plant`.  |
| `model_name` | | DER model identifier depending on the DER type, defined in `fixed_generators`, `flexible_generators` or CoBMo for flexible buildings. |
| `thermal_power_nominal` | W | Nominal thermal power, where loads are negative and generations are positive. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `thermal_grid_line_types`

Thermal line types for defining pipe characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `line_type` | | Unique type identifier. |
| `diameter` | m | Pipe diameter. |
| `absolute_roughness` | mm | Absolute roughness of the pipe. |
| `maximum_velocity` | m/s | Nominal maximum pipe velocity. |

### `thermal_grid_lines`

Thermal grid line (pipe) definitions. The definition only includes the supply side piping, as the return side is assumed be symmetric.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Thermal grid identifier as defined in `thermal_grids`. |
| `line_name` | | Unique line identifier (must only be unique within the associated thermal grid). |
| `line_type` | | Line type identifier as defined in `thermal_grid_line_types`. |
| `node_1_name` | | Start node identifier as defined in `thermal_grid_nodes` |
| `node_2_name` | | End node identifier as defined in `thermal_grid_nodes`. |
| `length` | km | Line length. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `thermal_grid_nodes`

Thermal grid nodes.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Thermal grid identifier as defined in `thermal_grids`. |
| `node_name` | | Unique node identifier (must only be unique within the associated thermal grid). |
| `latitude` | | Latitude. |
| `longitude` | | Longitude. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `thermal_grid_operation_limit_types`

Thermal line limits are currently defined in per unit of the nominal thermal power solution, i.e., the thermal power flow solution for nominal loading conditions as defined in `thermal_grid_ders`.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_operation_limit_type` | | Unique type identifier. |
| `node_head_per_unit_maximum` | - | Maximum node head, in per unit of the nominal thermal power solution. |
| `pipe_flow_per_unit_maximum` | - | Maximum pipe / branch flow, in per unit of the nominal thermal power solution. |

## Distributed energy resource (DER) data

For each DER type which requires the definition of timeseries values, these can be defined either directly as timeseries or through as a schedule. When defining by schedule, the timeseries is constructed by obtaining the appropriate values based on the `time_period` in `ddTHH:MM` format. Each value is kept constant at the given value for any daytime greater than or equal to `HH:MM` and any weekday greater than or equal to `dd` until the next defined `ddTHH:MM`. Note that the daily schedule is repeated for any weekday greater than or equal to `dd` until the next defined `dd`. The initial value for each `zone_constraint_profile` must start at `time_period = 01T00:00`.

Furthermore, the active / reactive / thermal power values can be defined as absolute values or in per unit values. Per unit values are assumed to be in per unit of the nominal active / reactive power as defined `electric_grid_ders`. Note that the sign of the active / reactive / thermal power values in the timeseries / schedule definition are ignored and superseded by the sign of the nominal active / reactive / thermal power value as defined in `electric_grid_ders` and `thermal_grid_ders`, where positive values are interpreted as generation and negative values as consumption.

### `cooling_plants`

Cooling plants for modelling distributed generation facilities / heat pumps in the thermal grid. Cooling plants are connected to both electric and thermal grid, therefore must be defined both in `electric_grid_ders` and `thermal_grid_ders`.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier (corresponding to `electric_grid_ders` / `thermal_grid_ders`). |
| `cooling_efficiency` | | Coefficient of performance (COP). |
| `plant_pump_efficiency` | - | Pump efficiency (pump power / electric power) of the primary side pumps, i.e. the pumps within the district cooling plant. |
| `condenser_pump_head` | m | Pump pressure head across the condenser. |
| `evaporator_pump_head` | m | Pump pressure head across the evaporator. |
| `chiller_set_beta` | - | Chiller set model beta factor, used to model the chiller efficiency. |
| `chiller_set_condenser_minimum_temperature_difference` | K | Chiller set minimum temperature difference at the condenser, i.e. between the condenser water cycle and chiller refrigerant cycle. |
| `chiller_set_evaporation_temperature` | K | Chiller set evaporation temperature. |
| `chiller_set_cooling_capacity` | W | Chiller nominal maximum cooling capacity. |
| `condenser_water_temperature_difference` | K | Condenser water temperature difference. |
| `condenser_water_enthalpy_difference` | J/kg | Condenser water enthalpy difference. |
| `cooling_tower_set_reference_temperature_condenser_water` | °C | Cooling tower set reference temperature for the condenser water, i.e. the temperature at which condenser water leaves the cooling tower. |
| `cooling_tower_set_reference_temperature_wet_bulb` | °C | Cooling tower set reference temperature for the wet bulb ambient air temperature. |
| `cooling_tower_set_reference_temperature_slope` | °C | Cooling tower reference temperature slope, used to model the cooling tower efficiency. |
| `cooling_tower_set_ventilation_factor` | - | Cooling tower set ventilation factor, used to model the ventilation requirements depending on the condenser water flow. |

### `fixed_ev_chargers`

EV charger model definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier (corresponding to `electric_grid_ders`). |
| `definition_type` | | DER definition type selector. Choices: `timeseries` (Defined as timeseries.) `schedule` (Defined as schedule.), `timeseries_per_unit` (Defined as timeseries in per unit values.), `schedule_per_unit` (Defined as schedule in per unit values.) |

### `fixed_ev_charger_schedules`

EV charger schedules definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier. |
| `time_period` | | Time period in `ddTHH:MM` format. `dd` is the weekday (`01` - Monday ... `07` - Sunday). `T` is the divider for date and time information according to ISO 8601. `HH:MM` is the daytime. |
| `active_power` | W | Active power value. |
| `reactive_power` | VAr | Reactive power value. |

### `fixed_ev_charger_timeseries`

EV charger timeseries definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier. |
| `time` | | Timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `active_power` | W | Active power value. |
| `reactive_power` | VAr | Reactive power value. |

### `fixed_loads`

Fixed load model definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier (corresponding to `electric_grid_ders`). |
| `definition_type` | | DER definition type selector. Choices: `timeseries` (Defined as timeseries.) `schedule` (Defined as schedule.), `timeseries_per_unit` (Defined as timeseries in per unit values.), `schedule_per_unit` (Defined as schedule in per unit values.) |

### `fixed_load_schedules`

Fixed load schedules definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier. |
| `time_period` | | Time period in `ddTHH:MM` format. `dd` is the weekday (`01` - Monday ... `07` - Sunday). `T` is the divider for date and time information according to ISO 8601. `HH:MM` is the daytime. |
| `active_power` | W | Active power value. |
| `reactive_power` | VAr | Reactive power value. |

### `fixed_load_timeseries`

Fixed load timeseries definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier. |
| `time` | | Timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `active_power` | W | Active power value. |
| `reactive_power` | VAr | Reactive power value. |

### `fixed_generators`

Fixed load model definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier (corresponding to `electric_grid_ders`). |
| `definition_type` | | DER definition type selector. Choices: `timeseries` (Defined as timeseries.) `schedule` (Defined as schedule.), `timeseries_per_unit` (Defined as timeseries in per unit values.), `schedule_per_unit` (Defined as schedule in per unit values.) |
| `levelized_cost_of_energy` | $/kWh | Leveled cost of energy. *Currently, prices / costs are assumed to be in SGD.* |

### `fixed_generator_schedules`

Fixed load schedules definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier. |
| `time_period` | | Time period in `ddTHH:MM` format. `dd` is the weekday (`01` - Monday ... `07` - Sunday). `T` is the divider for date and time information according to ISO 8601. `HH:MM` is the daytime. |
| `active_power` | W | Active power value. |
| `reactive_power` | VAr | Reactive power value. |

### `fixed_generator_timeseries`

Fixed load timeseries definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier. |
| `time` | | Timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `active_power` | W | Active power value. |
| `reactive_power` | VAr | Reactive power value. |

### `flexible_loads`

Flexible load model definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier (corresponding to `electric_grid_ders`). |
| `definition_type` | | DER definition type selector. Choices: `timeseries` (Defined as timeseries.) `schedule` (Defined as schedule.), `timeseries_per_unit` (Defined as timeseries in per unit values.), `schedule_per_unit` (Defined as schedule in per unit values.) |
| `power_increase_percentage_maximum` | - | Maximum permitted per unit power increase in each timestep. *To be revised* |
| `power_decrease_percentage_maximum` | - | Maximum permitted per unit power decrease in each timestep. *To be revised* |
| `time_period_power_shift_maximum` | - | Number timesteps for which energy consumption can be deferred or advanced. *To be revised* |

### `flexible_load_schedules`

Flexible load schedules definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier. |
| `time_period` | | Time period in `ddTHH:MM` format. `dd` is the weekday (`01` - Monday ... `07` - Sunday). `T` is the divider for date and time information according to ISO 8601. `HH:MM` is the daytime. |
| `active_power` | W | Active power value. |
| `reactive_power` | VAr | Reactive power value. |

### `flexible_load_timeseries`

Flexible load timeseries definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier. |
| `time` | | Timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `active_power` | W | Active power value. |
| `reactive_power` | VAr | Reactive power value. |

### `flexible_generators`

Fixed load model definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier (corresponding to `electric_grid_ders`). |
| `definition_type` | | DER definition type selector. Choices: `timeseries` (Defined as timeseries.) `schedule` (Defined as schedule.), `timeseries_per_unit` (Defined as timeseries in per unit values.), `schedule_per_unit` (Defined as schedule in per unit values.) |
| `levelized_cost_of_energy` | $/kWh | Leveled cost of energy. *Currently, prices / costs are assumed to be in SGD.* |

### `flexible_generator_schedules`

Fixed load schedules definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier. |
| `time_period` | | Time period in `ddTHH:MM` format. `dd` is the weekday (`01` - Monday ... `07` - Sunday). `T` is the divider for date and time information according to ISO 8601. `HH:MM` is the daytime. |
| `active_power` | W | Active power value. |
| `reactive_power` | VAr | Reactive power value. |

### `flexible_generator_timeseries`

Fixed load timeseries definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `model_name` | | DER model identifier. |
| `time` | | Timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `active_power` | W | Active power value. |
| `reactive_power` | VAr | Reactive power value. |
