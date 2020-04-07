BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "electric_grid_ders" (
    "der_name" TEXT,
    "electric_grid_name" TEXT,
    "der_type" TEXT,
    "model_name" TEXT,
    "node_name" TEXT,
    "is_phase_1_connected" TEXT,
    "is_phase_2_connected" TEXT,
    "is_phase_3_connected" TEXT,
    "connection" TEXT,
    "active_power" TEXT,
    "reactive_power" TEXT,
    PRIMARY KEY("der_name","electric_grid_name")
);
CREATE TABLE IF NOT EXISTS "electric_grid_line_types" (
    "line_type" TEXT,
    "n_phases" TEXT,
    "maximum_current" TEXT,
    PRIMARY KEY("line_type")
);
CREATE TABLE IF NOT EXISTS "electric_grid_line_types_matrices" (
    "line_type" TEXT,
    "row" INTEGER,
    "col" INTEGER,
    "resistance" TEXT,
    "reactance" TEXT,
    "capacitance" TEXT,
    PRIMARY KEY("line_type","row","col")
);
CREATE TABLE IF NOT EXISTS "electric_grid_lines" (
    "line_name" TEXT,
    "electric_grid_name" TEXT,
    "line_type" TEXT,
    "node_1_name" TEXT,
    "node_2_name" TEXT,
    "is_phase_1_connected" TEXT,
    "is_phase_2_connected" TEXT,
    "is_phase_3_connected" TEXT,
    "length" TEXT,
    PRIMARY KEY("line_name","electric_grid_name")
);
CREATE TABLE IF NOT EXISTS "electric_grid_nodes" (
     "node_name" TEXT,
     "electric_grid_name" TEXT,
     "is_phase_1_connected" TEXT,
     "is_phase_2_connected" TEXT,
     "is_phase_3_connected" TEXT,
     "voltage" TEXT,
     "latitude" TEXT,
     "longitude" TEXT,
     PRIMARY KEY("node_name","electric_grid_name")
);
CREATE TABLE IF NOT EXISTS "electric_grid_transformer_types" (
    "transformer_type" TEXT,
    "resistance_percentage" TEXT,
    "reactance_percentage" TEXT,
    "tap_maximum_voltage_per_unit" TEXT,
    "tap_minimum_voltage_per_unit" TEXT,
    PRIMARY KEY("transformer_type")
);
CREATE TABLE IF NOT EXISTS "electric_grid_transformers" (
    "transformer_name" TEXT,
    "electric_grid_name" TEXT,
    "transformer_type" TEXT,
    "node_1_name" TEXT,
    "node_2_name" TEXT,
    "is_phase_1_connected" TEXT,
    "is_phase_2_connected" TEXT,
    "is_phase_3_connected" TEXT,
    "connection" TEXT,
    "apparent_power" TEXT,
    PRIMARY KEY("electric_grid_name","transformer_name")
);
CREATE TABLE IF NOT EXISTS "electric_grids" (
    "electric_grid_name" TEXT,
    "source_node_name" TEXT,
    "base_frequency" TEXT,
    PRIMARY KEY("electric_grid_name")
);
CREATE TABLE IF NOT EXISTS "ev_charger_timeseries" (
    "timeseries_name" TEXT,
    "time" TEXT,
    "apparent_power_absolute" REAL,
    "apparent_power_per_unit" REAL,
    PRIMARY KEY("timeseries_name","time")
);
CREATE TABLE IF NOT EXISTS "ev_chargers" (
    "model_name" TEXT,
    "timeseries_name" TEXT,
    "definition_type" TEXT,
    PRIMARY KEY("model_name")
);
CREATE TABLE IF NOT EXISTS "fixed_load_timeseries" (
    "timeseries_name" TEXT,
    "time" TEXT,
    "apparent_power_absolute" REAL,
    "apparent_power_per_unit" REAL,
    PRIMARY KEY("timeseries_name","time")
);
CREATE TABLE IF NOT EXISTS "fixed_loads" (
     "model_name" TEXT,
     "timeseries_name" TEXT,
     "definition_type" TEXT,
     PRIMARY KEY("model_name")
);
CREATE TABLE IF NOT EXISTS "flexible_load_timeseries" (
    "timeseries_name" TEXT,
    "time" TEXT,
    "apparent_power_absolute" REAL,
    "apparent_power_per_unit" REAL,
    PRIMARY KEY("timeseries_name","time")
);
CREATE TABLE IF NOT EXISTS "flexible_loads" (
    "model_name" TEXT,
    "timeseries_name" TEXT,
    "definition_type" TEXT,
    "power_increase_percentage_maximum" TEXT,
    "power_decrease_percentage_maximum" TEXT,
    "time_period_power_shift_maximum" TEXT,
    PRIMARY KEY("model_name")
);
CREATE TABLE IF NOT EXISTS "parameters" (
    "parameter_set" TEXT,
    "parameter_name" TEXT,
    "parameter_value" REAL,
    PRIMARY KEY("parameter_set","parameter_name")
);
CREATE TABLE IF NOT EXISTS "price_timeseries" (
    "price_type" TEXT,
    "time" TEXT,
    "price_value" REAL,
    PRIMARY KEY("price_type","time")
);
CREATE TABLE IF NOT EXISTS "scenarios" (
    "scenario_name" TEXT,
    "electric_grid_name" TEXT,
    "thermal_grid_name" TEXT,
    "parameter_set" TEXT,
    "timestep_start" TEXT,
    "timestep_end" TEXT,
    "timestep_interval" TEXT,
    PRIMARY KEY("scenario_name")
);
CREATE TABLE IF NOT EXISTS "thermal_grid_cooling_plant_types" (
    "cooling_plant_type" TEXT,
    "pumping_total_efficiency" TEXT,
    "pump_head_cooling_water" TEXT,
    "pump_head_evaporators" TEXT,
    "chiller_set_beta" TEXT,
    "chiller_set_delta_temperature_cnd_min" TEXT,
    "chiller_set_evaporation_temperature" TEXT,
    "chiller_set_cooling_capacity" TEXT,
    "cooling_water_delta_temperature" TEXT,
    "cooling_tower_set_reference_temperature_cooling_water_supply" TEXT,
    "cooling_tower_set_reference_temperature_wet_bulb" TEXT,
    "cooling_tower_set_reference_temperature_slope" TEXT,
    "cooling_tower_set_ventilation_factor" TEXT,
    PRIMARY KEY("cooling_plant_type")
);
CREATE TABLE IF NOT EXISTS "thermal_grid_ders" (
    "thermal_grid_name" TEXT,
    "der_name" TEXT,
    "node_name" TEXT,
    "der_type" TEXT,
    "model_name" TEXT,
    "thermal_power_nominal" TEXT,
    PRIMARY KEY("thermal_grid_name","der_name")
);
CREATE TABLE IF NOT EXISTS "thermal_grid_lines" (
    "thermal_grid_name" TEXT,
    "line_name" TEXT,
    "node_1_name" TEXT,
    "node_2_name" TEXT,
    "length" TEXT,
    "diameter" TEXT,
    "absolute_roughness" TEXT,
    PRIMARY KEY("thermal_grid_name","line_name")
);
CREATE TABLE IF NOT EXISTS "thermal_grid_nodes" (
    "thermal_grid_name" TEXT,
    "node_name" TEXT,
    "node_type" TEXT,
    "latitude" TEXT,
    "longitude" TEXT,
    PRIMARY KEY("thermal_grid_name","node_name")
);
CREATE TABLE IF NOT EXISTS "thermal_grids" (
    "thermal_grid_name" TEXT,
    "enthalpy_difference_distribution_water" TEXT,
    "enthalpy_difference_cooling_water" TEXT,
    "water_density" TEXT,
    "water_kinematic_viscosity" TEXT,
    "pump_efficiency_secondary_pump" TEXT,
    "ets_head_loss" TEXT,
    "pipe_velocity_maximum" TEXT,
    "cooling_plant_type" TEXT,
    PRIMARY KEY("thermal_grid_name")
);
COMMIT;
