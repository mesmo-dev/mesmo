BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "electric_grid_ders" (
    "der_name" TEXT,
    "electric_grid_name" TEXT,
    "der_type" TEXT,
    "model_name" TEXT,
    "node_name" TEXT,
    "is_phase_1_connected" INTEGER,
    "is_phase_2_connected" INTEGER,
    "is_phase_3_connected" INTEGER,
    "connection" TEXT,
    "active_power" REAL,
    "reactive_power" REAL,
    PRIMARY KEY("der_name","electric_grid_name")
);
CREATE TABLE IF NOT EXISTS "electric_grid_line_types" (
    "line_type" TEXT,
    "n_phases" INTEGER,
    "maximum_current" REAL,
    PRIMARY KEY("line_type")
);
CREATE TABLE IF NOT EXISTS "electric_grid_line_types_matrices" (
    "line_type" TEXT,
    "row" INTEGER,
    "col" INTEGER,
    "resistance" REAL,
    "reactance" REAL,
    "capacitance" REAL,
    PRIMARY KEY("line_type","row","col")
);
CREATE TABLE IF NOT EXISTS "electric_grid_lines" (
    "line_name" TEXT,
    "electric_grid_name" TEXT,
    "line_type" TEXT,
    "node_1_name" TEXT,
    "node_2_name" TEXT,
    "is_phase_1_connected" INTEGER,
    "is_phase_2_connected" INTEGER,
    "is_phase_3_connected" INTEGER,
    "length" REAL,
    PRIMARY KEY("line_name","electric_grid_name")
);
CREATE TABLE IF NOT EXISTS "electric_grid_nodes" (
     "node_name" TEXT,
     "electric_grid_name" TEXT,
     "is_phase_1_connected" INTEGER,
     "is_phase_2_connected" INTEGER,
     "is_phase_3_connected" INTEGER,
     "voltage" REAL,
     "latitude" REAL,
     "longitude" REAL,
     PRIMARY KEY("node_name","electric_grid_name")
);
CREATE TABLE IF NOT EXISTS "electric_grid_transformer_types" (
    "transformer_type" TEXT,
    "resistance_percentage" REAL,
    "reactance_percentage", REAL,
    "tap_maximum_voltage_per_unit" REAL,
    "tap_minimum_voltage_per_unit" REAL,
    PRIMARY KEY("transformer_type")
);
CREATE TABLE IF NOT EXISTS "electric_grid_transformers" (
    "transformer_name" TEXT,
    "electric_grid_name" TEXT,
    "transformer_type" TEXT,
    "node_1_name" TEXT,
    "node_2_name" TEXT,
    "is_phase_1_connected" INTEGER,
    "is_phase_2_connected" INTEGER,
    "is_phase_3_connected" INTEGER,
    "connection" TEXT,
    "apparent_power" REAL,
    PRIMARY KEY("electric_grid_name","transformer_name")
);
CREATE TABLE IF NOT EXISTS "electric_grids" (
    "electric_grid_name" TEXT,
    "source_node_name" TEXT,
    "base_frequency" REAL,
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
    "power_increase_percentage_maximum" REAL,
    "power_decrease_percentage_maximum" REAL,
    "time_period_power_shift_maximum" REAL,
    PRIMARY KEY("model_name")
);

CREATE TABLE IF NOT EXISTS "scenarios" (
    "scenario_name" TEXT,
    "display_id" TEXT,
    "electric_grid_name" TEXT,
    "thermal_grid_name" TEXT,
    "case_name" TEXT,
    "optimal_scheduling_problem_name" TEXT,
    "timestep_start" TEXT,
    "timestep_end" TEXT,
    "timestep_interval_seconds" TEXT,
    PRIMARY KEY("scenario_name")
);
CREATE TABLE IF NOT EXISTS "operation_problems" (
    "optimal_scheduling_problem_name" TEXT,
    "maximum_voltage_per_unit" REAL,
    "minimum_voltage_per_unit" REAL,
    "price_slack" REAL,
    "price_flexible_load" REAL,
    "maximum_voltage_imbalance" REAL,
    PRIMARY KEY("optimal_scheduling_problem_name")
);
CREATE TABLE IF NOT EXISTS "operation_load_timeseries" (
    "load_name" TEXT,
    "scenario_name" TEXT,
    "time" TEXT,
    "active_power" REAL,
    "reactive_power" REAL,
    PRIMARY KEY("load_name","scenario_name","time")
);
CREATE TABLE IF NOT EXISTS "operation_line_timeseries" (
    "line_name" TEXT,
    "scenario_name" TEXT,
    "time" TEXT,
    "phase" REAL,
    "terminal" REAL,
    "active_power" REAL,
    "reactive_power" REAL,
    PRIMARY KEY("line_name","scenario_name","time","phase","terminal")
);
CREATE TABLE IF NOT EXISTS "operation_node_timeseries" (
    "node_name" TEXT,
    "scenario_name" TEXT,
    "time" TEXT,
    "phase" TEXT,
    "voltage_magnitude" REAL,
    "voltage_angle" REAL,
    "active_power" REAL,
    "reactive_power" REAL,
    PRIMARY KEY("node_name","scenario_name","time","phase")
);
CREATE TABLE IF NOT EXISTS "price_timeseries" (
    "price_name" TEXT,
    "market_name" TEXT,
    "time" TEXT,
    "price_value" REAL,
    PRIMARY KEY("price_name","time","market_name")
);
CREATE TABLE IF NOT EXISTS "thermal_grid_cooling_plant_types" (
    "cooling_plant_type" TEXT,
    "pumping_total_efficiency" REAL,
    "pump_head_cooling_water" REAL,
    "pump_head_evaporators" REAL,
    "chiller_set_beta" REAL,
    "chiller_set_delta_temperature_cnd_min" REAL,
    "chiller_set_evaporation_temperature" REAL,
    "chiller_set_cooling_capacity" REAL,
    "cooling_water_delta_temperature" REAL,
    "cooling_tower_set_reference_temperature_cooling_water_supply" REAL,
    "cooling_tower_set_reference_temperature_wet_bulb" REAL,
    "cooling_tower_set_reference_temperature_slope" REAL,
    "cooling_tower_set_ventilation_factor" REAL,
    PRIMARY KEY("cooling_plant_type")
);
CREATE TABLE IF NOT EXISTS "thermal_grid_ders" (
    "thermal_grid_name" TEXT,
    "der_name" TEXT,
    "node_name" TEXT,
    "der_type" TEXT,
    "model_name" TEXT,
    "thermal_power_nominal" REAL,
    PRIMARY KEY("thermal_grid_name","der_name")
);
CREATE TABLE IF NOT EXISTS "thermal_grid_lines" (
    "thermal_grid_name" TEXT,
    "line_name" TEXT,
    "node_1_name" TEXT,
    "node_2_name" TEXT,
    "length" REAL,
    "diameter" REAL,
    "absolute_roughness" REAL,
    PRIMARY KEY("thermal_grid_name","line_name")
);
CREATE TABLE IF NOT EXISTS "thermal_grid_nodes" (
    "thermal_grid_name" TEXT,
    "node_name" TEXT,
    "node_type" TEXT,
    "latitude" REAL,
    "longitude" REAL,
    PRIMARY KEY("thermal_grid_name","node_name")
);
CREATE TABLE IF NOT EXISTS "thermal_grids" (
    "thermal_grid_name" TEXT,
    "enthalpy_difference_distribution_water" REAL,
    "enthalpy_difference_cooling_water" REAL,
    "water_density" REAL,
    "water_kinematic_viscosity" REAL,
    "pump_efficiency_secondary_pump" REAL,
    "ets_head_loss" REAL,
    "pipe_velocity_maximum" REAL,
    "cooling_plant_type" TEXT,
    PRIMARY KEY("thermal_grid_name")
);
COMMIT;
