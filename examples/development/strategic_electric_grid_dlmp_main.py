"""Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mesmo.kkt_conditions_with_state_space import StrategicMarket
import mesmo


def main():

    # TODO: Currently not working. Review limits below.

    # scenarios = [None]
    # scenario_name = "strategic_dso_market"
    scenario_name = 'strategic_market_19_node'
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name
                                                # , price_type='singapore_wholesale'
                                                )
    price_data.price_sensitivity_coefficient = 1e-6

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
    optimization_centralized = mesmo.utils.OptimizationProblem()

    # Define electric grid problem.
    # TODO: Review limits.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference
    active_power_vector_minimum = 0.0 * np.real(electric_grid_model.der_power_vector_reference)
    active_power_vector_maximum = 1.3 * np.real(electric_grid_model.der_power_vector_reference)
    reactive_power_vector_minimum = 0.0 * np.imag(electric_grid_model.der_power_vector_reference)
    reactive_power_vector_maximum = 1.1 * np.imag(electric_grid_model.der_power_vector_reference)

    grid_cost_coefficient = 3.0

    der_model_set.define_optimization_problem(optimization_centralized,
                                              price_data,
                                              state_space_model=True,
                                              kkt_conditions=False,
                                              grid_cost_coefficient=grid_cost_coefficient
                                              )

    linear_electric_grid_model_set.define_optimization_problem(
        optimization_centralized,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
        kkt_conditions=False,
        grid_cost_coefficient=grid_cost_coefficient
    )

    strategic_scenario = True
    if strategic_scenario:
        strategic_der_model_set = StrategicMarket(scenario_name)
        strategic_der_model_set.strategic_optimization_problem(
            optimization_centralized,
            price_data,
            node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
            node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
            branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
            active_power_vector_minimum=active_power_vector_minimum,
            active_power_vector_maximum=active_power_vector_maximum,
            reactive_power_vector_minimum=reactive_power_vector_minimum,
            reactive_power_vector_maximum=reactive_power_vector_maximum,
            big_m=100,
            grid_cost_coefficient=grid_cost_coefficient
        )

    # Define DER problem.

    # Solve centralized optimization problem.
    optimization_centralized.solve()


    # Obtain results.
    flexible_der_type = ['flexible_generator', 'flexible_load']

    results = linear_electric_grid_model_set.get_optimization_results(optimization_centralized)
    a = results.der_active_power_vector_per_unit[flexible_der_type]
    b = results.der_reactive_power_vector_per_unit[flexible_der_type]
    if strategic_scenario:
        c = optimization_centralized.results['der_strategic_offer'].transpose()
        d = optimization_centralized.results['flexible_generator_strategic_offer']


    results_centralized = mesmo.problems.Results()

if __name__ == '__main__':
    main()

