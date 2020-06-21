"""Run script for energy market clearing example."""

import numpy as np
import pandas as pd

import fledge.config
import fledge.data_interface
import fledge.electric_grid_models
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    # results_path = fledge.utils.get_results_path('run_market_clearing', scenario_name)

    # Obtain electric grid model.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

    # Obtain DERs.
    ders = electric_grid_model.ders
    der_power_vector = electric_grid_model.der_power_vector_reference
    der_active_power_vector = np.real(der_power_vector)

    # Define abritrary DER bids.
    der_bids = dict.fromkeys(ders)
    for der_index, der in enumerate(ders):
        der_bids[der] = (
            pd.Series(
                [der_active_power_vector[der_index] / 2, der_active_power_vector[der_index] / 2],
                index=[0.0, 1.0]
            )
        )

    # Define arbitrary clearing price.
    cleared_price = 0.5

    # Obtain dispatch power.
    der_active_power_vector_dispatch = np.zeros(der_active_power_vector.shape, dtype=np.float)
    for der_index, der in enumerate(ders):
        if der_active_power_vector[der_index] < 0.0:
            der_active_power_vector_dispatch[der_index] += (
                der_bids[der].loc[der_bids[der].index > cleared_price].sum()
            )
        elif der_active_power_vector[der_index] > 0.0:
            der_active_power_vector_dispatch[der_index] += (
                der_bids[der].loc[der_bids[der].index < cleared_price].sum()
            )

    # Print results.
    print(f"der_bids = {der_bids}")
    print(f"der_active_power_vector_dispatch = {der_active_power_vector_dispatch}")


if __name__ == "__main__":
    main()
