"""Example script for setting up and solving a flexible biogas plant optimal operation problem.
"""

import matplotlib.pyplot as plt

import fledge.data_interface
import fledge.der_models
import fledge.problems


# Settings.
scenario_name = 'cigre_mv_network_with_all_ders'
plots = True  # If True, script may produce plots.

# Recreate / overwrite database, to incorporate changes in the CSV files.
fledge.data_interface.recreate_database()

# Obtain price timeseries.
price_data = fledge.data_interface.PriceData(scenario_name)
price_type = 'EPEX SPOT Power DE Day Ahead'
price_timeseries = price_data.price_timeseries_dict[price_type]

# obtain der models
der_model_set = fledge.der_models.DERModelSet(scenario_name)

# Instantiate optimization problem.
problem = fledge.problems.OptimalOperationProblem(scenario_name)
problem.solve()
in_per_unit = True
results = problem.get_results(in_per_unit=in_per_unit)
optimization_problem = problem.optimization_problem

results.update(
    der_model_set.get_optimization_results(
    problem.optimization_problem
    )
)

# Print results (DLMPs, etc.)
print(results)

flexible_biogas_plant_model = der_model_set.flexible_der_models['Biogas 9']
results = (
    flexible_biogas_plant_model.get_optimization_results(
        optimization_problem, price_timeseries
    )
)
# Plot results.
if plots:

    for output in flexible_biogas_plant_model.outputs:
        plt.plot(flexible_biogas_plant_model.output_maximum_timeseries[output], label="Maximum", drawstyle='steps-post')
        plt.plot(flexible_biogas_plant_model.output_minimum_timeseries[output], label="Minimum", drawstyle='steps-post')
        plt.plot(results['output_vector'][output], label="Optimal", drawstyle='steps-post')
        plt.legend()
        plt.xlabel('Timesteps (day and hour)')
        plt.title(f"Output: {output}")
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
                 'rotation', 45)
        plt.show()
        plt.close()

    for control in flexible_biogas_plant_model.controls:
        plt.plot(results['control_vector'][control], label="Optimal", drawstyle='steps-post', color='#D55E00')
        plt.legend()
        plt.xlabel('Timesteps (day and hour)')
        plt.title(f"Control: {control}")
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
                 'rotation', 45)
        plt.show()
        plt.close()

    for state in flexible_biogas_plant_model.states:
        plt.plot(results['state_vector'][state], label="Optimal", drawstyle='steps-post', color='#D55E00')
        plt.legend()
        plt.xlabel('Timesteps (day and hour)')
        plt.title(f"State: {state}")
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
                 'rotation', 45)
        plt.show()
        plt.close()

    plt.plot(results['profit_vector']['profit_value'], label="Optimal", drawstyle='steps-post', color='#D55E00')
    plt.legend()
    plt.title('Profit per time interval (euros)')
    plt.setp(plt.gca().xaxis.get_majorticklabels(),
             'rotation', 45)
    plt.xlabel('Timesteps (day and hour)')
    plt.ylabel('Profit (euros)')
    plt.show()
    plt.close()

    plt.plot(price_timeseries['price_value'], drawstyle='steps-post')
    plt.title(f"Price in euros per MWh: {price_type}")
    plt.setp(plt.gca().xaxis.get_majorticklabels(),
             'rotation', 45)
    plt.show()
    plt.close()

    print("Total profit in euros:", sum(results['profit_vector']['profit_value']))
