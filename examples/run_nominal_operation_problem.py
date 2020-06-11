"""Example script for setting up and solving an optimal operation problem.

- The nominal operation problem (alias: power flow problem, electric grid simulation problem)
  formulates the steady-state power flow problem for all timesteps of the given scenario
  subject to the nominal operation schedule of all DERs.
"""

import fledge.api
import fledge.problems


def main():

    # Settings.
    scenario_name = 'singapore_district1'

    # High-level API.
    # - Running the operation problem through the high-level API directly stores results in the results directory.
    # - This is intended for single runs of scenarios and does not allow successive changes to the problem.
    fledge.api.run_nominal_operation_problem(scenario_name)

    # Low-level API.
    # - Obtain a problem object, which can be used to incrementally apply changes, solve and retrieve results.
    # - This intended for scripting and to enable custom work flows.
    problem = fledge.problems.NominalOperationProblem(scenario_name)
    problem.solve()
    results = problem.get_results()
    print(results)


if __name__ == '__main__':
    main()
