"""Example script for setting up and solving an optimal operation problem.

- The optimal operation problem (alias: optimal dispatch problem, optimal power flow problem)
  formulates the optimization problem for minimizing the objective functions of DERs and grid operators
  subject to the model constraints of all DERs and grids.
"""

import fledge.api
import fledge.problems


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'

    # High-level API.
    # - Running the operation problem through the high-level API directly stores results in the results dictionary.
    # - This is intended for single runs of scenarios and does not allow successive changes to the problem.
    fledge.api.run_optimal_operation_problem(scenario_name)

    # Low-level API.
    # - Obtain a problem object, which can be used to incrementally apply changes, solve and retrieve results.
    # - This intended for scripting and to enable custom work flows.
    problem = fledge.problems.OptimalOperationProblem(scenario_name)
    problem.solve()
    results = problem.get_results()
    print(results)


if __name__ == '__main__':
    main()
