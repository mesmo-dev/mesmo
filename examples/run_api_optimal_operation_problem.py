"""Example script for setting up and solving an optimal operation problem.

- The optimal operation problem (alias: optimal dispatch problem, optimal power flow problem)
  formulates the optimization problem for minimizing the objective functions of DERs and grid operators
  subject to the model constraints of all DERs and grids.
"""

import mesmo


def main():

    # Settings.
    scenario_name = 'polimi_test_case'

    # High-level API call.
    a=mesmo.api.run_optimal_operation_problem(scenario_name, store_results=False)

    print(1)

if __name__ == '__main__':
    main()
