"""Example script for setting up and solving an nominal operation problem.

- The nominal operation problem (alias: power flow problem, electric grid simulation problem)
  formulates the steady-state power flow problem for all timesteps of the given scenario
  subject to the nominal operation schedule of all DERs.
"""

import mesmo


def main():

    # Settings.
    scenario_name = 'polimi_test_case'

    # High-level API call.
    a=mesmo.api.run_nominal_operation_problem(scenario_name, store_results=False)

    print(3)
if __name__ == '__main__':
    main()
