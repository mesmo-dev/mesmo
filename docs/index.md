## FLEDGE - Flexible Distribution Grid Demonstrator

The Flexible Distribution Grid Demonstrator (FLEDGE) is a software tool for the computation of operation problems for electric distribution grids along with distributed energy resources (DERs), such as flexible loads, electric vehicle (EV) chargers, distributed generators (DGs) and energy storage systems (ESS). To this end, it implements 1) electric grid models, 2) energy resource models, 3) power flow solver and 4) optimal power flow solver.

## Work in Progress

1. This repository is under active development and not all features listed below have yet been implemented.
2. Please create an issue if you find this project interesting and have ideas / comments / criticism that may help to make FLEDGE more relevant or useful for your type of problems.

## Features

- Electric grid models
    - Construction of nodal / branch admittance matrices.
    - Consideration of unbalanced / multiphase systems.
    - Generation of linear power flow approximations / sensitivity matrices.
- Energy resource models
    - Time series models for fixed loads.
    - Time series models for EV charging.
    - Time series models for photovoltaics.
    - Linear models for flexible loads.
    - Linear models for energy storage systems.
- Power flow solver
    - Iterative fixed-point power solver.
    - Integrated benchmarking against OpenDSS through [OpenDSSDirect](https://github.com/dss-extensions/OpenDSSDirect.py).
- Optimal power flow solver
    - Setup of centralized social welfare maximization problem.
    - Interfacing convex optimization solvers through [Pyomo](https://github.com/Pyomo/pyomo).

## Contents

``` toctree::
    :maxdepth: 2

    intro
    architecture
    api
    data
    contributing
```
