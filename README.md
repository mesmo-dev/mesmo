# FLEDGE - Flexible Distribution Grid Demonstrator

The Flexible Distribution Grid Demonstrator (FLEDGE) is a software tool for the computation of operation problems for electric distribution grids along with distributed energy resources (DERs), such as flexible loads, electric vehicle (EV) chargers, distributed generators (DGs) and energy storage systems (ESS). To this end, it implements 1) electric grid models, 2) energy resource models, 3) power flow solver and 4) optimal power flow solver.

## Work in Progress

1. This repository is under active development and not all features listed below have yet been implemented.
2. Please create an [issue](https://github.com/TUMCREATE-ESTL/FLEDGE.jl/issues) if you find this project interesting and have ideas / comments / criticism that may help to make it more relevant or useful for your type of problems.

## Features

- Electric grid models
    - [x] Construction of nodal / branch admittance matrices.
    - [x] Consideration of unbalanced / multiphase systems.
    - [x] Generation of linear power flow approximations / sensitivity matrices.
- Energy resource models
    - [x] Time series models for fixed loads.
    - [x] Time series models for EV charging.
    - [ ] Time series models for photovoltaics.
    - [ ] Linear models for flexible loads.
    - [ ] Linear models for energy storage systems.
- Power flow solver
    - [x] Iterative fixed-point power solver.
    - [x] Integrated benchmarking against OpenDSS through [OpenDSSDirect](https://github.com/dss-extensions/OpenDSSDirect.jl).
- Optimal power flow solver
    - [ ] Setup of centralized social welfare maximization problem.
    - [ ] Interfacing convex optimization solvers through [JuMP](https://github.com/JuliaOpt/JuMP.jl).

## Documentation

The preliminary documentation is located at [tumcreate-estl.github.io/FLEDGE.jl](https://tumcreate-estl.github.io/FLEDGE.jl/dev/).

## Installation

1. Check requirements:
    - [Julia 1.3](https://julialang.org/downloads/)
2. Clone or download repository.
3. In Julia, run `]activate path_to_fledge_repository; instantiate; precompile`.

Please also see [docs/src/intro.md](./docs/src/intro.md).

## Contributing

If you are keen to contribute to this project, please see [docs/src/contributing.md](./docs/src/contributing.md).
