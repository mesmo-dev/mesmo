# Configuration

MESMO configuration parameters can be set in `config.yml`. As an initial user, you likely do not need to modify the configuration.

## Configuration workflow

If you want to change the configuration, you can create or modify `config.yml` in the MESMO repository main directory. MESMO will automatically create `config.yml` during runtime, if it does not exist. Initially, `config.yml` will be empty. You can copy configuration parameters from `mesmo/config_default.yml` to `config.yml` and modify their value to define your local configuration. To define nested configuration parameters, you need to replicate the nested structure in `config.yml`.

The configuration parameters which are defined in `config.yml` will take precedence over those defined in `mesmo/config_default.yml`. If you would like to revert a parameter to its default value, just delete the parameter from `config.yml`. Please do not modify `mesmo/config_default.yml` directly.

(optimization-solver)=
## Optimization solver

To define CPLEX as the optimization solver, use:

```
optimization:
  solver_name: cplex
```
