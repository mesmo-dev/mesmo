# Contributing

Please create an [issue](https://github.com/TUMCREATE-ESTL/FLEDGE.jl/issues) if you find this project interesting and have ideas / comments / criticism that may help to make it more relevant or useful for your type of problems.

If you are keen to contribute to this project, please follow these guidelines:

- Before making any change, please first discuss via issue or email with the owners of this repository.
- Development is based on Julia 1.1 and Python 3.6.
- Git branches follow the [GitFlow principle](https://nvie.com/posts/a-successful-git-branching-model/).
- Release versioning follows the [Semantic Versioning principle](https://semver.org/).

## Git Branches

Based on the [GitFlow principle](https://nvie.com/posts/a-successful-git-branching-model/) there are the following branches:

1. `master` - Contains stable release versions of the repository. Only admins should send pull requests / commits to `master` when 1) fixing a bug or 2) publishing a new release.
2. `development` - This branch is intended as the main branch for development or improvement of features. Anyone can send pull requests to `develop`.
3. `feature/xxx` - This branch is dedicated to developing feature `xxx`. The idea is to keep development or improvement works separate from the main `develop` branch. Once the work is finished, a pull request is created for feature `xxx` to be merged back into the `develop` branch.

## Release Versioning

Every time the `master` branch changes, a new version number is defined according to the [Semantic Versioning principle](https://semver.org/):

1. New releases cause a changing version number in the first digit for major changes and in the second digit for minor changes (e.g. from 0.1.13 -> 0.2.0).
2. Bugfixes cause a changing version number in the third digit (eg. from 0.1.12 -> 0.1.13)

## Style Guide

- For Julia code, follow the [Julia style guide](https://docs.julialang.org/en/v1.1/manual/style-guide/).
- For Python code, follow the [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/) and check [this PEP8 Explainer](https://realpython.com/python-pep8/).
- Variable / function / object / class / module names:
    - Names are verbose and avoid abbreviations.
    - Variable / function / object names are in lowercase and underscore_case (all letters are lowercase and all words are separated by underscores).
    - Variable / object names start with a lowercase letter.
    - Class / module names start with an uppercase letter and are in CamelCase (all letters are lowercase except for the first letter of new words).
- Paths:
    - Use relative paths.
    - Use `os.join.path("x", "y")` (Python) or `joinpath("x", "y")` (Julia) instead of `"x/y"`.
- Docstrings / comments:
    - Docstrings should at minimum contain a short description of the function / class / module.
    - Docstrings and comments should only contain full sentences which conclude with a full stop (dot).
    - In Julia, docstrings follow [Julia documentation style](https://docs.julialang.org/en/v1.1/manual/documentation/).
    - In Python, docstrings follow [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- Exceptions / errors / warnings / debug info:
    - Use proper logging tools instead of `print("Error: ...")`.
    - In Julia, use [Logging](https://docs.julialang.org/en/v1/stdlib/Logging/) like `Logging.@error("...")` or `Logging.@warn("...")` or `Logging.@info("...")` or `Logging.@debug("...")`.
    - In Python, use [logging](https://docs.python.org/3.6/library/logging.html) like `logger.error("...")` or `logger.warning("...")` or `logger.debug("...")`.
- Line length:
    - In Julia, line lengths should not exceed 80 characters.
    - In Python, line lengths should not exceed 120 characters.
- Line breaks:
    - Use brackets to contain content spanning multiple lines.
    - In Python, do not use the `\` symbol for line breaks.
- Quotes / strings:
    - In Julia, always use double quotes `"..."`.
    - In Python, use single quotes `'...'` for parameters, indexes, pathes and use double quotes `"..."` for content, messages and docstrings.
