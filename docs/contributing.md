# Contributing

If you are keen to contribute to this project, please follow these guidelines:

- Before making any change, please first discuss via issue or email with the owners of this repository.
- Development is based on Python 3.8.
- Git branches follow the [GitFlow principle](https://nvie.com/posts/a-successful-git-branching-model/).
- Release versioning follows the [Semantic Versioning principle](https://semver.org/).
- Code style follows the [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/) and [Common Coding Conventions](https://github.com/tum-esi/common-coding-conventions), unless stated otherwise [below](#style-guide).

## Git branches

Based on the [GitFlow principle](https://nvie.com/posts/a-successful-git-branching-model/) there are the following branches:

1. `master` - Contains stable release versions of the repository. Only admins should send pull requests / commits to `master` when 1) fixing a critical bug or 2) publishing a new release.
2. `develop` - This branch is intended as the main branch for development or improvement of features. Anyone can send pull requests to `develop`. Pull requests must be discussed and approved by the administrators of this repository before merging into `develop`.
3. `feature/xxx` - This branch is dedicated to developing feature `xxx`. The idea is to keep development or improvement works separate from the main `develop` branch. Once the work is finished, a pull request is created for feature `xxx` to be merged back into the `develop` branch.

If in doubt, your code should be committed to a `feature/xxx` branch first.

## Release versioning

Every time the `master` branch changes, a new version number is defined according to the [Semantic Versioning principle](https://semver.org/):

1. New releases cause a changing version number in the first digit for major changes and in the second digit for minor changes (e.g. from 0.1.13 -> 0.2.0).
2. Bugfixes cause a changing version number in the third digit (eg. from 0.1.12 -> 0.1.13)

## Style guide

- Follow the [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/) and [Common Coding Conventions](https://github.com/tum-esi/common-coding-conventions). Also check [this PEP8 Explainer](https://realpython.com/python-pep8/).
- Variable / function / object / class / module names:
    - Names are verbose and avoid abbreviations.
    - Variable / function / object names are in lowercase and underscore_case (all letters are lowercase and all words are separated by underscores).
    - Variable / object names start with a lowercase letter.
    - Class / module names start with an uppercase letter and are in CamelCase (all letters are lowercase except for the first letter of new words).
- Paths:
    - Use relative paths.
    - Use `os.join.path("x", "y")` instead of `"x/y"`.
- Docstrings / comments:
    - Docstrings should at minimum contain a short description of the function / class / module.
    - Docstrings and comments should only contain full sentences which conclude with a full stop (dot).
    - Docstrings follow [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- Exceptions / errors / warnings / debug info:
    - Use proper logging tools instead of `print("Error: ...")`.
    - Use [logging](https://docs.python.org/3.6/library/logging.html) like `logger.error("...")` or `logger.warning("...")` or `logger.debug("...")`.
- Line length:
    - Line lengths should not exceed 120 characters.
- Line breaks:
    - Use brackets to contain content spanning multiple lines.
    - Do not use the `\` symbol for line breaks.
- Quotes / strings:
    - Use single quotes `'...'` for parameters, indexes, pathes and use double quotes `"..."` for content, messages and docstrings.
- Results / output files:
    - Store results / output files only in the `results` directory.
    - The results path should be obtained with `mesmo.utils.get_results_path()`
    - The content of the `results` directory should remain local, i.e., it should be ignored by Git and should not appear in any commits to the repository.

## Release checklist

Before pushing a new commit / release to the `master` branch, please go through the following steps:

1. Run tests locally and ensure that all tests complete successfully.
2. Ensure that change log entry has been added for this version in `docs/change_log.md`.
3. Ensure that version numbers and year numbers have been updated everywhere:
    - `setup.py` (at `version=`)
    - `docs/change_log.md`
    - `docs/conf.py` (at `copyright =`)
    - `CITATION.bib`
    - `LICENSE`
4. After pushing a new commit / release, create a tag and publish a new release on Github: <https://github.com/mesmo-dev/mesmo/releases>
5. After publishing a new release, edit the latest Zenodo entry: <https://doi.org/10.5281/zenodo.3562875>
    - Set title to "MESMO - Multi-Energy System Modeling and Optimization".
    - Set correct author names.
    - Set license to "MIT License".
