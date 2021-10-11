# Documentation

The documentation is automatically deployed to Github Pages whenever commits are pushed to Github. To generate the documentation locally, follow these steps:

1. In your Python environment, run `pip install -r docs/requirements.txt`.
2. From the repository base directory, run `sphinx-multiversion docs docs/_build/html`. (This builds the docs for all versions / branches.)
3. Alternatively, run `sphinx-build docs docs/_build/html`. (This builds the docs only for the current version / branch.)
