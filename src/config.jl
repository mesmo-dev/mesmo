# Configuration setup and initialization routines.

import DocStringExtensions
import Logging

# Define configuration parameters.
const _config = Dict([
    "fledge_path" => abspath(@__DIR__, "."),
    "data_path" => abspath(@__DIR__, "..", "data"),
])

# Automatically add signatures before all docstrings via `DocStringExtensions`.
DocStringExtensions.@template DEFAULT = (
    """
    $(DocStringExtensions.TYPEDSIGNATURES)
    $(DocStringExtensions.DOCSTRING)
    """
)

# Output configuration parameters.
if string(@__MODULE__) in ("FLEDGE", "Main.FLEDGE")
    Logging.@info(
        "FLEDGE configuration parameters:\n"
        * join(["$x\n" for x in _config])
    )
end

# Output compile information for debugging.
Logging.@debug(
    "Loading module $(@__MODULE__)."
)
