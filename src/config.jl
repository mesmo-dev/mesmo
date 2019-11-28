# Configuration setup and initialization routines.

import Memento
import DocStringExtensions

# Define configuration parameters.
const _config = Dict([
    "fledge_path" => abspath(@__DIR__, "."),
    "data_path" => abspath(@__DIR__, "..", "data"),
    "logger_level" => "debug" # Levels: debug, info, notice, warn, error.
])

# Create module logger.
const _logger = Memento.getlogger(@__MODULE__)
# Set root logger level to enable receiving all messages from module logger.
Memento.config!(_config["logger_level"])
# Set module logger level.
Memento.setlevel!(_logger, _config["logger_level"])
Memento.debug(
    _logger,
    "Created logger with level \"$(_config["logger_level"])\"."
)
# TODO: Revise logger configuration to follow recommendations from:
# <https://github.com/invenia/Memento.jl/issues/69>

# Set logger to be available for configuration at runtime,
# which is important for modifying the logging level during runtime.
function __init__()
    Memento.register(_logger)
end

# Output config parameters.
if string(@__MODULE__) in ("FLEDGE", "Main.FLEDGE")
    Memento.info(
        _logger,
        (
            "Configuration parameters:\n"
            * join(["$x\n" for x in _config])
        )
    )
end

# Automatically add signatures before all docstrings via `DocStringExtensions`.
DocStringExtensions.@template DEFAULT =
    """
    $(DocStringExtensions.TYPEDSIGNATURES)
    $(DocStringExtensions.DOCSTRING)
    """
