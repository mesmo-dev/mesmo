# Export definition.
# - Export everything in the API module except internal symbols, which are
#   defined as those which start with an underscore, e.g., `_config`.
# - Hence, do not add symbols to the `_excluded_symbols` list. Instead,
#   rename them with an underscore.
const _excluded_symbols = [Symbol(@__MODULE__), :eval, :include]
for symbol in names(API, all=true)
    symbol_string = string(symbol)
    if symbol in _excluded_symbols || startswith(symbol_string, "_")
        continue
    end
    if !(Base.isidentifier(symbol) || (startswith(symbol_string, "@") &&
         Base.isidentifier(symbol_string[2:end])))
       continue
    end
    @eval export $symbol
end
