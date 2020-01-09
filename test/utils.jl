# Utility functions for tests.

import Base
import Logging
import Printf

"Time string print modified for printing timing of tests."
function time_string_print(elapsedtime, bytes=0, gctime=0, allocs=0)
    time_string = Printf.@sprintf("%10.6f seconds", elapsedtime)
    if bytes != 0 || allocs != 0
        allocs, ma = (
            Base.prettyprint_getunits(allocs, length(Base._cnt_units),
            Int64(1000))
        )
        if ma == 1
            time_string *= (
                Printf.@sprintf(" (%d%s allocation%s: ", allocs,
                Base._cnt_units[ma], allocs==1 ? "" : "s")
            )
        else
            time_string *= (
                Printf.@sprintf(" (%.2f%s allocations: ", allocs,
                Base._cnt_units[ma])
            )
        end
        time_string *= String(Base.format_bytes(bytes))
        if gctime > 0
            time_string *= (
                Printf.@sprintf(", %.2f%% gc time", 100*gctime/elapsedtime)
            )
        end
        time_string *= (")")
    elseif gctime > 0
        time_string *= (
            Printf.@sprintf(", %.2f%% gc time", 100*gctime/elapsedtime)
        )
    end
    return time_string
end

"Time test execution and print results into `Logging.@info`"
macro time_log(note_string, ex)
    quote
        local val, elapsedtime, bytes, gctime, allocs = @timed($(esc(ex)))
        local time_string = (
            time_string_print(elapsedtime, bytes, gctime,
            Base.gc_alloc_count(allocs))
        )
        Logging.@info(
            $(esc(note_string)) * " | " * lstrip(time_string)
        )
        val
    end
end
