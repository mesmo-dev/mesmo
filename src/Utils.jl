"Utility functions for all FLEDGE modules."
module Utils

"""
Natural less-than comparison operator.

- Enables "natural" less-than comparison and sorting, according to:
  <https://discourse.julialang.org/t/sorting-strings-containing-numbers-so-that-a2-a10/>
- "Natural" means that "a2" is less than "a200", which is not the case
  for default Julia string comparison and sorting.
- Can be used for "natural" sorting like `sort(x, lt=natural_less_than)`.
"""
function natural_less_than(x, y)
    k(x) = [
        occursin(r"\d+", s) ? parse(Int, s) : s
        for s in split(replace(x, r"\d+" => s->" $s "))
    ]
    A = k(x)
    B = k(y)    
    for (a, b) in zip(A, B)
        if !isequal(a, b)
            return typeof(a) <: typeof(b) ? isless(a, b) :
                   isa(a,Int) ? true : false
        end
    end
    return length(A) < length(B)
end

end
