"""
    common_backend(args...)

Return the common GPU backend of several arguments, if it exists, and throw an error otherwise.
"""
function common_backend(args::Vararg{Any, N}) where {N}
    backends = map(_get_backend, args)
    if !all(x -> isnothing(x) || (x == backends[1]), backends)
        throw(ArgumentError("There are several different backends among the arguments: $(unique(backends))"))
    end
    return backends[1]
end

_get_backend(::Nothing) = nothing
_get_backend(x) = get_backend(x)
