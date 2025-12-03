"""
    common_backend(args...)

Return the common GPU backend of several arguments, if it exists, and throw an error otherwise.
"""
function common_backend(args::Vararg{Any, N}) where {N}
    backends = map(get_backend, args)
    if !all(==(backends[1]), backends)
        throw(ArgumentError("There are several different backends among the arguments: $(unique(backends))"))
    end
    return backends[1]
end
