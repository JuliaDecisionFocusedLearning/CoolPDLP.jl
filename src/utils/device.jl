function common_backend(args::Vararg{Any, N}) where {N}
    backends = map(get_backend, args)
    if !all(==(backends[1]), backends)
        throw(ArgumentError("There are several different backends among the arguments: $(unique(backends))"))
    end
    return backends[1]
end
