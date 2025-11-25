function common_backend(args::Vararg{Any, N}) where {N}
    backends = map(get_backend, args)
    @assert all(==(backends[1]), backends)
    return backends[1]
end
