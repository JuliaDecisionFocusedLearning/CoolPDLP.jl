function __init__()
    register(
        DataDep(
            "miplib2017-collection",
            """
            All instances in the MIPLIB 2017 collection set (size: 3.5 GB).
            Source: https://miplib.zib.de/index.html
            """,
            "https://miplib.zib.de/downloads/collection.zip",
            post_fetch_method = unpack,
        ),
    )
    return register(
        DataDep(
            "pdlp-miplib2017-subset",
            """
            List of instances in the MIPLIB 2017 collection set that were used for the original PDLP benchmarking.
            Source: https://github.com/google-research/FirstOrderLp.jl
            """,
            "https://raw.githubusercontent.com/google-research/FirstOrderLp.jl/69dcc66a88be58031efca740a83c82c0bad13227/benchmarking/mip_relaxations_instance_list",
        ),
    )
end

"""
    list_pdlp_miplib2017_subset()

Return a list of all [MIPLIB 2017](https://miplib.zib.de/) collection instances used in the original [PDLP benchmark](https://arxiv.org/abs/2106.04756).
"""
function list_pdlp_miplib2017_subset()
    list_path = joinpath(datadep"pdlp-miplib2017-subset", "mip_relaxations_instance_list")
    lines = open(list_path, "r") do file
        readlines(file)
    end
    return filter(l -> !startswith(l, "#"), lines)
end

"""
    read_miplib2017_instance(name::String)

Parse a particular [MIPLIB 2017](https://miplib.zib.de/) collection instance and return an [`MILP`](@ref) object along with the path to the source file.

!!! danger
    This will (after manual validation) download the entire MIPLIB 2017 collection, which is 3.5 GB when compressed.
"""
function read_miplib2017_instance(name::String)
    name = lowercase(name)
    mps_gz_path = joinpath(datadep"miplib2017-collection", "$name.mps.gz")
    milp = read_milp(mps_gz_path)
    return milp, mps_gz_path
end

"""
    list_netlib_instances(; exclude_failing::Bool=false)

List all available [Netlib](https://www.netlib.org/lp/) instances.
"""
function list_netlib_instances(; exclude_failing::Bool = false)
    netlib_path = fetch_netlib()
    valid_instances = filter(n -> endswith(n, ".SIF"), readdir(netlib_path))
    instances_nosuffix = map(n -> lowercase(chopsuffix(n, ".SIF")), valid_instances)
    if exclude_failing
        instances_nosuffix = filter(n -> !in(n, ("agg", "blend", "dfl001", "forplan", "gfrd-pnc", "sierra")), instances_nosuffix)
    end
    return instances_nosuffix
end

"""
    read_netlib_instance(name::String)

Parse a particular [Netlib](https://www.netlib.org/lp/) instance and return an [`MILP`](@ref) object.
"""
function read_netlib_instance(name::String)
    name = uppercase(name)
    netlib_path = fetch_netlib()
    sif_path = joinpath(netlib_path, "$name.SIF")
    milp = read_milp(sif_path)
    return milp, sif_path
end
