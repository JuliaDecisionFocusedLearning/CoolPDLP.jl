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
    pdlp_miplib2017_subset()

Return a list of all MIPLIB 2017 instances used in the original PDLP benchmark.
"""
function pdlp_miplib2017_subset()
    list_path = joinpath(datadep"pdlp-miplib2017-subset", "mip_relaxations_instance_list")
    lines = open(list_path, "r") do file
        readlines(file)
    end
    return filter(l -> !startswith(l, "#"), lines)
end

"""
    miplib2017_instance(name::String)

Parse a particular MIPLIB 2017 instance and return a [`MILP`](@ref) object.
"""
function miplib2017_instance(name::String)
    mps_gz_path = joinpath(datadep"miplib2017-collection", "$name.mps.gz")
    return read_milp(mps_gz_path)
end
