module CoolPDLPPaPILOExt

using CoolPDLP:
    CoolPDLP, MILP, PrimalDualSolution, PaPILOPresolver,
    milp_to_mps, mps_to_milp, nbcons, nbvar, relax
using PaPILO: PaPILO

struct PaPILOPresolveInfo{M <: AbstractMatrix, V <: AbstractVector, S <: PrimalDualSolution}
    archive::Vector{UInt8}
    var_names_orig::Vector{String}
    con_names_orig::Vector{String}
    var_names_reduced::Vector{String}
    con_names_reduced::Vector{String}
    c_reduced::V
    At_reduced::M
    sol_orig_proto::S
end

function CoolPDLP.presolve(presolver::PaPILOPresolver, milp::MILP)
    # PaPILO matches solutions by name, so we use unique names that survive an MPS file
    var_names = string.("x", 1:nbvar(milp))
    con_names = string.("c", 1:nbcons(milp))
    return mktempdir() do dir
        input_file = joinpath(dir, "input.mps")
        archive_file = joinpath(dir, "archive.postsolve")
        reduced_file = joinpath(dir, "reduced.mps")
        # PaPILO records no dual information for integer problems
        milp_to_mps(relax(milp), input_file; var_names, con_names)
        try
            run_papilo(presolver.verbose) do
                PaPILO.presolve_write_from_file(
                    input_file, archive_file, reduced_file; dual_postsolve = true
                )
            end
        catch e
            e isa ProcessFailedException || rethrow()
            error(
                "PaPILO failed to presolve the problem, which happens when it proves the " *
                    "problem infeasible or unbounded. Use `PaPILOPresolver(; verbose = true)` " *
                    "to see its diagnostic."
            )
        end
        m = mps_to_milp(
            reduced_file; milp.dataset, name = string(milp.name, " [presolved]"), milp.path
        )
        # PaPILO writes the objective constant with the opposite sign to what QPSReader expects
        milp_reduced = MILP(;
            m.c, c0 = -m.c0, m.lv, m.uv, m.A, m.At, m.lc, m.uc,
            m.var_names, m.con_names, m.dataset, m.name, m.path,
        )
        presolve_info = PaPILOPresolveInfo(
            read(archive_file), var_names, con_names, milp_reduced.var_names, milp_reduced.con_names,
            milp_reduced.c, milp_reduced.At, PrimalDualSolution(milp),
        )
        return milp_reduced, presolve_info
    end
end

function CoolPDLP.postsolve(
        presolver::PaPILOPresolver, info::PaPILOPresolveInfo, sol_reduced::PrimalDualSolution
    )
    return mktempdir() do dir
        file(name) = joinpath(dir, "$name.sol")
        archive_file = joinpath(dir, "archive.postsolve")
        write(archive_file, info.archive)
        x_reduced = Array(sol_reduced.x)
        y_reduced = Array(sol_reduced.y)
        z_reduced = info.c_reduced - info.At_reduced * y_reduced  # reduced costs
        PaPILO.write_sol(file(:x_reduced), Dict(zip(info.var_names_reduced, x_reduced)))
        PaPILO.write_sol(file(:y_reduced), Dict(zip(info.con_names_reduced, y_reduced)))
        PaPILO.write_sol(file(:z_reduced), Dict(zip(info.var_names_reduced, z_reduced)))
        run_papilo(presolver.verbose) do
            PaPILO.postsolve_from_file(
                archive_file, file(:x_reduced), file(:x);
                dual_reduced_solution = file(:y_reduced),
                costs_reduced_solution = file(:z_reduced),
                dualsolution = file(:y), reducedcosts = file(:z),
            )
        end
        x = gather_values(PaPILO.read_sol(file(:x)), info.var_names_orig)
        y = gather_values(PaPILO.read_sol(file(:y)), info.con_names_orig)
        (; sol_orig_proto) = info
        return PrimalDualSolution(
            copyto!(similar(sol_reduced.x, size(sol_orig_proto.x)), x),
            copyto!(similar(sol_reduced.y, size(sol_orig_proto.y)), y),
        )
    end
end

# muting redirects the stdout of the whole process, which is not task-safe
run_papilo(f, verbose::Bool) = verbose ? f() : redirect_stdout(f, devnull)

# PaPILO omits zeros from its solution files
function gather_values(values::Dict{String, T}, names::Vector{String}) where {T <: Number}
    index = Dict(zip(names, eachindex(names)))
    gathered = zeros(T, length(names))
    for (name, value) in values
        haskey(index, name) || error("PaPILO returned a value for the unknown name `$name`")
        gathered[index[name]] = value
    end
    return gathered
end

end
