module CoolPDLPPaPILOExt

using CoolPDLP:
    CoolPDLP, MILP, PrimalDualSolution, PaPILOPresolver,
    milp_to_mps, mps_to_milp, nbcons, nbvar, relax
using DocStringExtensions: TYPEDFIELDS
using PaPILO: PaPILO

"""
    PaPILOPresolveInfo

The `presolve_info` object produced by `presolve(::PaPILOPresolver, milp)` and consumed by
`postsolve(::PaPILOPresolver, presolve_info, sol_reduced)`. It lives in the `CoolPDLPPaPILOExt`
extension rather than in `CoolPDLP` itself, since nothing outside this backend needs it.

# Fields

$(TYPEDFIELDS)
"""
struct PaPILOPresolveInfo{M <: AbstractMatrix, V <: AbstractVector, S <: PrimalDualSolution}
    "contents of the postsolve archive written by PaPILO"
    archive::Vector{UInt8}
    "variable names of the original problem, as written to the input MPS file"
    var_names_orig::Vector{String}
    "constraint names of the original problem, as written to the input MPS file"
    con_names_orig::Vector{String}
    "variable names of the presolved problem (as they appear in the reduced MPS file)"
    var_names_reduced::Vector{String}
    "constraint names of the presolved problem (as they appear in the reduced MPS file)"
    con_names_reduced::Vector{String}
    """
    objective vector of the presolved problem, to derive its reduced costs `c - Aᵀy` which
    PaPILO wants alongside its dual solution. Left empty (rather than `nothing`) when
    `dual_postsolve` is off, so that the type of this object never depends on that flag
    """
    c_reduced::V
    "transposed constraint matrix of the presolved problem, for the same reason"
    At_reduced::M
    "zero solution of the original problem, giving `postsolve` its shape"
    sol_orig_proto::S
end

"""
    presolve(presolver::PaPILOPresolver, milp) -> (milp_reduced, presolve_info)

Write `milp` to a temporary MPS file, run PaPILO's presolve command, and read the (typically
smaller) reduced problem back as a CPU-`Float64` `MILP` (`solve` converts it afterwards anyway).

The problem handed to PaPILO is the continuous relaxation of `milp`: `solve` only ever tackles
that relaxation, so an integer-specific reduction would presolve a problem nobody is solving.

With `presolver.dual_postsolve` (the default), PaPILO is restricted to the reductions whose dual
information it knows how to record, so that [`postsolve`](@ref) can recover the dual solution as
well as the primal one. The reduced problem is then usually larger.

Throw an error if PaPILO fails, which it does when it proves `milp` infeasible or unbounded.

!!! warning
    The objective of `milp_reduced` lacks the constant contributed by the variables that PaPILO
    eliminated (a [`MILP`](@ref) has no objective constant), so objective values computed on
    it differ from those of `milp` by that constant.
"""
function CoolPDLP.presolve(presolver::PaPILOPresolver, milp::MILP)
    (; verbose, dual_postsolve) = presolver
    # the names of `milp` may not survive an MPS file (spaces, duplicates), and PaPILO matches
    # its solution files up by name, so the problem is written under names of our own
    var_names = string.("x", 1:nbvar(milp))
    con_names = string.("c", 1:nbcons(milp))
    return mktempdir() do dir
        input_file = joinpath(dir, "input.mps")
        archive_file = joinpath(dir, "archive.postsolve")
        reduced_file = joinpath(dir, "reduced.mps")
        # PaPILO reduces what it is given, and what CoolPDLP solves is the relaxation; dropping
        # integrality also happens to be what lets PaPILO record dual postsolve information,
        # which it never does for a problem with integer variables
        milp_to_mps(relax(milp), input_file; var_names, con_names)
        try
            run_papilo(verbose) do
                return PaPILO.presolve_write_from_file(
                    input_file, archive_file, reduced_file; dual_postsolve
                )
            end
        catch e
            e isa ProcessFailedException || rethrow()
            error(
                "PaPILO failed to presolve the problem. This happens when it proves the " *
                    "problem infeasible or unbounded: use `PaPILOPresolver(; verbose = true)` " *
                    "to see its diagnostic."
            )
        end
        milp_reduced = mps_to_milp(
            reduced_file; dataset = milp.dataset,
            name = string(milp.name, " [presolved with PaPILO]"), path = milp.path,
        )
        # the reduced problem's objective and matrix are only kept to derive the reduced costs
        # `c - Aᵀy` that PaPILO asks for next to the dual solution
        c_reduced = dual_postsolve ? milp_reduced.c : similar(milp_reduced.c, 0)
        At_reduced = dual_postsolve ? milp_reduced.At : similar(milp_reduced.At, 0, 0)
        presolve_info = PaPILOPresolveInfo(
            read(archive_file),
            var_names,
            con_names,
            milp_reduced.var_names,
            milp_reduced.con_names,
            c_reduced,
            At_reduced,
            PrimalDualSolution(milp),
        )
        return milp_reduced, presolve_info
    end
end

"""
    postsolve(presolver::PaPILOPresolver, presolve_info, sol_reduced) -> PrimalDualSolution

Write `sol_reduced` to plain-text solution files, run PaPILO's postsolve command, and read the
original-space solution back, converting it to the proper format (the shape `presolve_info`
memorized, and the element and array types of `sol_reduced`).

The dual travels back with the primal whenever the archive was written with
`presolver.dual_postsolve`, which is what `PaPILOPresolver` does by default. It comes back
filled with `NaN` otherwise, since PaPILO then has nothing to reconstruct it from.
"""
function CoolPDLP.postsolve(
        presolver::PaPILOPresolver,
        presolve_info::PaPILOPresolveInfo,
        sol_reduced::PrimalDualSolution,
    )
    (; verbose, dual_postsolve) = presolver
    (; var_names_orig, con_names_orig, var_names_reduced, con_names_reduced) = presolve_info
    return mktempdir() do dir
        names = (:primal_reduced, :dual_reduced, :costs_reduced, :primal_orig, :dual_orig, :costs_orig)
        files = NamedTuple{names}(map(name -> joinpath(dir, "$name.sol"), names))
        archive_file = joinpath(dir, "archive.postsolve")
        write(archive_file, presolve_info.archive)

        x_reduced = Array(sol_reduced.x)
        PaPILO.write_sol(files.primal_reduced, name_values(var_names_reduced, x_reduced))
        dual_files = if dual_postsolve
            y_reduced = Array(sol_reduced.y)
            z_reduced = Array(presolve_info.c_reduced - presolve_info.At_reduced * y_reduced)
            PaPILO.write_sol(files.dual_reduced, name_values(con_names_reduced, y_reduced))
            PaPILO.write_sol(files.costs_reduced, name_values(var_names_reduced, z_reduced))
            (;
                dual_reduced_solution = files.dual_reduced,
                costs_reduced_solution = files.costs_reduced,
                dualsolution = files.dual_orig,
                reducedcosts = files.costs_orig,
            )
        else
            (;)
        end
        run_papilo(verbose) do
            return PaPILO.postsolve_from_file(
                archive_file, files.primal_reduced, files.primal_orig; dual_files...
            )
        end
        # PaPILO keys its solution files by name and omits the zeros, so the values are gathered
        # back into the column and row order of the original MILP
        proto = presolve_info.sol_orig_proto
        x_orig = gather_values(PaPILO.read_sol(files.primal_orig), var_names_orig)
        y_orig = if dual_postsolve
            gather_values(PaPILO.read_sol(files.dual_orig), con_names_orig)
        else
            fill(eltype(x_orig)(NaN), length(con_names_orig))
        end
        # the shapes come from the prototype and the containers from `sol_reduced`,
        # which the algorithm already produced in its own types
        x = copyto!(similar(sol_reduced.x, size(proto.x)), x_orig)
        y = copyto!(similar(sol_reduced.y, size(proto.y)), y_orig)
        return PrimalDualSolution(x, y)
    end
end

"""
    run_papilo(f, verbose)

Run `f`, which shells out to the PaPILO binary, muting the binary's own chatter unless `verbose`.

Muting redirects the `stdout` of the whole process, which is not task-safe: PaPILO.jl offers no
way to redirect the output of the binary alone.
"""
run_papilo(f, verbose::Bool) = verbose ? f() : redirect_stdout(f, devnull)

"""
    name_values(names, values)

Pair each of `names` with the entry of `values` at the same index, in the mapping form
`PaPILO.write_sol` expects.
"""
name_values(names::Vector{String}, values::AbstractVector) = Dict(zip(names, values))

"""
    gather_values(values, names)

Arrange the `values` mapping read by `PaPILO.read_sol` into a vector indexed like `names`.
One of `names` absent from the mapping stands for a zero, which PaPILO omits. A name of the
mapping absent from `names` is an error: it means the file does not describe the problem that
`names` belong to.
"""
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
