module CoolPDLPPaPILOExt

using CoolPDLP:
    CoolPDLP, MILP, PrimalDualSolution, PaPILOPresolver,
    milp_to_mps, mps_to_milp, relax
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
    "path to the postsolve archive written by PaPILO"
    postsolve_file::String
    "variable names of the original problem (as they appear in the input MPS file)"
    var_names_orig::Vector{String}
    "constraint names of the original problem (as they appear in the input MPS file)"
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
"""
function CoolPDLP.presolve(presolver::PaPILOPresolver, milp::MILP)
    (; verbose, dual_postsolve) = presolver
    input_file = tempname() * ".mps"
    postsolve_file = tempname() * ".postsolve"
    reduced_file = tempname() * ".mps"
    try
        # PaPILO reduces what it is given, and what CoolPDLP solves is the relaxation; dropping
        # integrality also happens to be what lets PaPILO record dual postsolve information,
        # which it never does for a problem with integer variables
        milp_to_mps(relax(milp), input_file)
        run_papilo(verbose) do
            return PaPILO.presolve_write_from_file(
                input_file, postsolve_file, reduced_file; dual_postsolve
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
            postsolve_file,
            milp.var_names,
            milp.con_names,
            milp_reduced.var_names,
            milp_reduced.con_names,
            c_reduced,
            At_reduced,
            PrimalDualSolution(milp),
        )
        return milp_reduced, presolve_info
    finally
        isfile(input_file) && rm(input_file; force = true)
        isfile(reduced_file) && rm(reduced_file; force = true)
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
    names = (:primal_reduced, :dual_reduced, :costs_reduced, :primal_orig, :dual_orig, :costs_orig)
    files = NamedTuple{names}(map(_ -> tempname() * ".sol", names))
    try
        x_reduced = Array(sol_reduced.x)
        PaPILO.write_sol(
            files.primal_reduced, name_values(presolve_info.var_names_reduced, x_reduced)
        )
        dual_files = if dual_postsolve
            y_reduced = Array(sol_reduced.y)
            z_reduced = Array(presolve_info.c_reduced - presolve_info.At_reduced * y_reduced)
            PaPILO.write_sol(
                files.dual_reduced, name_values(presolve_info.con_names_reduced, y_reduced)
            )
            PaPILO.write_sol(
                files.costs_reduced, name_values(presolve_info.var_names_reduced, z_reduced)
            )
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
                presolve_info.postsolve_file, files.primal_reduced, files.primal_orig; dual_files...
            )
        end
        # PaPILO keys its solution files by name and omits the zeros, so the values are gathered
        # back into the column and row order of the original MILP, which `presolve_info` recorded
        proto = presolve_info.sol_orig_proto
        x_orig = gather_values(PaPILO.read_sol(files.primal_orig), presolve_info.var_names_orig)
        y_orig = if dual_postsolve
            gather_values(PaPILO.read_sol(files.dual_orig), presolve_info.con_names_orig)
        else
            fill(eltype(x_orig)(NaN), length(presolve_info.con_names_orig))
        end
        # the shapes come from the prototype and the containers from `sol_reduced`,
        # which the algorithm already produced in its own types
        x = copyto!(similar(sol_reduced.x, size(proto.x)), x_orig)
        y = copyto!(similar(sol_reduced.y, size(proto.y)), y_orig)
        return PrimalDualSolution(x, y)
    finally
        isfile(presolve_info.postsolve_file) && rm(presolve_info.postsolve_file; force = true)
        for file in files
            isfile(file) && rm(file; force = true)
        end
    end
end

"""
    run_papilo(f, verbose)

Run `f`, which shells out to the PaPILO binary, muting the binary's own chatter unless `verbose`.
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

Look each of `names` up in the `values` mapping read by `PaPILO.read_sol`, returning a vector
indexed like `names`. A name absent from the file stands for a zero, which PaPILO omits.
"""
function gather_values(values::Dict{String, T}, names::Vector{String}) where {T <: Number}
    return [get(values, name, zero(T)) for name in names]
end

end
