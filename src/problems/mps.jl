"""
    milp_to_mps(milp::MILP, file::AbstractString; var_names = milp.var_names, con_names = milp.con_names)

Write `milp` to an MPS file at `file`.

Variables and constraints are written under `var_names` and `con_names`, in the order of the
columns and rows of `milp`. The MPS writer replaces spaces and disambiguates duplicates, so
a caller which needs to match names back afterwards (against a solution file, say) should pass
names it controls rather than rely on the MILP's own.

A row without any finite bound constrains nothing: it is written as a free row, which
[`mps_to_milp`](@ref) and other MPS readers skip.

The file always holds a minimization problem, like `milp` itself. If `milp` was read from a
maximization problem, its objective was negated on the way in (see [`MILP`](@ref)), so the file
holds that negated objective and not the original one.
"""
function milp_to_mps(
        milp::MILP, file::AbstractString;
        var_names::Vector{String} = milp.var_names, con_names::Vector{String} = milp.con_names,
    )
    isbatched(milp) && throw(ArgumentError("Cannot write a batched MILP to an MPS file"))
    # MPS is a plain-text, `Float64` format, so the problem is brought back to the CPU whatever
    # backend and matrix type it lived on
    milp_cpu = adapt(CPU(), milp)
    (; c, c0, lv, uv, lc, uc, int_var) = milp_cpu
    At = SparseMatrixCSC(milp_cpu.At)  # column `i` of `At` is row `i` of `A`

    model = MOI.FileFormats.MPS.Model()
    x = MOI.add_variables(model, nbvar(milp_cpu))
    for j in eachindex(x)
        MOI.set(model, MOI.VariableName(), x[j], var_names[j])
        isfinite(lv[j]) && MOI.add_constraint(model, x[j], MOI.GreaterThan(Float64(lv[j])))
        isfinite(uv[j]) && MOI.add_constraint(model, x[j], MOI.LessThan(Float64(uv[j])))
        int_var[j] && MOI.add_constraint(model, x[j], MOI.Integer())
    end

    objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(Float64.(c), x), Float64(c0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(objective)}(), objective)

    for i in axes(At, 2)
        terms = [
            MOI.ScalarAffineTerm(Float64(nonzeros(At)[k]), x[rowvals(At)[k]])
                for k in nzrange(At, i)
        ]
        row = MOI.add_constraint(
            model, MOI.ScalarAffineFunction(terms, 0.0),
            MOI.Interval(Float64(lc[i]), Float64(uc[i])),
        )
        MOI.set(model, MOI.ConstraintName(), row, con_names[i])
    end

    MOI.write_to_file(model, file)
    return file
end

"""
    mps_to_milp(file::AbstractString; dataset = "", name = "", path = "")

Read the MPS file at `file` into a [`MILP`](@ref), with
[QPSReader.jl](https://github.com/JuliaSmoothOptimizers/QPSReader.jl). Rows and columns keep
the order and the names they have in the file. A maximization problem is turned into a
minimization one by negating its objective (see [`MILP`](@ref)).

MPS is a `Float64`, host-memory format, so the result is a CPU-`Float64` [`MILP`](@ref) built on
`SparseMatrixCSC`.

`dataset`, `name` and `path` are the provenance metadata of the [`MILP`](@ref) to build.
"""
function mps_to_milp(
        file::AbstractString;
        dataset::AbstractString = "", name::AbstractString = "", path::AbstractString = "",
    )
    # QPSReader narrates its parsing through the logger
    qps = with_logger(NullLogger()) do
        return readqps(file)
    end
    isempty(qps.qvals) || throw(
        ArgumentError("MILP only supports linear objectives, but $file has a quadratic one")
    )
    return MILP(qps; dataset, name, path)
end
