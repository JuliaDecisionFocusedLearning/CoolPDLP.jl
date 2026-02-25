macro public(ex)
    return if VERSION >= v"1.11.0-DEV.469"
        args = ex isa Symbol ? (ex,) : Base.isexpr(ex, :tuple) ? ex.args : error("Wrong expression format")
        esc(Expr(:public, args...))
    else
        nothing
    end
end
