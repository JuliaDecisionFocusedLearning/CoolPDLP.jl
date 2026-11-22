# Development

## Type stability with DispatchDoctor

All of CoolPDLP is under DispatchDoctor's `@stable`: the source in `src/`, the MOI wrapper and
every package extension in `ext/`. The one exception is `MOI.get` for a
`MOI.RawOptimizerAttribute`, marked `@unstable`, since a raw attribute holds whatever value the
user stored and its return type is `Any` by design.

What the checks do depends on the `dispatch_doctor_mode` preference of CoolPDLP:

| who | mode | why |
|---|---|---|
| users (no preference set) | `"disable"` | `@stable` expands to the plain code. With the checks on, their wrappers raise the inlining cost of hot-path broadcasts, and some of them then allocate |
| test groups `Core`, `MOI`, `Reactant`, `cuda`, `metal` | `"error"` | a call whose return type is not concrete throws a `TypeInstabilityError` |
| test groups `Perf`, `OpenCL` | `"disable"` | `Perf` measures allocations, which only mean something for the code users run |

`test/runtests.jl` sets the preference at the start of every run, with `force = true`. It
writes it to `test/LocalPreferences.toml` (ignored by git), so after a local test run, any
`julia --project=test` session runs with the checks still on, until the next run changes the
mode or you delete the file. Changing the preference recompiles CoolPDLP.

Two consequences when you change code:

- A type instability in the hot loop is caught by `Core` (or `Reactant`), not by `Perf`. If
  `Perf` reports allocations, first check with `Core` that the code is type stable.
- Don't switch `Perf` to `"error"` to debug it: the wrappers change what is inlined, so the
  allocation counts no longer reflect what users get.

### In package extensions

DispatchDoctor reads the preference under the UUID of the module that calls `@stable`, and an
extension has its own UUID, distinct from CoolPDLP's. A literal `default_mode` in an extension
would therefore ignore the test suite's `"error"` and leave the extension unchecked. Every
extension must start with

```julia
using DispatchDoctor: @stable

@stable default_mode = CoolPDLP.dispatch_doctor_mode() begin
    # ...
end
```

`CoolPDLP.dispatch_doctor_mode()` returns CoolPDLP's preference (`"disable"` when unset). The
`Formalities` test set checks that it reaches the GPUArrays extension.

When a method has to return a small union by design (for instance `nothing` or a `Float64`),
widen the limit for that method only, with a nested
`@stable default_union_limit = 2 function ... end`, rather than marking it `@unstable`.
