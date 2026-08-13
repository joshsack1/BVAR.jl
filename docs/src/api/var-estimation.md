# [VAR Estimation](@id api-var-estimation)

```@meta
CurrentModule = BVAR
```

Stage 3: reduced-form VAR estimation by stacked OLS or equation-by-equation
fixed-effect models. See [The `VARestimate` object](@ref varestimate-object) for its fields.

```@index
Pages = ["var-estimation.md"]
```

## Public API

```@autodocs
Modules = [BVAR]
Pages = [
    "var-estimation.jl",
]
Order = [:module, :type, :function, :macro, :constant]
Public = true
Private = false
```

## Internals

Not exported. Documented because the public docstrings above refer to them by
name, and because the closed-form updates and sampler internals are where the
numerical substance of this package lives.

```@autodocs
Modules = [BVAR]
Pages = [
    "var-estimation.jl",
]
Order = [:module, :type, :function, :macro, :constant]
Public = false
Private = true
```
