# [Data Testing](@id api-data-testing)

```@meta
CurrentModule = BVAR
```

Stage 1: unit-root and cointegration testing, plus the bridge from a `DataFrame`
to the endogenous data matrix every later stage consumes.

```@index
Pages = ["data-testing.md"]
```

## Public API

```@autodocs
Modules = [BVAR]
Pages = [
    "data-testing.jl",
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
    "data-testing.jl",
]
Order = [:module, :type, :function, :macro, :constant]
Public = false
Private = true
```
