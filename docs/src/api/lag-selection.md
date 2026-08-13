# [Lag Selection](@id api-lag-selection)

```@meta
CurrentModule = BayesianVectorAutoregressions
```

Stage 2: the four information criteria used to pick the lag order `p`, and the
lightweight `VARresult` fit they score.

```@index
Pages = ["lag-selection.md"]
```

## Public API

```@autodocs
Modules = [BayesianVectorAutoregressions]
Pages = [
    "information-criterion.jl",
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
Modules = [BayesianVectorAutoregressions]
Pages = [
    "information-criterion.jl",
]
Order = [:module, :type, :function, :macro, :constant]
Public = false
Private = true
```
