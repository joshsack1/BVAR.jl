# [Impulse Responses](@id api-irf)

```@meta
CurrentModule = BayesianVectorAutoregressions
```

The impulse-response machinery shared by both identification paths: the
coefficient-layout helpers, the companion form, and the long-run multiplier.

```@index
Pages = ["irf.md"]
```

## Public API

```@autodocs
Modules = [BayesianVectorAutoregressions]
Pages = [
    "structural-identification/irf.jl",
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
    "structural-identification/irf.jl",
]
Order = [:module, :type, :function, :macro, :constant]
Public = false
Private = true
```
