# [Priors](@id api-priors)

```@meta
CurrentModule = BayesianVectorAutoregressions
```

Stage 4a: the nine prior families, the composable dummy-observation priors, and
the closed-form posterior updates they support.

```@index
Pages = ["priors.md"]
```

## Public API

```@autodocs
Modules = [BayesianVectorAutoregressions]
Pages = [
    "priors/priors.jl",
    "priors/types.jl",
    "priors/minnesota.jl",
    "priors/dummy-observations.jl",
    "priors/conjugate.jl",
    "priors/hamilton.jl",
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
    "priors/priors.jl",
    "priors/types.jl",
    "priors/minnesota.jl",
    "priors/dummy-observations.jl",
    "priors/conjugate.jl",
    "priors/hamilton.jl",
]
Order = [:module, :type, :function, :macro, :constant]
Public = false
Private = true
```
