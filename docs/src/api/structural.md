# [Structural Identification](@id api-structural)

```@meta
CurrentModule = BVAR
```

Stage 5: Cholesky short-run identification, sign restrictions, and the general
Baumeister–Hamilton ``p(A)`` framework with its SIR and Metropolis-Hastings samplers.

```@index
Pages = ["structural.md"]
```

## Public API

```@autodocs
Modules = [BVAR]
Pages = [
    "structural-identification/structural-identification.jl",
    "structural-identification/types.jl",
    "structural-identification/hamilton-structural.jl",
    "structural-identification/identification.jl",
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
    "structural-identification/structural-identification.jl",
    "structural-identification/types.jl",
    "structural-identification/hamilton-structural.jl",
    "structural-identification/identification.jl",
]
Order = [:module, :type, :function, :macro, :constant]
Public = false
Private = true
```
