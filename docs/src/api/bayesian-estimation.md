# [Posterior Sampling](@id api-bayesian-estimation)

```@meta
CurrentModule = BVAR
```

Stage 4b: `sample_posterior` and its six methods — closed-form i.i.d. draws for
the conjugate families, Gibbs sampling for the independent NIW prior.

```@index
Pages = ["bayesian-estimation.md"]
```

## Public API

```@autodocs
Modules = [BVAR]
Pages = [
    "bayesian-estimation/bayesian-estimation.jl",
    "bayesian-estimation/types.jl",
    "bayesian-estimation/conjugate-sampling.jl",
    "bayesian-estimation/independent-niw-gibbs.jl",
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
    "bayesian-estimation/bayesian-estimation.jl",
    "bayesian-estimation/types.jl",
    "bayesian-estimation/conjugate-sampling.jl",
    "bayesian-estimation/independent-niw-gibbs.jl",
]
Order = [:module, :type, :function, :macro, :constant]
Public = false
Private = true
```
