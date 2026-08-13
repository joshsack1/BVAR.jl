# [Hyperparameters](@id api-hyperparameters)

```@meta
CurrentModule = BayesianVectorAutoregressions
```

Marginal-likelihood evaluation and the hand-rolled golden-section coordinate ascent used to tune
the shrinkage hyperparameters.

Nothing here is exported — this machinery is reached through
[`build_prior`](@ref)'s `hyperparameter_method` keyword rather than called directly. It is
documented because the choice it makes on your behalf materially changes the posterior; see
[What the tuning is maximizing](@ref) in the guide.

```@index
Pages = ["hyperparameters.md"]
```

```@autodocs
Modules = [BayesianVectorAutoregressions]
Pages = [
    "priors/hyperparameters.jl",
]
Order = [:module, :type, :function, :macro, :constant]
```
