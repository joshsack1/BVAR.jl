# How This Package Was Built

This page exists because `BVAR.jl` was written with substantial help from a large language
model, and you should know that before you use it to produce a number you intend to
publish. Nothing in the Julia General registry requires this disclosure. It is here
because econometric software fails quietly, and you should be able to calibrate how much
to trust this package rather than having to guess.

## The division of labor

**The mathematics is the author's.** Every method here was chosen and specified by the
author from the literature cited in the docstrings and the Bibliography — the Litterman
(1986) Minnesota prior, the Bańbura–Giannone–Reichlin (2010) dummy-observation
implementation, Chan's (2022) asymmetric conjugate prior, the Baumeister–Hamilton (2019)
structural framework, and so on. The specification of what to build, and of what the
right answer would look like, did not come from a model.

**The implementation code was drafted by a language model.** The author wrote out the
mathematics to be implemented, a language model produced the Julia code in `src/`, and the
author then reviewed that code against the specification. The review was line-by-line, but
it was a review — that is not the same assurance as code written from scratch by someone
holding the whole derivation in mind at the time of writing.

**The tests were written jointly.** The author specified what needed to be verified; the
model helped write the assertions. What they cover is described below.

**The documentation was almost entirely model-written.** Every executable example in these
pages runs on every documentation build, and the build fails if any of them errors — so
the code you see here is verified in the strongest sense available. The surrounding prose,
the mathematical exposition, and the literature citations were drafted by a model and
reviewed by the author, but not audited claim-by-claim. Where this prose and a cited paper
disagree, the paper is right. If you find a claim here that is wrong, please open an
issue.

## What verification stands behind the code

So that you can weigh the above against something concrete, the suite in `test/` contains:

- **Hand-computed known-answer tests.** `ar_residual_covariance` against a hand-worked
  two-variable case; `lag_blocks` against a known simple case; `nonorthogonalized_irf`
  for a VAR(1) against successive powers of ``\Phi``, and for a VAR(2) against
  ``\Phi_1^2 + \Phi_2``; `long_run_multiplier` at ``A = I`` reduced to the familiar
  long-run multiplier; `normal_wishart_prior` moments against the closed-form
  Kadiyala–Karlsson expressions.
- **Reduction identities**, which are what catch a wrong generalization: `dummy_long_run`
  with ``H = I`` must reproduce `dummy_sum_of_coefficients`; a `HamiltonStructuralPrior`
  with ``A`` fully fixed at ``I`` must reproduce the reduced-form path; a composed
  `NormalWishartPrior` must match a manual stack; `lag_blocks` must round-trip
  `VARestimate.β_hat`'s row convention.
- **Cross-method agreement**: the `:ols` and `:fem` VAR estimators against each other; the
  `:sir` and `:mh` structural samplers against each other on a genuinely free case.
- **Simulation-based recovery**: coefficients recovered from data simulated with known
  parameters, both in closed form and through the Gibbs sampler, and posterior means
  checked against the analytic `*_posterior` helpers.
- **Contract and error tests** on the cross-stage seams — the places where a dimension or
  ordering mismatch would otherwise be silent rather than loud. The contracts listed on
  the Home page are documented there because a review found real instances of exactly
  that.
- **Package QA** through `Aqua.jl`.

That suite runs in
[continuous integration](https://github.com/joshsack1/BVAR.jl/actions/workflows/CI.yml) on
every push and pull request, against both the Julia version declared as this package's floor
and latest stable — so the claim that these tests pass is one you can check rather than one
you have to accept. You can also run all of it yourself:

```julia
using Pkg
Pkg.test("BVAR")
```

What this does **not** include: the package has not been validated by replicating the
published results of any paper it cites. If you are using it for research, that is the
check worth doing.

## What this means for you

The failure mode to worry about in model-drafted numerical code is not a crash. It is a
result that is plausible, correctly typed, dimensionally consistent, and wrong — a
coefficient block sliced in the wrong order, a normalization dropped, a prior scaled by
the wrong factor. The tests above are aimed squarely at that class of error, but no test
suite closes it completely.

So, concretely:

- Before you publish a result, reproduce something whose answer you already know.
- Each docstring cites the paper its formula comes from. If a number matters, check the
  formula against the source.
- Please report anything that looks wrong, including in the prose. Bug reports against a
  package built this way are unusually valuable.

## Responsibility

However the code was drafted, the author is responsible for it. `BVAR.jl` is released
under the MIT License, which disclaims warranty — but a disclaimer in a license file is
not the point of this page. The point is that the author reviewed this code, chose to
publish it, and intends to fix what turns out to be broken in it.
