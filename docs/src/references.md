# Bibliography

The works cited in the docstrings and guide pages.

- **Bańbura, M., Giannone, D., & Reichlin, L.** (2010). "Large Bayesian Vector Auto
  Regressions." *Journal of Applied Econometrics*, 25(1), 71–92.
  The dummy-observation implementation of the Minnesota prior — `dummy_minnesota`.

- **Baumeister, C., & Hamilton, J. D.** (2019). "Structural Interpretation of Vector
  Autoregressions with Incomplete Identification: Revisiting the Role of Oil Supply and Demand
  Shocks." *American Economic Review*, 109(5), 1873–1910.
  The reference prior and the ``p(A)`` structural framework — `baumeister_hamilton_prior`,
  `structural_prior`, `sample_structural`.

- **Blanchard, O. J., & Quah, D.** (1989). "The Dynamic Effects of Aggregate Demand and Supply
  Disturbances." *American Economic Review*, 79(4), 655–673.
  Long-run identifying restrictions — `long_run_multiplier`, `long_run_sign_restriction`.

- **Chan, J. C. C.** (2022). "Asymmetric Conjugate Priors for Large Bayesian VARs."
  *Quantitative Economics*, 13(3), 1145–1169.
  Equation-specific shrinkage that keeps a closed-form posterior —
  `asymmetric_conjugate_prior`.

- **Doan, T., Litterman, R., & Sims, C.** (1984). "Forecasting and Conditional Projection Using
  Realistic Prior Distributions." *Econometric Reviews*, 3(1), 1–100.
  The sum-of-coefficients ("single unit root") prior — `dummy_sum_of_coefficients`.

- **Giannone, D., Lenza, M., & Primiceri, G. E.** (2015). "Prior Selection for Vector
  Autoregressions." *Review of Economics and Statistics*, 97(2), 436–451.
  Hierarchical treatment of the shrinkage hyperparameters.

- **Giannone, D., Lenza, M., & Primiceri, G. E.** (2019). "Priors for the Long Run." *Journal of
  the American Statistical Association*, 114(526), 565–580.
  Generalizes the co-persistence prior to arbitrary linear combinations — `dummy_long_run`.

- **Hamilton, J. D.** (1994). *Time Series Analysis*. Princeton University Press.
  The standard reference for the companion form, impulse responses (p. 260) and the
  information criteria.

- **Kadiyala, K. R., & Karlsson, S.** (1997). "Numerical Methods for Estimation and Inference in
  Bayesian VAR-Models." *Journal of Applied Econometrics*, 12(2), 99–132.
  The Normal-Wishart-compatible form of the Minnesota prior — `normal_wishart_prior`.

- **Kilian, L., & Murphy, D. P.** (2012). "Why Agnostic Sign Restrictions Are Not Enough:
  Understanding the Dynamics of Oil Market VAR Models." *Journal of the European Economic
  Association*, 10(5), 1166–1188.
  Bound (rather than pure sign) restrictions on individual structural coefficients.

- **Litterman, R. B.** (1986). "Forecasting with Bayesian Vector Autoregressions — Five Years of
  Experience." *Journal of Business & Economic Statistics*, 4(1), 25–38.
  The original Minnesota prior — `minnesota_prior`.

- **Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T.** (2010). "Structural Vector
  Autoregressions: Theory of Identification and Algorithms for Inference." *Review of Economic
  Studies*, 77(2), 665–696.
  The QR-rotation algorithm behind `identify_sign_restrictions`.

- **Sims, C. A.** (1980). "Macroeconomics and Reality." *Econometrica*, 48(1), 1–48.
  Recursive (Cholesky) identification — `identify_short_run`.

- **Sims, C. A.** (1993). "A Nine-Variable Probabilistic Macroeconomic Forecasting Model." In
  *Business Cycles, Indicators, and Forecasting*, NBER Studies in Business Cycles, vol. 28,
  179–212. University of Chicago Press.
  The dummy-initial-observation ("co-persistence") prior — `dummy_initial_observation`.

- **Uhlig, H.** (2005). "What Are the Effects of Monetary Policy on Output? Results from an
  Agnostic Identification Procedure." *Journal of Monetary Economics*, 52(2), 381–419.
  Sign-restriction identification — `identify_sign_restrictions`.
