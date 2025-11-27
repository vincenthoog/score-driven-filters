# score-driven-filters

Research code for spatio-temporal, multivariate score-driven (GAS/DCS) time-series models with different types of scaling matrices.

## General

Code for building, estimating, and experimenting with score-driven (a.k.a. GAS / DCS) time-series models, with a focus on the effect of different scaling matrices in multivariate settings.

This repository underlies my MSc thesis on **Evaluation of Scaling in Multivariate Score-Driven Filters**, where I compare explicit and implicit filters and study how scaling choices impact stability and forecasting performance.

## Features

- **Explicit and implicit filters**  
  - Explicit and implicit multivariate score-driven filters for Student-t observations.  
  - Unified starting-parameter determination for both frameworks.

- **Multiple scaling choices**  
  - Standard gradient-based (“identity”) scaling.  
  - GASP scaling.  
  - Inverse-Fisher and square-root inverse-Fisher scalings.  
  - Additional variants such as inverse-Hessian / EWMA / local-Hessian (depending on configuration).

- **Spatio-temporal structure**  
  - Spatial dependence via a row-stochastic weight matrix \( W \) built from cross-sectional correlations.  
  - Time-varying latent mean \( \mu_t \) per series with score-driven dynamics.

- **Unified start-parameter logic**  
  - Shared routines to get good initial values for \( \rho_1, \rho_2, \nu, \beta, \lambda, \kappa \).  
  - Grid search + IRLS + local optimization wrapped in one interface.

- **Example analysis scripts**  
  - Generic analysis on user-supplied panel data (e.g. volatilities, returns).  
  - Example using Fama–French 49 industry portfolios (daily excess returns aggregated to monthly log realized variance).  
  - Out-of-sample forecast comparison and Diebold–Mariano tests against baselines such as random walk and sample mean.

## Repository structure

Roughly:

```text
score-driven-filters/
  src/
    score_driven_filters/
      __init__.py
      explicit_filter.py         # explicit score-driven filter + OOS helpers
      implicit_filter.py         # implicit score-driven filter + OOS helpers
      determine_start_parameters.py  # unified start-parameter determination
      get_data_and_W.py          # Fama–French loader + W-construction helper
      analyse.py                 # generic analysis on user-provided data
      analyse_forecast.py        # example forecast script (Fama–French)
