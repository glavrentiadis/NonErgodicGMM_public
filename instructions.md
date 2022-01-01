Instructions for non-ergodic GMM <br> development using STAN
================
Greg Lavrentiadis
12/31/2021

This documents contains the instructions for estimating the non-ergodic
parameters and hyper-parameters of the known scenarios, and predicting
the non-ergodic parameters for the new scenarios using the
`stan_BA18_f1.00hz_NGAWest2CA_unbounded_hyp.py` and
`stan_BA18_f1.00hz_NGAWest2CA_compute_coeffs.py` scripts. The known
scenarios refer to the scenarios in the regression flatfile. The new
scenarios refer to the scenarios for which the non-ergodic ground motion
is predicted.

## Non-ergodic Regression

The estimation of the non-ergodic coefficients and hyper-parameters is
performed by `stan_BA18_f1.00hz_NGAWest2CA_unbounded_hyp.py`

The user specified arguments for the non-ergodic regression are:

-   `fname_analysis` is the name of the analysis
-   `fname_flatfile` is the file name for the ground motion flatfile
-   `fname_cellinfo` is the file name for the attenuation cells flatfile
-   `fname_celldistfile` is the file name for the cell path distances
-   `dir_out` is the output directory

The main output files are:

-   `fname_analysis + "_stan_fit.pkl"` which contains the raw output
    files,
-   `fname_analysis + "_stan_posterior_raw.csv"` which contains all the
    MCMC samples of the posterior distribution,
-   `fname_analysis + "_stan_posterior.csv"` which contains the
    posterior distributions of the hyper-parameters,
-   `fname_analysis + "_stan_coefficients.csv"` which contains the
    estimated non-ergodic coefficients,
-   `fname_analysis + "_stan_catten.csv"` which contains the estimated
    cell specific attenuation coefficients, and
-   `fname_analysis + "_stan_residuals.csv"` which contains the
    residuals of the non-ergodic model

## Coefficient Prediction

The prediction of the non-ergodic coefficients at new locations is
performed by `stan_BA18_f1.00hz_NGAWest2CA_compute_coeffs.py`

The user specified arguments `fname_analysis`, `fname_resfile`, and
`fname_cellinfo` are the same as in
`stan_BA18_f1.00hz_NGAWest2CA_unbounded_hyp.py`, while `dir_stan`
corresponds to the output directory for the STAN regression.

The output files of this script contain the non-ergodic coefficients
predicted at the new locations.
