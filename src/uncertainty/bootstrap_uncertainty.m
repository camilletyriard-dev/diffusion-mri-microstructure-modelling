function [boot_params, ci_2sigma, ci_95] = bootstrap_uncertainty( ...
    meas, bvals, qhat, fit_fn, from_surrogate_fn, nBoot, verbose)
% BOOTSTRAP_UNCERTAINTY  Classical non-parametric bootstrap for dMRI parameter
%                        uncertainty quantification.
%
%   [BOOT_PARAMS, CI_2SIGMA, CI_95] = BOOTSTRAP_UNCERTAINTY(
%       MEAS, BVALS, QHAT, FIT_FN, FROM_SURROGATE_FN, NBOOT, VERBOSE)
%
%   Implements the classical bootstrap (Efron & Tibshirani 1993) to estimate
%   the sampling distribution of model parameter estimates. In each bootstrap
%   replicate, K measurements are resampled with replacement and the model is
%   re-fitted from scratch using the same multi-start procedure as the original
%   fit. The distribution of bootstrap estimates approximates the posterior
%   p(theta | data) without requiring distributional assumptions.
%
%   Algorithm (per bootstrap sample t = 1, ..., T):
%     1. Sample K indices uniformly at random with replacement.
%     2. Extract the corresponding subset of {meas, bvals, qhat}.
%     3. Re-fit the model using FIT_FN on the bootstrap sample.
%     4. Store the recovered physical parameters.
%
%   Inputs:
%     meas             [1xK]  measured diffusion signal
%     bvals            [1xK]  b-values, s/mm^2
%     qhat             [3xK]  unit gradient directions
%     fit_fn           function handle
%                        @(meas_b, bvals_b, qhat_b) -> x_surr_best
%                        Fits the model and returns the best surrogate params.
%     from_surrogate_fn  function handle
%                        @(x_surr) -> p_phys  (row vector of physical params)
%     nBoot            integer  number of bootstrap replicates (default: 1000)
%     verbose          logical  print progress every 100 samples (default: false)
%
%   Outputs:
%     boot_params  [nBoot x nParams]  physical parameter estimates per sample
%     ci_2sigma    [2 x nParams]      [mean - 2*std; mean + 2*std] intervals
%     ci_95        [2 x nParams]      empirical 2.5th/97.5th percentile range
%
%   Usage:
%     fit_fn = @(mb, bb, qb) my_multistart_fn(mb, bb, qb);
%     from_surr = @(x) [x(1)^2, x(2)^2, cos(x(3))^2, x(4), x(5)];
%     [bparams, ci2s, ci95] = bootstrap_uncertainty(meas, bvals, qhat, ...
%                                 fit_fn, from_surr, 1000, true);
%
%   Note on convergence:
%     Alexander (2009) and McLaughlin (1999) recommend T >= 1000 for stable
%     estimates. Convergence can be assessed by plotting the running mean and
%     standard deviation as a function of T.
%
%   See also: mcmc_sampler, multistart_fit.
%
%   Reference:
%     Alexander (2009). Modelling, Fitting and Sampling in Diffusion MRI.
%     Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap.

if nargin < 6, nBoot   = 1000; end
if nargin < 7, verbose = false; end

K           = length(meas);
boot_params = [];

for t = 1:nBoot

    % --- Resample with replacement ---
    idx    = randi(K, 1, K);
    meas_b = meas(idx);
    bvals_b = bvals(idx);
    qhat_b  = qhat(:, idx);

    % --- Fit model to bootstrap sample ---
    try
        x_surr = fit_fn(meas_b, bvals_b, qhat_b);
        p_phys = from_surrogate_fn(x_surr);
        boot_params(end+1, :) = p_phys; %#ok<AGROW>
    catch
        % Silently skip failed bootstrap replicates
    end

    if verbose && mod(t, 100) == 0
        fprintf('  Bootstrap: %d / %d samples\n', t, nBoot);
    end

end

% --- Compute uncertainty intervals ---
mu  = mean(boot_params, 1);
sig = std(boot_params, 0, 1);

ci_2sigma = [mu - 2*sig; mu + 2*sig];
ci_95     = prctile(boot_params, [2.5, 97.5], 1);

end
