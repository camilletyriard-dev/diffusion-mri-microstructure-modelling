function [chain, acceptance_rate] = mcmc_sampler( ...
    ssd_fn, x0_phys, from_surrogate_fn, sigma_noise, ...
    nIter, burnin, sigma_proposal, verbose)
% MCMC_SAMPLER  Metropolis-Hastings MCMC for diffusion MRI parameter posteriors.
%
%   [CHAIN, ACCEPTANCE_RATE] = MCMC_SAMPLER(SSD_FN, X0_PHYS,
%       FROM_SURROGATE_FN, SIGMA_NOISE, NITER, BURNIN, SIGMA_PROPOSAL, VERBOSE)
%
%   Samples the posterior distribution p(theta | data) using the
%   Metropolis-Hastings algorithm with Gaussian proposals. The likelihood
%   is assumed to be Gaussian:
%
%       log p(data | theta) = -SSD(theta) / (2 * sigma_noise^2)
%
%   so the posterior satisfies:
%
%       log p(theta | data) proportional to -SSD(theta) / (2 * sigma_noise^2)
%
%   A uniform (improper) prior is used over the physical parameter space.
%
%   Algorithm:
%     Initialise at x0_phys (physical space).
%     For each iteration:
%       1. Propose x' = x_current + N(0, diag(sigma_proposal^2))
%       2. Enforce physical validity of x' (clamp or reject).
%       3. Compute log acceptance ratio:
%              log alpha = [SSD(x_current) - SSD(x')] / (2*sigma_noise^2)
%       4. Accept x' with probability min(1, exp(log alpha)).
%     Discard the first BURNIN samples.
%
%   Inputs:
%     ssd_fn            function handle  @(x_phys) -> scalar SSD
%     x0_phys           [1 x nParams]   initial physical parameters
%     from_surrogate_fn (unused here; physical proposals are used directly)
%     sigma_noise       scalar          noise standard deviation
%     nIter             integer         total MCMC iterations (incl. burn-in)
%     burnin            integer         number of burn-in iterations to discard
%     sigma_proposal    [1 x nParams]   Gaussian proposal widths (physical space)
%     verbose           logical         print acceptance rate (default: false)
%
%   Outputs:
%     chain            [(nIter-burnin) x nParams]  post-burn-in samples (physical)
%     acceptance_rate  scalar                       overall acceptance fraction
%
%   Proposal tuning:
%     Aim for an acceptance rate of 20-50% (Roberts & Rosenthal 2001).
%     Tune sigma_proposal sequentially per parameter, then rescale jointly.
%     Typical values for ball-and-stick on HCP data:
%       sigma_proposal = [38, 1.9e-5, 0.011, 0.046, 0.057]
%
%   Example:
%     [chain, acc] = mcmc_sampler( ...
%         @(p) ball_stick_ssd(p, Avox, bvals, qhat), ...
%         x0_phys, [], sigma_noise, 100000, 10000, sigma_prop, true);
%     % Posterior statistics (post burn-in):
%     mu   = mean(chain, 1);
%     sig  = std(chain,  0, 1);
%     ci95 = prctile(chain, [2.5, 97.5], 1);
%
%   See also: bootstrap_uncertainty, ball_stick_ssd.
%
%   Reference:
%     Alexander (2009). Modelling, Fitting and Sampling in Diffusion MRI.
%     Roberts, G.O. & Rosenthal, J.S. (2001). Optimal scaling for various
%       Metropolis-Hastings algorithms. Statistical Science 16(4):351-367.

if nargin < 8, verbose = false; end

nParams  = length(x0_phys);
chain    = zeros(nIter, nParams);
x_curr   = x0_phys;
ssd_curr = ssd_fn(x_curr);
n_accept = 0;

for i = 1:nIter

    % --- Propose new parameters (physical space, Gaussian) ---
    x_prop = x_curr + sigma_proposal .* randn(1, nParams);

    % --- Enforce physical constraints (hard reject if violated) ---
    if x_prop(1) <= 0 || x_prop(2) <= 0 || x_prop(3) <= 0 || x_prop(3) >= 1
        chain(i, :) = x_curr;
        continue
    end
    if nParams >= 6 && (x_prop(4) <= 0 || x_prop(4) >= x_prop(2))
        chain(i, :) = x_curr;
        continue
    end

    % --- Compute log acceptance ratio ---
    ssd_prop  = ssd_fn(x_prop);
    log_alpha = (ssd_curr - ssd_prop) / (2 * sigma_noise^2);

    % --- Accept / reject ---
    if log(rand()) < log_alpha
        x_curr   = x_prop;
        ssd_curr = ssd_prop;
        n_accept = n_accept + 1;
    end

    chain(i, :) = x_curr;

end

% Discard burn-in
chain           = chain(burnin+1:end, :);
acceptance_rate = n_accept / nIter;

if verbose
    fprintf('MCMC complete. Acceptance rate: %.1f%%\n', 100 * acceptance_rate);
    fprintf('Retained %d samples after %d-iteration burn-in.\n', ...
            size(chain, 1), burnin);
end

end
