function [best_x, best_rn, p_global, all_rn, all_x] = multistart_fit( ...
    res_fun, to_surrogate_fn, startx_phys, sigma_phys, nRuns, opts)
% MULTISTART_FIT  Multi-start Levenberg-Marquardt optimisation with global
%                 minimum identification.
%
%   [BEST_X, BEST_RN, P_GLOBAL, ALL_RN, ALL_X] = MULTISTART_FIT(
%       RES_FUN, TO_SURROGATE_FN, STARTX_PHYS, SIGMA_PHYS, NRUNS, OPTS)
%
%   Runs NRUNS independent Levenberg-Marquardt fits from randomly perturbed
%   starting points and identifies the candidate global minimum as the run
%   with the lowest RESNORM.
%
%   Algorithm:
%     For each run r = 1, ..., nRuns:
%       1. Sample a perturbed starting point in physical parameter space:
%              p = startx_phys + sigma_phys .* randn(1, nParams)
%       2. Clamp to physically valid ranges (S0>0, d>0, f in (0,1),
%          and if nParams==6, lambda2 in (0, d)).
%       3. Transform p -> surrogate space via TO_SURROGATE_FN(p).
%       4. Run lsqnonlin (Levenberg-Marquardt) in surrogate space.
%       5. Record the RESNORM and surrogate-space solution.
%     Select the run with the globally lowest RESNORM.
%
%   Global minimum confidence:
%     p_global = #{r : rn_r < best_rn * (1 + tol)} / nRuns
%     where tol = 1e-4. This estimates the probability of finding the
%     global minimum in a single random-start run.
%
%   Inputs:
%     res_fun          function handle  @(x) residuals(x, ...)
%                      Must return a row or column vector of residuals.
%     to_surrogate_fn  function handle  @(p_phys) x_surr
%                      Maps physical -> surrogate parameter vector.
%     startx_phys      [1 x nParams]  nominal physical starting point
%     sigma_phys        [1 x nParams]  standard deviations for perturbation
%     nRuns            integer         number of random-start runs
%     opts             optimoptions    options struct from optimoptions('lsqnonlin',...)
%
%   Outputs:
%     best_x     [1 x nParams]  surrogate-space parameters of the best run
%     best_rn    scalar         lowest RESNORM found
%     p_global   scalar         estimated probability of finding the global min
%     all_rn     [nRuns x 1]   RESNORM for each run (Inf for failed runs)
%     all_x      [nRuns x nP]  surrogate parameters for each run
%
%   Recovering physical parameters from best_x:
%     Use the inverse of TO_SURROGATE_FN. For ball-and-stick:
%       S0 = best_x(1)^2;  d = best_x(2)^2;  f = cos(best_x(3))^2;
%       theta = best_x(4);  phi = best_x(5);
%
%   Example:
%     opts = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
%                         'MaxFunEvals', 20000, 'TolX', 1e-10, ...
%                         'TolFun', 1e-10, 'Display', 'off');
%     to_surr = @(p) [sqrt(p(1)), sqrt(p(2)), acos(sqrt(p(3))), p(4), p(5)];
%     [best_x, best_rn, p_glob] = multistart_fit( ...
%         @(x) ball_stick_residuals(x, meas, bvals, qhat), ...
%         to_surr, [1, 1.5e-3, 0.5, 1.5, 3.0], [0.3, 5e-4, 0.2, 0.5, 1.0], ...
%         1000, opts);
%
%   See also: ball_stick_residuals, zeppelin_stick_residuals,
%             zeppelin_stick_tort_residuals, constraint_transforms.
%
%   Reference:
%     Panagiotaki et al. (2012). NeuroImage 59(3):2241-2254.
%     Alexander (2009). Modelling, Fitting and Sampling in Diffusion MRI.

nP     = length(startx_phys);
all_rn = inf(nRuns, 1);
all_x  = zeros(nRuns, nP);

for r = 1:nRuns

    % --- 1. Perturb starting point in physical space ---
    p = startx_phys + sigma_phys .* randn(1, nP);

    % --- 2. Clamp to physically valid ranges ---
    p(1) = max(p(1), 1e-4);                        % S0 > 0
    p(2) = max(p(2), 1e-6);                        % d  > 0
    p(3) = min(max(p(3), 1e-4), 1 - 1e-4);         % f  in (0, 1)

    if nP == 6
        % Zeppelin-and-Stick: lambda2 in (0, d)
        p(4) = min(max(p(4), 1e-8), p(2) * (1 - 1e-4));
    end

    % --- 3. Transform to surrogate space ---
    x0 = to_surrogate_fn(p);

    % --- 4. Optimise in surrogate space (Levenberg-Marquardt) ---
    try
        [x_hat, rn] = lsqnonlin(res_fun, x0, [], [], opts);
        all_x(r, :) = x_hat;
        all_rn(r)   = rn;
    catch
        % Silently skip failed runs; RESNORM remains Inf
    end

end

% --- 5. Identify candidate global minimum ---
[best_rn, idx] = min(all_rn);
best_x = all_x(idx, :);

% Estimate probability of finding the global minimum
tol      = 1e-4;
p_global = sum(all_rn < best_rn * (1 + tol)) / nRuns;

end
