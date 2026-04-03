function results = compute_information_criteria(model_names, N_params, SSD_vals, K)
% COMPUTE_INFORMATION_CRITERIA  AIC, BIC, and Akaike weights for model ranking.
%
%   RESULTS = COMPUTE_INFORMATION_CRITERIA(MODEL_NAMES, N_PARAMS, SSD_VALS, K)
%
%   Computes the Akaike Information Criterion (AIC) and Bayesian Information
%   Criterion (BIC) for a set of candidate models, ranks them, and computes
%   Akaike weights as a measure of relative model support.
%
%   Criteria (Gaussian likelihood, estimating sigma from data):
%
%       AIC = 2 * (N_p + 1) + K * log(SSD / K)
%       BIC = (N_p + 1) * log(K) + K * log(SSD / K)
%
%   The "+1" accounts for the noise variance sigma^2 as an additional free
%   parameter (estimated implicitly from the residuals).
%
%   Akaike weights (model probabilities under AIC):
%
%       w_i = exp((AIC_min - AIC_i) / 2) / sum_j exp((AIC_min - AIC_j) / 2)
%
%   Inputs:
%     model_names  {1 x M}  cell array of model name strings
%     N_params     [1 x M]  number of free parameters per model
%     SSD_vals     [1 x M]  minimum SSD (RESNORM) per model
%     K            scalar   number of measurements
%
%   Output:
%     results  struct with fields:
%       .AIC         [1 x M]  AIC values
%       .BIC         [1 x M]  BIC values
%       .AIC_rank    [1 x M]  model indices sorted by AIC (ascending = better)
%       .BIC_rank    [1 x M]  model indices sorted by BIC
%       .AIC_weights [1 x M]  Akaike weights (sum to 1)
%       .delta_AIC   [1 x M]  AIC_i - min(AIC)
%       .delta_BIC   [1 x M]  BIC_i - min(BIC)
%       .consistent  logical  whether AIC and BIC agree on the full ranking
%
%   Usage:
%     results = compute_information_criteria( ...
%         {'DT', 'Ball-Stick', 'Zep-Stick', 'Zep-Stick-Tort'}, ...
%         [7, 5, 6, 5], [223.85, 15.11, 10.82, 11.61], 3612);
%     disp(results.AIC_rank)    % rank 1 = best model
%
%   Interpretation guidelines (Burnham & Anderson 2002):
%     delta_AIC < 2:       substantial support — model is competitive
%     2 < delta_AIC < 7:   considerably less support
%     delta_AIC > 10:      essentially no support
%
%   See also: multistart_fit, ball_stick_residuals, zeppelin_stick_residuals.
%
%   Reference:
%     Burnham, K.P. & Anderson, D.R. (2002). Model Selection and Multimodel
%       Inference: A Practical Information-Theoretic Approach (2nd ed.). Springer.
%     Akaike, H. (1974). IEEE Trans. Autom. Control 19(6):716-723.
%     Schwarz, G. (1978). Ann. Stat. 6(2):461-464.

M   = length(model_names);
AIC = zeros(1, M);
BIC = zeros(1, M);

for i = 1:M
    Np     = N_params(i) + 1;              % +1 for sigma^2
    logL   = K * log(SSD_vals(i) / K);    % Gaussian log-likelihood kernel
    AIC(i) = 2 * Np + logL;
    BIC(i) = Np * log(K) + logL;
end

% Rankings (ascending = better)
[~, AIC_rank] = sort(AIC);
[~, BIC_rank] = sort(BIC);

% Akaike weights
delta_AIC = AIC - min(AIC);
L_aic     = exp(-delta_AIC / 2);
AIC_weights = L_aic / sum(L_aic);

delta_BIC = BIC - min(BIC);

% Consistency check
consistent = all(AIC_rank == BIC_rank);

% Pack output
results.AIC         = AIC;
results.BIC         = BIC;
results.delta_AIC   = delta_AIC;
results.delta_BIC   = delta_BIC;
results.AIC_rank    = AIC_rank;
results.BIC_rank    = BIC_rank;
results.AIC_weights = AIC_weights;
results.consistent  = consistent;

% --- Print summary table ---
fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('%-26s  %4s  %10s  %10s  %10s  %10s\n', ...
        'Model', 'Np', 'SSD', 'AIC', 'dAIC', 'BIC');
fprintf('%s\n', repmat('-', 1, 80));
for i = 1:M
    fprintf('%-26s  %4d  %10.4f  %10.2f  %10.2f  %10.2f\n', ...
            model_names{i}, N_params(i), SSD_vals(i), AIC(i), delta_AIC(i), BIC(i));
end
fprintf('%s\n', repmat('=', 1, 80));

fprintf('\nAIC ranking:\n');
for r = 1:M
    i = AIC_rank(r);
    fprintf('  %d. %-26s  w_AIC = %.4f  (%.1f%%)\n', ...
            r, model_names{i}, AIC_weights(i), 100*AIC_weights(i));
end

if consistent
    fprintf('\nAIC and BIC rankings are consistent.\n');
else
    fprintf('\nWARNING: AIC and BIC rankings disagree.\n');
    fprintf('BIC ranking:\n');
    for r = 1:M
        i = BIC_rank(r);
        fprintf('  %d. %-26s  dBIC = %.2f\n', r, model_names{i}, delta_BIC(i));
    end
end

end
