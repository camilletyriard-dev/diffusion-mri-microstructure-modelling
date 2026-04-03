%% run_section3_model_selection.m
%
% Section 3: Model Selection — ISBI 2015 White Matter Benchmark
% COMP0118 — Computational Modelling for Biomedical Imaging
%
% Reproduces all results from Section 3 of the report:
%   Q3.1 — Ball-and-stick fitting to ISBI 2015 data (DT-informed start, LM)
%   Q3.2 — Model comparison: DT, Ball-Stick, Zeppelin-Stick, Zep-Stick-Tort
%   Q3.3 — AIC and BIC model ranking; Akaike weights
%
% Data: ISBI 2015 white matter benchmark — see data/README.md.
%
% Reference:
%   Panagiotaki et al. (2012). NeuroImage 59(3):2241-2254.
%   Ferizi et al. (2014). MRM 72(6):1785-1792.
%   Ferizi et al. (2017). NMR in Biomedicine.

clearvars; close all; clc;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src')));

%% =========================================================================
% Load ISBI 2015 Data
% =========================================================================
[meas, bvals, qhat] = load_isbi_data('data/', 1);

K           = length(meas);
sigma_noise = 0.04;
expected_RN = K * sigma_noise^2;

fprintf('K = %d measurements,  sigma = %.2f,  expected RESNORM = %.4f\n', ...
        K, sigma_noise, expected_RN);

%% =========================================================================
% Q3.1 — Ball-and-Stick Fit (DT-informed initialisation)
% =========================================================================
fprintf('\n--- Q3.1: Ball-and-stick fit (ISBI 2015 data) ---\n');

% Diffusion tensor fit for informed starting point
[~, eigvals_dt, eigvecs_dt, S0_dt, FA_dt, MD_dt] = diffusion_tensor_fit(meas, bvals, qhat);

e1       = eigvecs_dt(:, 1);
theta_dt = acos(min(max(e1(3), -1), 1));
phi_dt   = atan2(e1(2), e1(1));

startx_phys = [S0_dt, MD_dt, FA_dt, theta_dt, phi_dt];
sigma_phys  = [0.3, 5e-4, 0.2, 0.5, 1.0];
nRuns       = 1000;

fprintf('DT starting point: S0=%.4f  d=%.4e  f=%.4f  theta=%.4f  phi=%.4f\n', ...
        startx_phys);

% Optimiser: Levenberg-Marquardt via lsqnonlin
opts_lm = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
    'MaxFunEvals', 20000, 'TolX', 1e-10, 'TolFun', 1e-10, 'Display', 'off');

% Surrogate transform functions (local, defined at end of file)
to_surr_BS  = @(p) [sqrt(p(1)), sqrt(p(2)), acos(sqrt(p(3))), p(4), p(5)];
from_surr_BS = @(x) [x(1)^2, x(2)^2, cos(x(3))^2, x(4), x(5)];

fprintf('\nQ3.1: Running %d multi-start LM fits...\n', nRuns);
[best_x_BS, best_RN_BS, p_glob_BS] = multistart_fit( ...
    @(x) ball_stick_residuals(x, meas, bvals, qhat), ...
    to_surr_BS, startx_phys, sigma_phys, nRuns, opts_lm);

params_BS = from_surr_BS(best_x_BS);
N_95_BS   = ceil(log(0.05) / log(1 - max(p_glob_BS, 1e-9)));

fprintf('Best fit: S0=%.4f  d=%.4e  f=%.4f  theta=%.4f  phi=%.4f\n', params_BS);
fprintf('RESNORM = %.4f  (%.2f x expected)  p_global = %.3f  N_95 = %d\n', ...
        best_RN_BS, best_RN_BS/expected_RN, p_glob_BS, N_95_BS);

% Visualise fit
dn_BS = [cos(params_BS(5))*sin(params_BS(4)), ...
          sin(params_BS(5))*sin(params_BS(4)), cos(params_BS(4))] * qhat;
S_BS  = params_BS(1) * (params_BS(3) * exp(-bvals .* params_BS(2) .* dn_BS.^2) + ...
         (1-params_BS(3)) * exp(-bvals .* params_BS(2)));

figure('Color','w','Position',[50 50 1200 480]);
subplot(1,2,1);
scatter(bvals, meas, 5, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.4);
hold on;
scatter(bvals, S_BS, 5, 'r', 'filled', 'MarkerFaceAlpha', 0.4);
xlabel('b-value (s/mm^2)'); ylabel('Signal'); grid on; hold off;
legend('Data','Model'); title(sprintf('Q3.1: Ball-and-Stick (RN=%.4f)', best_RN_BS));
box off; set(gca,'TickDir','out');

subplot(1,2,2);
resid = meas - S_BS;
scatter(bvals, resid, 5, [0.2 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.4);
hold on; yline(0,'k-'); yline(2*sigma_noise,'r--'); yline(-2*sigma_noise,'r--');
xlabel('b-value (s/mm^2)'); ylabel('Residual');
title('Residuals with \pm2\sigma bounds'); grid on; hold off;
box off; set(gca,'TickDir','out');
sgtitle('Q3.1: Ball-and-Stick fit to ISBI 2015 data');

%% Angular dependence plot
cos_alpha = abs(dn_BS);
figure('Color','w');
scatter(cos_alpha, meas, 5, bvals, 'filled', 'MarkerFaceAlpha', 0.4);
colormap(hot); cb = colorbar; cb.Label.String = 'b-value (s/mm^2)';
xlabel('|cos(\alpha)| = |q \cdot n|'); ylabel('Normalised signal');
title('Q3.1: Signal vs gradient-fibre angle');
box off; grid on; set(gca,'TickDir','out');

%% =========================================================================
% Q3.2 — Model Comparison
% =========================================================================
fprintf('\n--- Q3.2: Fitting four models ---\n');

%% Model 1: Diffusion Tensor (linear fit, no multi-start needed)
[D_phys, eigvals_dt2, eigvecs_dt2, S0_dt2, FA_dt2, MD_dt2] = ...
    diffusion_tensor_fit(meas, bvals, qhat);

dn2 = qhat' * eigvecs_dt2;
S_DT = zeros(1, K);
for k = 1:K
    b = bvals(k);
    S_DT(k) = S0_dt2 * exp(-b * (eigvals_dt2(1)*dn2(k,1)^2 + ...
                                   eigvals_dt2(2)*dn2(k,2)^2 + ...
                                   eigvals_dt2(3)*dn2(k,3)^2));
end
RESNORM_DT = sum((meas - S_DT).^2);
N_params_DT = 7;

%% Model 2: Ball-and-Stick (from DT start) — already fitted above
N_params_BS = 5;

%% Model 3: Zeppelin-and-Stick (initialised from BS)
lam2_init = (1 - params_BS(3)) * params_BS(2);   % tortuosity approximation
startx_ZS = [params_BS(1), params_BS(2), params_BS(3), lam2_init, params_BS(4), params_BS(5)];
sigma_ZS   = [0.3, 5e-4, 0.2, 2e-4, 0.5, 1.0];

to_surr_ZS = @(p) [sqrt(p(1)), sqrt(p(2)), acos(sqrt(p(3))), ...
                    asin(sqrt(max(p(4)/p(2), 0))), p(5), p(6)];
from_surr_ZS = @(x) [x(1)^2, x(2)^2, cos(x(3))^2, x(2)^2*sin(x(4))^2, x(5), x(6)];

fprintf('\nFitting Zeppelin-and-Stick (%d runs)...\n', nRuns);
[best_x_ZS, best_RN_ZS, p_glob_ZS] = multistart_fit( ...
    @(x) zeppelin_stick_residuals(x, meas, bvals, qhat), ...
    to_surr_ZS, startx_ZS, sigma_ZS, nRuns, opts_lm);

params_ZS = from_surr_ZS(best_x_ZS);
N_params_ZS = 6;
fprintf('ZS: RESNORM=%.4f  p_global=%.3f\n', best_RN_ZS, p_glob_ZS);

%% Model 4: Zeppelin-Stick-Tortuosity (from BS, same parameter count)
startx_ZST = params_BS;
sigma_ZST   = sigma_phys;

fprintf('\nFitting Zeppelin-Stick-Tortuosity (%d runs)...\n', nRuns);
[best_x_ZST, best_RN_ZST, p_glob_ZST] = multistart_fit( ...
    @(x) zeppelin_stick_tort_residuals(x, meas, bvals, qhat), ...
    to_surr_BS, startx_ZST, sigma_ZST, nRuns, opts_lm);

params_ZST  = from_surr_BS(best_x_ZST);
N_params_ZST = 5;
fprintf('ZST: RESNORM=%.4f  p_global=%.3f\n', best_RN_ZST, p_glob_ZST);

%% Compute predicted signals for all models
dn_ZS  = [cos(params_ZS(6))*sin(params_ZS(5)), ...
           sin(params_ZS(6))*sin(params_ZS(5)), cos(params_ZS(5))] * qhat;
S_ZS   = params_ZS(1) * (params_ZS(3) * exp(-bvals .* params_ZS(2) .* dn_ZS.^2) + ...
          (1-params_ZS(3)) * exp(-bvals .* (params_ZS(4) + (params_ZS(2)-params_ZS(4)) .* dn_ZS.^2)));

dn_ZST = [cos(params_ZST(5))*sin(params_ZST(4)), ...
           sin(params_ZST(5))*sin(params_ZST(4)), cos(params_ZST(4))] * qhat;
S_ZST  = params_ZST(1) * (params_ZST(3) * exp(-bvals .* params_ZST(2) .* dn_ZST.^2) + ...
           (1-params_ZST(3)) * exp(-bvals .* params_ZST(2) .* ((1-params_ZST(3)) + params_ZST(3) .* dn_ZST.^2)));

model_names   = {'Diffusion Tensor', 'Ball-and-Stick', 'Zeppelin-Stick', 'Zep-Stick-Tort'};
model_signals = {S_DT, S_BS, S_ZS, S_ZST};
model_RN      = [RESNORM_DT, best_RN_BS, best_RN_ZS, best_RN_ZST];
model_pglob   = [NaN, p_glob_BS, p_glob_ZS, p_glob_ZST];

% Model comparison table
fprintf('\n%-26s  %4s  %10s  %8s  %8s\n', 'Model','Np','RESNORM','RN/E[RN]','p_glob');
Np_all = [N_params_DT, N_params_BS, N_params_ZS, N_params_ZST];
for i = 1:4
    if isnan(model_pglob(i))
        fprintf('%-26s  %4d  %10.4f  %8.2f  %8s\n', model_names{i}, Np_all(i), ...
                model_RN(i), model_RN(i)/expected_RN, 'linear');
    else
        fprintf('%-26s  %4d  %10.4f  %8.2f  %8.3f\n', model_names{i}, Np_all(i), ...
                model_RN(i), model_RN(i)/expected_RN, model_pglob(i));
    end
end
fprintf('Expected RESNORM (K*sigma^2) = %.4f\n', expected_RN);

% Data vs model plot
figure('Color','w','Position',[50 50 1400 750]);
for m = 1:4
    subplot(2,2,m);
    scatter(bvals, meas, 3, [0.7 0.7 0.7],'filled','MarkerFaceAlpha',0.4);
    hold on;
    scatter(bvals, model_signals{m}, 3, 'r','filled','MarkerFaceAlpha',0.4);
    xlabel('b-value (s/mm^2)'); ylabel('Signal');
    title(sprintf('%s  (RN=%.4f)', model_names{m}, model_RN(m)));
    legend('Data','Model','Location','northeast'); grid on; hold off;
    box off; set(gca,'TickDir','out');
end
sgtitle('Q3.2: Model fits to ISBI 2015 data');

% Residual plot
figure('Color','w','Position',[80 80 1400 750]);
for m = 1:4
    subplot(2,2,m);
    resid_m = meas - model_signals{m};
    scatter(bvals, resid_m, 3, [0.2 0.5 0.8],'filled','MarkerFaceAlpha',0.4);
    hold on;
    yline(0,'k-'); yline(2*sigma_noise,'r--'); yline(-2*sigma_noise,'r--');
    xlabel('b-value (s/mm^2)'); ylabel('Residual (data - model)');
    title(sprintf('%s  (RN=%.4f)', model_names{m}, model_RN(m)));
    grid on; hold off; box off; set(gca,'TickDir','out');
end
sgtitle('Q3.2: Residuals with \pm2\sigma noise bounds');

%% =========================================================================
% Q3.3 — AIC and BIC Model Selection
% =========================================================================
fprintf('\n--- Q3.3: AIC and BIC model ranking ---\n');

results = compute_information_criteria( ...
    model_names, Np_all, model_RN, K);

% Bar chart
figure('Color','w','Position',[100 100 900 400]);

col_AIC = [0.2 0.45 0.7];
col_BIC = [0.85 0.4 0.2];

subplot(1,2,1);
bar(results.delta_AIC / 1e3, 'FaceColor', col_AIC, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
set(gca, 'XTickLabel', model_names, 'XTickLabelRotation', 22, 'FontSize', 10);
ylabel('\Delta AIC (\times 10^3)');
title('AIC relative to best', 'FontWeight', 'normal');
box off; set(gca,'TickDir','out');

subplot(1,2,2);
bar(results.delta_BIC / 1e3, 'FaceColor', col_BIC, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
set(gca, 'XTickLabel', model_names, 'XTickLabelRotation', 22, 'FontSize', 10);
ylabel('\Delta BIC (\times 10^3)');
title('BIC relative to best', 'FontWeight', 'normal');
box off; set(gca,'TickDir','out');

sgtitle('Q3.3: Model ranking by information criteria');
