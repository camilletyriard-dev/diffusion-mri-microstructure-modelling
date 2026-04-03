%% run_section2_uncertainty_estimation.m
%
% Section 2: Uncertainty Estimation via Bootstrap and MCMC
% COMP0118 — Computational Modelling for Biomedical Imaging
%
% Reproduces all results from Section 2 of the report:
%   Q2.1 — Classical bootstrap (T=1000) for S0, d, f; convergence analysis
%   Q2.2 — MCMC Metropolis-Hastings (N=100,000 iter); comparison with bootstrap
%
% Data: Human Connectome Project (HCP) — see data/README.md.
% Run run_section1_parameter_estimation.m first (or load a saved workspace).
%
% Reference:
%   Alexander (2009). Modelling, Fitting and Sampling in Diffusion MRI.
%   Roberts & Rosenthal (2001). Stat. Science 16(4):351-367.

clearvars; close all; clc;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src')));

%% =========================================================================
% Load Data
% =========================================================================
[dwis, bvals, qhat] = load_hcp_data('data/');

sigma_noise = 200;
Avox        = dwis(:, 92, 65, 72);

% Reference fit (constrained, best from multi-start — Section 1.3)
startx_phys = [3500, 1.5e-3, 0.25, 0, 0];
sigma_phys  = [1000, 1e-3,   0.1,  0.5, 1.0];
nRuns_fit   = 4;      % N_95 from Section 1.3

h = optimset('MaxFunEvals', 20000, 'Algorithm', 'quasi-newton', ...
             'TolX', 1e-10, 'TolFun', 1e-10, 'Display', 'off');

% Find reference MLE
best_rn = Inf; best_params = [];
for r = 1:nRuns_fit
    p0 = startx_phys + sigma_phys .* randn(1, 5);
    p0(1)=abs(p0(1)); p0(2)=abs(p0(2)); p0(3)=min(max(p0(3),1e-6),1-1e-6);
    x0=[sqrt(p0(1)),sqrt(p0(2)),asin(sqrt(p0(3))),p0(4),p0(5)];
    try
        [xh,rn]=fminunc(@(x)ball_stick_ssd_constrained(x,Avox,bvals,qhat),x0,h);
        if rn < best_rn
            best_rn = rn;
            best_params = [xh(1)^2, xh(2)^2, sin(xh(3))^2, xh(4), xh(5)];
        end
    catch; end
end
fprintf('Reference MLE: S0=%.1f  d=%.4e  f=%.4f  RESNORM=%.4e\n', ...
        best_params(1), best_params(2), best_params(3), best_rn);

%% =========================================================================
% Q2.1 — Classical Bootstrap
% =========================================================================
fprintf('\n--- Q2.1: Classical bootstrap (T=1000) ---\n');

nBoot = 1000;
K     = length(Avox);

% Inner fit function for bootstrap
function x_best = inner_fit(mb, bb, qb, startx, sigma, nR, h_opts)
    best_rn = Inf; x_best = zeros(1,5);
    for ri = 1:nR
        p0=startx + sigma.*randn(1,5);
        p0(1)=abs(p0(1)); p0(2)=abs(p0(2)); p0(3)=min(max(p0(3),1e-6),1-1e-6);
        x0=[sqrt(p0(1)),sqrt(p0(2)),asin(sqrt(p0(3))),p0(4),p0(5)];
        try
            [xh,rn]=fminunc(@(x)ball_stick_ssd_constrained(x,mb,bb,qb),x0,h_opts);
            if rn < best_rn, best_rn=rn; x_best=[xh(1)^2,xh(2)^2,sin(xh(3))^2,xh(4),xh(5)]; end
        catch; end
    end
end

boot_samples = zeros(nBoot, 5);
fprintf('Running bootstrap...\n');
tic;
for t = 1:nBoot
    idx     = randi(K, 1, K);
    mb      = Avox(idx);
    bb      = bvals(idx);
    qb      = qhat(:, idx);
    boot_samples(t,:) = inner_fit(mb, bb, qb, startx_phys, sigma_phys, nRuns_fit, h);
    if mod(t, 100) == 0
        fprintf('  Bootstrap %d/%d (%.0f s elapsed)\n', t, nBoot, toc);
    end
end
fprintf('Bootstrap complete: %.1f min\n', toc/60);

% Compute intervals for S0, d, f
param_names = {'S0', 'd', 'f'};
for pi = 1:3
    s = boot_samples(:, pi);
    mu_b  = mean(s);
    std_b = std(s);
    ci2s  = [mu_b - 2*std_b, mu_b + 2*std_b];
    ci95  = prctile(s, [2.5, 97.5]);
    fprintf('%s: mean=%.4g  2-sigma=[%.4g, %.4g]  95%%=[%.4g, %.4g]\n', ...
            param_names{pi}, mu_b, ci2s(1), ci2s(2), ci95(1), ci95(2));
end

% Histograms
figure('Color', 'w', 'Position', [50 50 1200 380]);
param_idx = [1, 2, 3];
for pi = 1:3
    subplot(1, 3, pi);
    s    = boot_samples(:, param_idx(pi));
    histogram(s, 40, 'Normalization', 'pdf', ...
              'FaceColor', [0.3 0.55 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    hold on;
    mu_b = mean(s); std_b = std(s);
    x_range = linspace(min(s), max(s), 200);
    plot(x_range, normpdf(x_range, mu_b, std_b), 'r-', 'LineWidth', 1.5);
    xline(prctile(s, 2.5),  'k--', 'LineWidth', 1.2);
    xline(prctile(s, 97.5), 'k--', 'LineWidth', 1.2, 'DisplayName', '95% CI');
    xline(mu_b - 2*std_b, 'g-', 'LineWidth', 1.2);
    xline(mu_b + 2*std_b, 'g-', 'LineWidth', 1.2, 'DisplayName', '2\sigma');
    xlabel(param_names{pi}); ylabel('Density');
    title(sprintf('Bootstrap: %s', param_names{pi}));
    legend({'Samples','Gaussian fit','95% CI','','2\sigma'}, 'FontSize', 8);
    box off; set(gca, 'TickDir', 'out');
    hold off;
end
sgtitle('Q2.1: Bootstrap parameter distributions — voxel [92,65,72]');

% Convergence plot
figure('Color','w','Position',[50 100 1200 600]);
for pi = 1:3
    s = boot_samples(:, param_idx(pi));
    running_mean = cumsum(s) ./ (1:nBoot)';
    running_std  = arrayfun(@(n) std(s(1:n)), 1:nBoot)';
    subplot(2,3,pi);
    plot(1:nBoot, running_mean, 'b-', 'LineWidth', 1.2);
    xlabel('Bootstrap samples'); ylabel('Running mean');
    title(sprintf('%s: convergence of mean', param_names{pi}));
    box off; grid on; set(gca,'TickDir','out');
    subplot(2,3,pi+3);
    plot(1:nBoot, running_std, 'r-', 'LineWidth', 1.2);
    xlabel('Bootstrap samples'); ylabel('Running std');
    title(sprintf('%s: convergence of std', param_names{pi}));
    box off; grid on; set(gca,'TickDir','out');
end
sgtitle('Q2.1: Convergence of bootstrap statistics');

%% =========================================================================
% Q2.2 — MCMC (Metropolis-Hastings)
% =========================================================================
fprintf('\n--- Q2.2: MCMC Metropolis-Hastings ---\n');

% Tuned proposal widths (target acceptance rate 20-50%)
sigma_prop  = [38.12, 1.906e-5, 0.01144, 0.04575, 0.05719];
nIter       = 100000;
burnin      = 10000;

ssd_fn = @(p) ball_stick_ssd(p, Avox, bvals, qhat);

[chain, acc_rate] = mcmc_sampler(ssd_fn, best_params, [], sigma_noise, ...
                                  nIter, burnin, sigma_prop, true);

% Posterior statistics (S0, d, f)
fprintf('\nMCMC posterior (post burn-in, N=%d samples):\n', size(chain,1));
for pi = 1:3
    s     = chain(:, pi);
    mu_m  = mean(s);
    std_m = std(s);
    ci95  = prctile(s, [2.5, 97.5]);
    fprintf('%s: mean=%.4g  2-sigma=[%.4g, %.4g]  95%%=[%.4g, %.4g]\n', ...
            param_names{pi}, mu_m, mu_m-2*std_m, mu_m+2*std_m, ci95(1), ci95(2));
end

% Compare Bootstrap vs MCMC
fprintf('\n%-6s  %-35s  %-35s\n', 'Param', 'Bootstrap (mean ± 2std)', 'MCMC (mean ± 2std)');
for pi = 1:3
    sb = boot_samples(:, param_idx(pi));
    sm = chain(:, pi);
    fprintf('%-6s  [%10.4g, %10.4g]  [%10.4g, %10.4g]\n', ...
            param_names{pi}, ...
            mean(sb)-2*std(sb), mean(sb)+2*std(sb), ...
            mean(sm)-2*std(sm), mean(sm)+2*std(sm));
end

% MCMC trace plots
param_labels = {'S_0', 'd', 'f', '\theta', '\phi'};
figure('Color','w','Position',[50 150 1100 700]);
thin = 500;   % plot every 500th sample for clarity
for pi = 1:5
    subplot(3,2,pi);
    plot((1:thin:size(chain,1)), chain(1:thin:end, pi), 'b-', 'LineWidth', 0.8);
    xlabel('Retained sample index'); ylabel(param_labels{pi});
    title(sprintf('Chain: %s', param_labels{pi}));
    box off; set(gca,'TickDir','out');
end
sgtitle(sprintf('Q2.2: MCMC trace (acceptance = %.1f%%)', 100*acc_rate));

% Posterior distributions (overlay with bootstrap)
figure('Color','w','Position',[80 200 1200 380]);
for pi = 1:3
    subplot(1,3,pi);
    sb = boot_samples(:, param_idx(pi));
    sm = chain(:, pi);
    histogram(sb, 40, 'Normalization','pdf', ...
              'FaceColor',[0.3 0.55 0.8],'FaceAlpha',0.5,'EdgeColor','none');
    hold on;
    histogram(sm, 40, 'Normalization','pdf', ...
              'FaceColor',[0.85 0.4 0.2],'FaceAlpha',0.5,'EdgeColor','none');
    xlabel(param_names{pi}); ylabel('Density');
    title(sprintf('Q2: Bootstrap vs MCMC — %s', param_names{pi}));
    legend({'Bootstrap','MCMC'}, 'FontSize', 9);
    box off; set(gca,'TickDir','out'); hold off;
end
sgtitle('Q2.2: Bootstrap vs MCMC posterior distributions — voxel [92,65,72]');
