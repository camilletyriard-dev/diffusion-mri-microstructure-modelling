%% run_section1_parameter_estimation.m
%
% Section 1: Ball-and-Stick Parameter Estimation and Whole-Slice Mapping
% COMP0118 — Computational Modelling for Biomedical Imaging
%
% Reproduces all results, figures, and tables from Section 1 of the report:
%   Q1.1 — Unconstrained single-voxel fit (fminunc, quasi-Newton)
%   Q1.2 — Constrained fit via surrogate reparameterisation; f-encoding comparison
%   Q1.3 — Multi-start global minimum analysis; p_global and N_95 across voxels
%   Q1.4 — Whole-slice parameter maps (S0, d, f, RESNORM, fibre directions)
%   Q1.5 — Optimisation strategy comparison: fmincon vs DT-informed starts
%
% Data: Human Connectome Project (HCP) — see data/README.md for download.
%
% Dependencies: src/ (add to path before running)
%   addpath(genpath('src'))
%
% Expected runtime:
%   Q1.1-Q1.3: ~5 min  (1000 random starts)
%   Q1.4:      ~2 min  (4 runs/voxel, ~8000 brain voxels)
%   Q1.5:      ~10 min (1000 fmincon runs + DT mapping)
%
% Reference:
%   Alexander (2009). Modelling, Fitting and Sampling in Diffusion MRI.
%   Panagiotaki et al. (2012). NeuroImage 59(3):2241-2254.

clearvars; close all; clc;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src')));

%% =========================================================================
% Load Data
% =========================================================================
[dwis, bvals, qhat] = load_hcp_data('data/');

sigma_noise      = 200;             % noise std dev (signal units)
K                = size(dwis, 1);   % 108 measurements
expected_RESNORM = K * sigma_noise^2;

fprintf('Expected RESNORM under correct model: %.4e\n', expected_RESNORM);

% Reference voxel (white matter, used throughout Sections 1 & 2)
Avox = dwis(:, 92, 65, 72);

%% =========================================================================
% Q1.1 — Unconstrained fit
% =========================================================================
fprintf('\n--- Q1.1: Unconstrained fit ---\n');

startx = [3.5e3, 3e-3, 0.25, 0, 0];

h = optimset('MaxFunEvals', 20000, 'Algorithm', 'quasi-newton', ...
             'TolX', 1e-10, 'TolFun', 1e-10, 'Display', 'off');

[param_hat, RESNORM_unc] = fminunc( ...
    @(x) ball_stick_ssd(x, Avox, bvals, qhat), startx, h);

fprintf('S0=%.2f  d=%.4e  f=%.4f  theta=%.4f  phi=%.4f\n', param_hat);
fprintf('RESNORM = %.4e  (ratio: %.2f x expected)\n', ...
        RESNORM_unc, RESNORM_unc / expected_RESNORM);

plot_model_fit(Avox, param_hat, bvals, qhat, 'ball-stick', ...
               'Q1.1: Unconstrained fit — voxel [92,65,72]');

%% =========================================================================
% Q1.2 — Constrained fit via surrogate reparameterisation
% =========================================================================
fprintf('\n--- Q1.2: Constrained fit (surrogate reparameterisation) ---\n');

% Inverse transform to surrogate starting point
x0_c = [sqrt(startx(1)), sqrt(startx(2)), asin(sqrt(startx(3))), 0, 0];

[x_hat_c, RESNORM_c] = fminunc( ...
    @(x) ball_stick_ssd_constrained(x, Avox, bvals, qhat), x0_c, h);

% Recover physical parameters (sin^2 encoding)
S0_c    = x_hat_c(1)^2;
d_c     = x_hat_c(2)^2;
f_c     = sin(x_hat_c(3))^2;
theta_c = x_hat_c(4);
phi_c   = x_hat_c(5);

fprintf('S0=%.2f  d=%.4e  f=%.4f  theta=%.4f  phi=%.4f\n', ...
        S0_c, d_c, f_c, theta_c, phi_c);
fprintf('RESNORM = %.4e  (ratio: %.2f x expected)\n', ...
        RESNORM_c, RESNORM_c / expected_RESNORM);

params_c = [S0_c, d_c, f_c, theta_c, phi_c];
plot_model_fit(Avox, params_c, bvals, qhat, 'ball-stick', ...
               'Q1.2: Constrained fit — voxel [92,65,72]');

%% =========================================================================
% Q1.3 — Multi-start global minimum analysis
% =========================================================================
fprintf('\n--- Q1.3: Multi-start analysis (1000 runs) ---\n');

startx_phys = [3500, 1.5e-3, 0.25, 0, 0];
sigma_phys  = [1000, 1e-3,   0.1,  0.5, 1.0];
nRuns       = 1000;

all_rn = zeros(nRuns, 1);
all_p  = zeros(nRuns, 5);

tic;
for r = 1:nRuns
    p0    = startx_phys + sigma_phys .* randn(1, 5);
    p0(1) = abs(p0(1));
    p0(2) = abs(p0(2));
    p0(3) = min(max(p0(3), 1e-6), 1 - 1e-6);
    x0    = [sqrt(p0(1)), sqrt(p0(2)), asin(sqrt(p0(3))), p0(4), p0(5)];
    try
        [x_hat, rn] = fminunc( ...
            @(x) ball_stick_ssd_constrained(x, Avox, bvals, qhat), x0, h);
        all_p(r,:)  = [x_hat(1)^2, x_hat(2)^2, sin(x_hat(3))^2, x_hat(4), x_hat(5)];
        all_rn(r)   = rn;
    catch
        all_rn(r) = Inf;
    end
end
t_per_run = toc / nRuns;

globalMin = min(all_rn);
p_global  = sum(all_rn < globalMin * (1 + 1e-4)) / nRuns;
N_95      = ceil(log(0.05) / log(1 - p_global));

fprintf('Global min RESNORM = %.4e  (%.2f x expected)\n', ...
        globalMin, globalMin / expected_RESNORM);
fprintf('p_global = %.2f   N_95 = %d   Time/run = %.3f s\n', ...
        p_global, N_95, t_per_run);

% RESNORM histogram
figure('Color', 'w');
histogram(all_rn(all_rn < 5*globalMin), 80, ...
          'FaceColor', [0.2 0.47 0.7], 'EdgeColor', 'none');
xline(globalMin, 'r--', 'LineWidth', 2, 'DisplayName', 'Global min');
xlabel('RESNORM'); ylabel('Count');
title(sprintf('Q1.3: RESNORM distribution (%d multi-start runs)', nRuns));
legend; grid on; box off; set(gca, 'TickDir', 'out');

%% Multi-voxel analysis
vox_coords = [92, 65; 60, 70; 30, 80; 85, 80];
vox_labels = {'WM [92,65]', 'WM [60,70]', 'GM [30,80]', 'CSF [85,80]'};

fprintf('\n%-14s  %10s  %8s  %4s  %8s  %10s  %6s\n', ...
        'Voxel', 'RESNORM', 'p_global', 'N95', 'S0', 'd (mm2/s)', 'f');
for v = 1:size(vox_coords, 1)
    Av   = dwis(:, vox_coords(v,1), vox_coords(v,2), 72);
    rns  = zeros(nRuns, 1);
    pars = zeros(nRuns, 5);
    for r = 1:nRuns
        p0 = startx_phys + sigma_phys .* randn(1, 5);
        p0(1)=abs(p0(1)); p0(2)=abs(p0(2));
        p0(3)=min(max(p0(3),1e-6),1-1e-6);
        x0=[sqrt(p0(1)),sqrt(p0(2)),asin(sqrt(p0(3))),p0(4),p0(5)];
        try
            [xh,rn]=fminunc(@(x)ball_stick_ssd_constrained(x,Av,bvals,qhat),x0,h);
            pars(r,:)=[xh(1)^2,xh(2)^2,sin(xh(3))^2,xh(4),xh(5)];
            rns(r)=rn;
        catch; rns(r)=Inf; end
    end
    gm=min(rns); pg=sum(rns<gm*(1+1e-4))/nRuns;
    n95=ceil(log(0.05)/log(1-max(pg,1e-9)));
    [~,bi]=min(rns);
    fprintf('%-14s  %10.3e  %8.2f  %4d  %8.1f  %10.4e  %6.4f\n', ...
            vox_labels{v}, gm, pg, n95, pars(bi,1), pars(bi,2), pars(bi,3));
end

%% =========================================================================
% Q1.4 — Whole-slice parameter mapping
% =========================================================================
fprintf('\n--- Q1.4: Whole-slice parameter mapping ---\n');

mask = compute_brain_mask(dwis, bvals, 72, 1300);
maps = parameter_map_slice(dwis, bvals, qhat, 72, mask, N_95, startx_phys, sigma_phys);

% Scalar parameter maps
figure('Color', 'w', 'Position', [100 100 1000 800]);

subplot(2,2,1); imagesc(flipud(maps.S0'));
colormap(gca, gray); colorbar; axis image off;
title('S_0 (a.u.)'); set(gca,'TickDir','out');

subplot(2,2,2); imagesc(flipud(maps.d'));
colormap(gca, gray); colorbar; clim([0, 4e-3]); axis image off;
title('d (mm^2/s)');

subplot(2,2,3); imagesc(flipud(maps.f'));
colormap(gca, hot); colorbar; clim([0, 1]); axis image off;
title('f (intra-cellular fraction)');

subplot(2,2,4); imagesc(flipud(maps.resnorm'));
colormap(gca, jet); colorbar; axis image off;
title('RESNORM');

sgtitle('Q1.4: Ball-and-Stick Parameter Maps — Slice z = 72');

% Fibre direction quiver plot
n_x =  cos(maps.phi) .* sin(maps.theta);
n_y =  sin(maps.phi) .* sin(maps.theta);
flip = cos(maps.theta) < 0;
n_x(flip) = -n_x(flip);  n_y(flip) = -n_y(flip);

step = 3;
[XI, YI] = ndgrid(1:step:size(maps.f,1), 1:step:size(maps.f,2));
u = maps.f(1:step:end, 1:step:end) .* n_x(1:step:end, 1:step:end);
v = maps.f(1:step:end, 1:step:end) .* n_y(1:step:end, 1:step:end);
m = mask(1:step:end, 1:step:end);
u(~m) = 0; v(~m) = 0;

figure('Color', 'w', 'Position', [150 150 700 700]);
imagesc(flipud(maps.f')); colormap(gray); clim([0,1]); colorbar;
hold on;
quiver(XI, size(maps.f,2)+1-YI, u, -v, 0.5, ...
       'Color', [1 0.45 0], 'LineWidth', 0.8);
axis image off;
title('Q1.4: Fibre directions f \cdot n (subsampled, step=3)');
hold off;

%% =========================================================================
% Q1.5.1 — fmincon with explicit bounds
% =========================================================================
fprintf('\n--- Q1.5.1: fmincon (interior-point, explicit bounds) ---\n');

lb = [1e-6, 1e-6, 0,  -Inf, -Inf];
ub = [Inf,  Inf,  1,   Inf,  Inf];
opts_fc = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'MaxFunEvals', 20000, 'TolX', 1e-10, 'TolFun', 1e-10, 'Display', 'off');

all_rn_fc = zeros(nRuns, 1);
tic;
for r = 1:nRuns
    p0 = startx_phys + sigma_phys .* randn(1,5);
    p0(1)=max(p0(1),1e-6); p0(2)=max(p0(2),1e-6); p0(3)=min(max(p0(3),0),1);
    try
        [~,rn]=fmincon(@(x)ball_stick_ssd(x,Avox,bvals,qhat),p0,[],[],[],[],lb,ub,[],opts_fc);
        all_rn_fc(r)=rn;
    catch; all_rn_fc(r)=Inf; end
end
t_fc = toc / nRuns;

gm_fc = min(all_rn_fc);
pg_fc = sum(all_rn_fc < gm_fc*(1+1e-4)) / nRuns;
n95_fc = ceil(log(0.05)/log(1-max(pg_fc,1e-9)));
fprintf('fmincon: p_global=%.2f  N_95=%d  time/run=%.3fs\n', pg_fc, n95_fc, t_fc);

%% =========================================================================
% Q1.5.2 — DT-informed starting point
% =========================================================================
fprintf('\n--- Q1.5.2: DT-informed initialisation ---\n');

tic_dt = tic;
[~, eigvals_dt, eigvecs_dt, S0_dt, FA_dt, MD_dt] = diffusion_tensor_fit(Avox, bvals, qhat);
t_dt = toc(tic_dt);

e1       = eigvecs_dt(:, 1);
theta_dt = acos(min(max(e1(3), -1), 1));
phi_dt   = atan2(e1(2), e1(1));

% Strategy M2 (best in experiments): MD + FA + e1
startx_dt  = [S0_dt, MD_dt, FA_dt, theta_dt, phi_dt];

all_rn_dt  = zeros(nRuns, 1);
tic;
for r = 1:nRuns
    p0 = startx_dt + sigma_phys .* randn(1,5);
    p0(1)=abs(p0(1)); p0(2)=abs(p0(2)); p0(3)=min(max(p0(3),1e-6),1-1e-6);
    x0=[sqrt(p0(1)),sqrt(p0(2)),asin(sqrt(p0(3))),p0(4),p0(5)];
    try
        [xh,rn]=fminunc(@(x)ball_stick_ssd_constrained(x,Avox,bvals,qhat),x0,h);
        all_rn_dt(r)=rn;
    catch; all_rn_dt(r)=Inf; end
end
t_dt_run = toc / nRuns;

gm_dt  = min(all_rn_dt);
pg_dt  = sum(all_rn_dt < gm_dt*(1+1e-4)) / nRuns;
n95_dt = ceil(log(0.05)/log(1-max(pg_dt,1e-9)));

n_brain = sum(mask(:));
fprintf('DT-informed (M2): p_global=%.2f  N_95=%d  time/run=%.3fs\n', pg_dt, n95_dt, t_dt_run);
fprintf('Estimated slice time: %.1f min (vs %.1f min random starts)\n', ...
        n95_dt * t_dt_run * n_brain / 60, N_95 * t_per_run * n_brain / 60);

%% Summary comparison table
fprintf('\n%-35s  %8s  %4s  %10s  %10s\n', 'Method', 'p_glob', 'N95', 't/run (s)', 'Slice (min)');
fprintf('%-35s  %8.2f  %4d  %10.3f  %10.1f\n', 'Q1.3 fminunc + transform', ...
        p_global, N_95, t_per_run, N_95 * t_per_run * n_brain / 60);
fprintf('%-35s  %8.2f  %4d  %10.3f  %10.1f\n', 'Q1.5.1 fmincon + bounds', ...
        pg_fc, n95_fc, t_fc, n95_fc * t_fc * n_brain / 60);
fprintf('%-35s  %8.2f  %4d  %10.3f  %10.1f\n', 'Q1.5.2 DT-informed (M2)', ...
        pg_dt, n95_dt, t_dt_run, n95_dt * t_dt_run * n_brain / 60);
