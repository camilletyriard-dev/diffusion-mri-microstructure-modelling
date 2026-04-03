function maps = parameter_map_slice(dwis, bvals, qhat, slice_z, mask, nRuns, startx_phys, sigma_phys)
% PARAMETER_MAP_SLICE  Fit the ball-and-stick model over a full axial slice.
%
%   MAPS = PARAMETER_MAP_SLICE(DWIS, BVALS, QHAT, SLICE_Z, MASK, NRUNS,
%                              STARTX_PHYS, SIGMA_PHYS)
%
%   Iterates over all voxels within MASK at axial index SLICE_Z, fitting the
%   constrained ball-and-stick model at each voxel using NRUNS random-start
%   optimisations. The run with the lowest RESNORM is retained.
%
%   The constrained surrogate reparameterisation (Alexander 2009) is applied
%   at each voxel, ensuring physically valid parameter estimates:
%     S0 = alpha1^2,  d = alpha2^2,  f = sin(alpha3)^2
%
%   Inputs:
%     dwis         [K x nx x ny x nz]  4-D dMRI data array
%     bvals        [1 x K]             b-values, s/mm^2
%     qhat         [3 x K]             unit gradient directions
%     slice_z      integer             axial slice index (1-based)
%     mask         [nx x ny]           logical brain mask for this slice
%     nRuns        integer             number of random-start runs per voxel
%     startx_phys  [1 x 5]            nominal physical starting point
%                                      [S0, d, f, theta, phi]
%     sigma_phys   [1 x 5]            perturbation standard deviations
%
%   Output:
%     maps  struct with fields:
%       .S0       [nx x ny]   baseline signal (a.u.)
%       .d        [nx x ny]   intrinsic diffusivity (mm^2/s)
%       .f        [nx x ny]   intra-cellular volume fraction
%       .theta    [nx x ny]   fibre polar angle (rad)
%       .phi      [nx x ny]   fibre azimuthal angle (rad)
%       .resnorm  [nx x ny]   minimum RESNORM
%
%   Usage:
%     maps = parameter_map_slice(dwis, bvals, qhat, 72, mask, 4, ...
%                [3500, 1.5e-3, 0.3, 0, 0], [1000, 1e-3, 0.1, 0.5, 1.0]);
%     figure; imagesc(maps.f'); colormap(hot); colorbar;
%
%   Computational cost:
%     Approximately 0.003 s per run per voxel (fminunc, quasi-Newton).
%     With N_runs = 4 and ~8000 brain voxels, total time ≈ 1.5 min.
%     Use N_runs = N_95 from a prior single-voxel multi-start analysis.
%
%   See also: ball_stick_ssd_constrained, compute_brain_mask, multistart_fit.
%
%   Reference:
%     Alexander (2009). Modelling, Fitting and Sampling in Diffusion MRI.

nx = size(dwis, 2);
ny = size(dwis, 3);

maps.S0      = zeros(nx, ny);
maps.d       = zeros(nx, ny);
maps.f       = zeros(nx, ny);
maps.theta   = zeros(nx, ny);
maps.phi     = zeros(nx, ny);
maps.resnorm = zeros(nx, ny);

opts = optimset('MaxFunEvals', 20000, ...
                'Algorithm',   'quasi-newton', ...
                'TolX',        1e-10, ...
                'TolFun',      1e-10, ...
                'Display',     'off');

nVox_total = sum(mask(:));
vox_count  = 0;

fprintf('Mapping %d voxels (%d runs each) in slice z=%d...\n', ...
        nVox_total, nRuns, slice_z);
tic;

for xi = 1:nx
    for yi = 1:ny

        if ~mask(xi, yi), continue; end

        Avox    = squeeze(dwis(:, xi, yi, slice_z));
        best_rn = Inf;
        best_p  = zeros(1, 5);

        for r = 1:nRuns

            % Perturb starting point, clamp to physical bounds
            p0    = startx_phys + sigma_phys .* randn(1, 5);
            p0(1) = abs(p0(1));
            p0(2) = abs(p0(2));
            p0(3) = min(max(p0(3), 1e-6), 1 - 1e-6);

            % Transform to surrogate space (sin^2 encoding)
            x0 = [sqrt(p0(1)), sqrt(p0(2)), asin(sqrt(p0(3))), p0(4), p0(5)];

            try
                [p_hat, rn] = fminunc( ...
                    @(x) ball_stick_ssd_constrained(x, Avox, bvals, qhat), x0, opts);
            catch
                continue
            end

            if rn < best_rn
                best_rn = rn;
                best_p  = p_hat;
            end
        end

        % Recover physical parameters from surrogate
        maps.S0(xi, yi)      = best_p(1)^2;
        maps.d(xi, yi)       = best_p(2)^2;
        maps.f(xi, yi)       = sin(best_p(3))^2;
        maps.theta(xi, yi)   = best_p(4);
        maps.phi(xi, yi)     = best_p(5);
        maps.resnorm(xi, yi) = best_rn;

        vox_count = vox_count + 1;
        if mod(vox_count, 500) == 0
            fprintf('  %d / %d voxels  (%.1f min elapsed)\n', ...
                    vox_count, nVox_total, toc / 60);
        end
    end
end

fprintf('Done. Total time: %.1f min.\n', toc / 60);

end
