function [D_phys, eigvals, eigvecs, S0_dt, FA, MD] = diffusion_tensor_fit(meas, bvals, qhat)
% DIFFUSION_TENSOR_FIT  Weighted linear least-squares diffusion tensor estimation.
%
%   [D_PHYS, EIGVALS, EIGVECS, S0_DT, FA, MD] = DIFFUSION_TENSOR_FIT(
%       MEAS, BVALS, QHAT)
%
%   Fits the diffusion tensor model to diffusion-weighted measurements using
%   weighted linear least squares (wLLS) on the log-signal. The model is:
%
%       log S(b, q) = log S0 - b * q' * D * q
%
%   where D is the 3x3 symmetric diffusion tensor with unique elements
%   [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]. This is linearised as:
%
%       log S_k = [1, -b*qx^2, -2b*qx*qy, -2b*qx*qz, -b*qy^2, -2b*qy*qz, -b*qz^2] * beta
%
%   Weights proportional to S_k^2 are used to account for noise amplification
%   by the log transform (Basser et al. 1994).
%
%   This function is primarily used to obtain a fast, physics-informed starting
%   point for subsequent non-linear ball-and-stick fitting (Section 1.5).
%
%   Inputs:
%     meas   [1xK]  diffusion signal (normalised or raw; row vector)
%     bvals  [1xK]  b-values, s/mm^2
%     qhat   [3xK]  unit gradient directions
%
%   Outputs:
%     D_phys   [3x3]  estimated diffusion tensor (mm^2/s)
%     eigvals  [3x1]  eigenvalues in descending order (lambda1 >= lambda2 >= lambda3)
%     eigvecs  [3x3]  corresponding eigenvectors (columns), sorted by eigvals
%     S0_dt    scalar estimated baseline signal (exp of log-intercept)
%     FA       scalar fractional anisotropy, in [0, 1]
%     MD       scalar mean diffusivity = trace(D)/3, mm^2/s
%
%   DT-to-ball-and-stick initialisation:
%     The principal eigenvector (eigvecs(:,1)) gives the dominant fibre
%     direction. FA approximates the intra-cellular volume fraction f.
%     The largest eigenvalue lambda1 approximates the intrinsic diffusivity d.
%
%     Recommended mapping (Strategy M2, highest p_global in experiments):
%       startx_phys = [S0_dt, MD, FA, theta_dt, phi_dt]
%
%   Example:
%     [~, eigvals, eigvecs, S0_dt, FA, MD] = diffusion_tensor_fit(meas, bvals, qhat);
%     e1       = eigvecs(:, 1);
%     theta_dt = acos(min(max(e1(3), -1), 1));
%     phi_dt   = atan2(e1(2), e1(1));
%     startx_phys = [S0_dt, eigvals(1), FA, theta_dt, phi_dt];
%
%   See also: load_hcp_data, load_isbi_data, multistart_fit.
%
%   Reference:
%     Basser, P.J., Mattiello, J. & LeBihan, D. (1994). MRM 31(4):423-427.
%     Alexander (2009). Modelling, Fitting and Sampling in Diffusion MRI.

% --- Select diffusion-weighted measurements ---
dw_idx = find(bvals > 50 & meas > 0.01);

if length(dw_idx) < 7
    error('diffusion_tensor_fit:insufficientData', ...
          'Too few DW measurements (%d) for tensor fitting (need >= 7).', ...
          length(dw_idx));
end

% --- Build design matrix ---
K_dw = length(dw_idx);
G_dt = zeros(K_dw, 7);
logS = zeros(K_dw, 1);

for i = 1:K_dw
    k  = dw_idx(i);
    b  = bvals(k);
    qx = qhat(1, k);
    qy = qhat(2, k);
    qz = qhat(3, k);
    G_dt(i, :) = [1, -b*qx^2, -2*b*qx*qy, -2*b*qx*qz, ...
                      -b*qy^2, -2*b*qy*qz,  -b*qz^2];
    logS(i) = log(max(meas(k), 1e-10));   % guard against log(0)
end

% --- Weighted least squares (weights = S^2) ---
w      = meas(dw_idx).^2;
W      = diag(w);
beta   = (G_dt' * W * G_dt) \ (G_dt' * W * logS);

% --- Reconstruct symmetric tensor ---
D_phys = [beta(2), beta(3), beta(4);
           beta(3), beta(5), beta(6);
           beta(4), beta(6), beta(7)];

S0_dt  = exp(beta(1));

% --- Eigendecomposition ---
[V, Lambda] = eig(D_phys);
lambdas     = real(diag(Lambda));

% Sort descending
[lambdas_sorted, idx] = sort(lambdas, 'descend');
eigvals = max(lambdas_sorted, 0);     % clamp to non-negative
eigvecs = V(:, idx);

% --- Scalar indices ---
MD = mean(eigvals);
FA = sqrt(1.5) * norm(eigvals - MD) / max(norm(eigvals), 1e-12);
FA = min(max(real(FA), 0), 1);        % clamp to [0, 1]

end
