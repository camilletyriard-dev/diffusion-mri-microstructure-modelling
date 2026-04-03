function R = zeppelin_stick_residuals(x, meas, bvals, qhat)
% ZEPPELIN_STICK_RESIDUALS  Residual vector for the Zeppelin-and-Stick model.
%
%   R = ZEPPELIN_STICK_RESIDUALS(X, MEAS, BVALS, QHAT) returns the signed
%   residuals (meas - model) for the two-compartment Zeppelin-and-Stick model:
%
%       S(b, q) = S0 * [ f * SI(b,q) + (1-f) * SE(b,q) ]
%
%   where the intra-cellular signal is the stick model:
%
%       SI(b, q) = exp(-b * d * (q.n)^2)
%
%   and the extra-cellular signal is an axially symmetric zeppelin:
%
%       SE(b, q) = exp(-b * [lambda2 + (lambda1 - lambda2) * (q.n)^2])
%
%   with lambda1 = d (parallel diffusivity, shared with the stick) and
%   lambda2 in [0, d] (perpendicular diffusivity, free parameter).
%
%   Surrogate reparameterisation (6 parameters):
%
%       S0      = x(1)^2                 ensures S0 > 0
%       d       = x(2)^2                 ensures d  > 0  (= lambda1)
%       f       = cos(x(3))^2            ensures f  in [0, 1]
%       lambda2 = d * sin(x(4))^2        ensures lambda2 in [0, d]
%       theta   = x(5)                   fibre polar angle
%       phi     = x(6)                   fibre azimuthal angle
%
%   Inputs:
%     x      [1x6]   surrogate parameter vector
%     meas   [1xK]   measured normalised diffusion signal
%     bvals  [1xK]   b-values, s/mm^2
%     qhat   [3xK]   unit gradient directions
%
%   Output:
%     R      [1xK]   signed residuals: meas - S_model
%
%   See also: ball_stick_residuals, zeppelin_stick_tort_residuals,
%             multistart_fit, compute_information_criteria.
%
%   Reference:
%     Panagiotaki et al. (2012). NeuroImage 59(3):2241-2254, Table 1.
%     Ferizi et al. (2014). MRM 72(6):1785-1792.

S0   = x(1)^2;
d    = x(2)^2;
f    = cos(x(3))^2;
lam2 = d * sin(x(4))^2;    % lambda2 in [0, d]
theta = x(5);
phi   = x(6);

n  = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
dn = n * qhat;   % [1 x K] q.n dot products

S_I = exp(-bvals .* d .* dn.^2);
S_E = exp(-bvals .* (lam2 + (d - lam2) .* dn.^2));
S   = S0 * (f * S_I + (1-f) * S_E);

R   = meas - S;

end
