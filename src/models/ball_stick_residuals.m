function R = ball_stick_residuals(x, meas, bvals, qhat)
% BALL_STICK_RESIDUALS  Residual vector for the ball-and-stick model (LM-compatible).
%
%   R = BALL_STICK_RESIDUALS(X, MEAS, BVALS, QHAT) returns the vector of
%   signed residuals (meas - model) for use with Levenberg-Marquardt via
%   lsqnonlin. The sum of squares of R equals the SSD objective.
%
%   Surrogate reparameterisation (cosine-based, Table B.7 in Alexander 2009):
%
%       S0    = x(1)^2              ensures S0 > 0
%       d     = x(2)^2              ensures d  > 0
%       f     = cos(x(3))^2         ensures f  in [0, 1]
%       theta = x(4)                fibre polar angle
%       phi   = x(5)                fibre azimuthal angle
%
%   The cosine-based f mapping (vs. sine-based in ball_stick_ssd_constrained)
%   is numerically equivalent but consistent with the Panagiotaki (2012) and
%   Ferizi (2014) convention used in the ISBI 2015 challenge codebase.
%
%   Inputs:
%     x      [1x5]   surrogate parameter vector
%     meas   [1xK]   measured (normalised) diffusion signal — row vector
%     bvals  [1xK]   b-values, s/mm^2 — row vector
%     qhat   [3xK]   unit gradient directions — columns are directions
%
%   Output:
%     R      [1xK]   signed residuals: meas - S_model
%
%   Usage with lsqnonlin (Levenberg-Marquardt):
%     opts = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
%                         'MaxFunEvals', 20000, 'TolX', 1e-10, ...
%                         'TolFun', 1e-10, 'Display', 'off');
%     x_hat = lsqnonlin(@(x) ball_stick_residuals(x, meas, bvals, qhat), x0, [], [], opts);
%
%   See also: ball_stick_ssd_constrained, multistart_fit, constraint_transforms.
%
%   Reference:
%     Panagiotaki et al. (2012). NeuroImage 59(3):2241-2254.
%     Ferizi et al. (2014). MRM 72(6):1785-1792.

S0    = x(1)^2;
d     = x(2)^2;
f     = cos(x(3))^2;
theta = x(4);
phi   = x(5);

fibdir     = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
fibdotgrad = fibdir * qhat;   % [1 x K] dot products

S_I = exp(-bvals .* d .* (fibdotgrad.^2));   % intra-cellular (stick)
S_E = exp(-bvals .* d);                       % extra-cellular (ball)
S   = S0 * (f * S_I + (1-f) * S_E);

R   = meas - S;   % [1xK] residual vector

end
