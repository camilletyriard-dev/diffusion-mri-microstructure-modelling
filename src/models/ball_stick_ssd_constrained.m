function ssd = ball_stick_ssd_constrained(x, Avox, bvals, qhat)
% BALL_STICK_SSD_CONSTRAINED  Constrained SSD via surrogate reparameterisation.
%
%   SSD = BALL_STICK_SSD_CONSTRAINED(X, AVOX, BVALS, QHAT) computes the
%   sum of squared differences between the measured signal AVOX and the
%   ball-and-stick model, with physical constraints enforced implicitly
%   through a smooth surrogate mapping.
%
%   Surrogate reparameterisation (Alexander 2009, Appendix B):
%
%       S0    = x(1)^2              ensures S0 > 0
%       d     = x(2)^2              ensures d  > 0
%       f     = sin(x(3))^2         ensures f  in [0, 1]
%       theta = x(4)                unconstrained (spherical coordinate)
%       phi   = x(5)                unconstrained (spherical coordinate)
%
%   The angular parameters theta and phi do not require constraints because
%   n(theta, phi) is a unit vector by construction.
%
%   Inputs:
%     x      [1x5]   surrogate parameter vector (unconstrained reals)
%     Avox   [Kx1]   measured diffusion signal
%     bvals  [1xK]   b-values, s/mm^2
%     qhat   [3xK]   unit gradient direction vectors
%
%   Output:
%     ssd    scalar   sum of squared residuals
%
%   Usage with fminunc:
%     % Starting point: transform physical -> surrogate
%     x0 = [sqrt(S0_init), sqrt(d_init), asin(sqrt(f_init)), theta_init, phi_init];
%     opts = optimset('Algorithm', 'quasi-newton', 'MaxFunEvals', 20000, ...
%                     'TolX', 1e-10, 'TolFun', 1e-10, 'Display', 'off');
%     [x_hat, ssd_min] = fminunc(@(x) ball_stick_ssd_constrained(x, Avox, bvals, qhat), x0, opts);
%     % Recover physical parameters:
%     [S0, d, f, theta, phi] = from_surrogate_bs(x_hat);
%
%   See also: ball_stick_ssd, ball_stick_residuals, constraint_transforms.
%
%   Reference:
%     Alexander (2009). Modelling, Fitting and Sampling in Diffusion MRI.

S0    = x(1)^2;
d     = x(2)^2;
f     = sin(x(3))^2;
theta = x(4);
phi   = x(5);

fibdir     = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
fibdotgrad = sum(qhat .* repmat(fibdir, [length(qhat) 1])');

S   = S0 * (f * exp(-bvals .* d .* (fibdotgrad.^2)) + (1-f) * exp(-bvals .* d));
ssd = sum((Avox - S').^2);

end
