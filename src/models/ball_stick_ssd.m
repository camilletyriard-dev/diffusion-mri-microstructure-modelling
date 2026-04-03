function ssd = ball_stick_ssd(x, Avox, bvals, qhat)
% BALL_STICK_SSD  Sum-of-squared-differences objective for the ball-and-stick model.
%
%   SSD = BALL_STICK_SSD(X, AVOX, BVALS, QHAT) computes the sum of squared
%   differences between the measured signal AVOX and the ball-and-stick model
%   prediction for parameter vector X, over K measurements.
%
%   The ball-and-stick signal model is:
%
%       S(b, q) = S0 * [ f * exp(-b*d*(q.n)^2) + (1-f) * exp(-b*d) ]
%
%   where n = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]' is the
%   unit fibre direction vector in spherical coordinates.
%
%   Parameters (physical space — NO constraints enforced):
%     x(1) = S0     baseline signal intensity  (should be > 0)
%     x(2) = d      intrinsic diffusivity, mm^2/s  (should be > 0)
%     x(3) = f      intra-cellular volume fraction  (should be in [0,1])
%     x(4) = theta  polar angle of fibre direction, radians
%     x(5) = phi    azimuthal angle of fibre direction, radians
%
%   Inputs:
%     x      [1x5]   parameter vector in physical space
%     Avox   [Kx1]   measured diffusion signal
%     bvals  [1xK]   b-values, s/mm^2
%     qhat   [3xK]   unit gradient direction vectors
%
%   Output:
%     ssd    scalar   sum of squared residuals
%
%   Note:
%     This function does NOT enforce physical constraints. It is provided for
%     reference and for use with constrained solvers (e.g. fmincon). For
%     gradient-based unconstrained optimisation, use ball_stick_ssd_constrained
%     with the surrogate reparameterisation.
%
%   See also: ball_stick_ssd_constrained, ball_stick_residuals,
%             constraint_transforms.
%
%   Reference:
%     Alexander (2009). Modelling, Fitting and Sampling in Diffusion MRI.
%     Panagiotaki et al. (2012). NeuroImage 59(3):2241-2254.

S0    = x(1);
d     = x(2);
f     = x(3);
theta = x(4);
phi   = x(5);

fibdir     = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
fibdotgrad = sum(qhat .* repmat(fibdir, [length(qhat) 1])');

S   = S0 * (f * exp(-bvals * d .* (fibdotgrad.^2)) + (1-f) * exp(-bvals * d));
ssd = sum((Avox - S').^2);

end
