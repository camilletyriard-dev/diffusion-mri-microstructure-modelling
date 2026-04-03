function [x_surr, x_phys, x_surr_ZS] = constraint_transforms()
% CONSTRAINT_TRANSFORMS  Surrogate <-> physical parameter transformations.
%
%   This file documents and provides the forward and inverse transformations
%   used throughout this codebase to map unconstrained surrogate parameters
%   to physically valid model parameters.
%
%   The approach follows the reparameterisation strategy described in
%   Alexander (2009, Appendix B), ensuring that gradient-based unconstrained
%   optimisers operate in a space where all reachable points correspond to
%   physically valid models.
%
%   -------------------------------------------------------------------------
%   BALL-AND-STICK / ZEPPELIN-STICK-TORTUOSITY  (5 parameters)
%   -------------------------------------------------------------------------
%
%   Physical -> Surrogate (inverse transform, for initialisation):
%
%     alpha1 = sqrt(S0)          x(1)
%     alpha2 = sqrt(d)           x(2)
%     alpha3 = acos(sqrt(f))     x(3)    [cosine encoding, f in [0,1]]
%     alpha4 = theta             x(4)
%     alpha5 = phi               x(5)
%
%   Surrogate -> Physical (forward transform, applied after fitting):
%
%     S0    = x(1)^2
%     d     = x(2)^2
%     f     = cos(x(3))^2
%     theta = x(4)
%     phi   = x(5)
%
%   -------------------------------------------------------------------------
%   ZEPPELIN-AND-STICK  (6 parameters, adds lambda2)
%   -------------------------------------------------------------------------
%
%   Physical -> Surrogate:
%
%     alpha1 = sqrt(S0)                  x(1)
%     alpha2 = sqrt(d)                   x(2)
%     alpha3 = acos(sqrt(f))             x(3)
%     alpha4 = asin(sqrt(lambda2 / d))   x(4)  [maps lambda2 in [0,d] -> [0,pi/2]]
%     alpha5 = theta                     x(5)
%     alpha6 = phi                       x(6)
%
%   Surrogate -> Physical:
%
%     S0      = x(1)^2
%     d       = x(2)^2
%     f       = cos(x(3))^2
%     lambda2 = d * sin(x(4))^2          [guaranteed in [0, d]]
%     theta   = x(5)
%     phi     = x(6)
%
%   -------------------------------------------------------------------------
%   ALTERNATIVE f ENCODINGS (for comparison, Section 1.2)
%   -------------------------------------------------------------------------
%
%     sin^2 encoding:  f = sin(x3)^2,  inverse: x3 = asin(sqrt(f))
%     Gaussian:        f = exp(-x3^2), inverse: x3 = sqrt(-log(f))
%     Sigmoid:         f = 1/(1+exp(-x3)), inverse: x3 = log(f/(1-f))
%
%   All three map all of R to (0,1). The cosine/sine encodings are periodic
%   and may create multiple equivalent minima; the sigmoid is monotone and
%   avoids this, at the cost of gradient attenuation near f=0 or f=1.
%
%   -------------------------------------------------------------------------
%   HELPER FUNCTIONS (used in scripts and multistart_fit)
%   -------------------------------------------------------------------------

%   These functions are provided as nested/local examples and can be pasted
%   directly into scripts or called as anonymous function handles.

% function x = to_surrogate_bs(p)
%     x = [sqrt(p(1)), sqrt(p(2)), acos(sqrt(p(3))), p(4), p(5)];
% end
%
% function p = from_surrogate_bs(x)
%     p = [x(1)^2, x(2)^2, cos(x(3))^2, x(4), x(5)];
% end
%
% function x = to_surrogate_zs(p)
%     x = [sqrt(p(1)), sqrt(p(2)), acos(sqrt(p(3))), ...
%          asin(sqrt(p(4)/p(2))), p(5), p(6)];
% end
%
% function p = from_surrogate_zs(x)
%     d = x(2)^2;
%     p = [x(1)^2, d, cos(x(3))^2, d*sin(x(4))^2, x(5), x(6)];
% end

% This function returns empty; its purpose is documentation.
x_surr    = [];
x_phys    = [];
x_surr_ZS = [];

end
