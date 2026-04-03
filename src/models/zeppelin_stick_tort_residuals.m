function R = zeppelin_stick_tort_residuals(x, meas, bvals, qhat)
% ZEPPELIN_STICK_TORT_RESIDUALS  Residuals for the Zeppelin-Stick-Tortuosity model.
%
%   R = ZEPPELIN_STICK_TORT_RESIDUALS(X, MEAS, BVALS, QHAT) returns the
%   signed residuals for the Zeppelin-and-Stick model with the tortuosity
%   constraint:
%
%       lambda2 = (1 - f) * d
%
%   This physically motivated constraint links the perpendicular extracellular
%   diffusivity to the axonal volume fraction f. As f increases, the extracellular
%   space becomes more confined (tortuous), reducing lambda2. This is the
%   constraint used in NODDI (Zhang et al. 2012) and MMWMD (Alexander et al. 2010).
%
%   Under this constraint, the extra-cellular signal simplifies to:
%
%       SE(b, q) = exp(-b * d * [(1-f) + f * (q.n)^2])
%
%   yielding a 5-parameter model (same count as ball-and-stick):
%
%   Surrogate reparameterisation:
%
%       S0    = x(1)^2            ensures S0 > 0
%       d     = x(2)^2            ensures d  > 0
%       f     = cos(x(3))^2       ensures f  in [0, 1]
%       theta = x(4)              fibre polar angle
%       phi   = x(5)              fibre azimuthal angle
%
%   Inputs:
%     x      [1x5]   surrogate parameter vector
%     meas   [1xK]   measured normalised diffusion signal
%     bvals  [1xK]   b-values, s/mm^2
%     qhat   [3xK]   unit gradient directions
%
%   Output:
%     R      [1xK]   signed residuals: meas - S_model
%
%   Note:
%     This model has the same parameter count as ball-and-stick (N_p = 5) but a
%     strictly different model structure. In AIC/BIC comparisons it is penalised
%     equally to ball-and-stick, while typically achieving a better fit.
%
%   See also: zeppelin_stick_residuals, ball_stick_residuals,
%             compute_information_criteria.
%
%   Reference:
%     Zhang et al. (2012). NODDI. NeuroImage 61(4):1000-1016.
%     Panagiotaki et al. (2012). NeuroImage 59(3):2241-2254.

S0    = x(1)^2;
d     = x(2)^2;
f     = cos(x(3))^2;
theta = x(4);
phi   = x(5);

n  = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
dn = n * qhat;   % [1 x K]

S_I = exp(-bvals .* d .* dn.^2);
S_E = exp(-bvals .* d .* ((1-f) + f .* dn.^2));
S   = S0 * (f * S_I + (1-f) * S_E);

R   = meas - S;

end
