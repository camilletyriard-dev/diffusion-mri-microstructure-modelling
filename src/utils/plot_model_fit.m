function plot_model_fit(meas, params, bvals, qhat, model_type, title_str)
% PLOT_MODEL_FIT  Visualise measured signal vs model prediction.
%
%   PLOT_MODEL_FIT(MEAS, PARAMS, BVALS, QHAT, MODEL_TYPE, TITLE_STR)
%
%   Plots the measured diffusion signal alongside the model prediction for a
%   fitted parameter set. Supports ball-and-stick, Zeppelin-and-stick, and
%   Zeppelin-stick-tortuosity models.
%
%   Inputs:
%     meas        [Kx1] or [1xK]  measured diffusion signal
%     params      [1 x Np]        fitted physical parameters
%                   Ball-Stick / ZST: [S0, d, f, theta, phi]
%                   Zeppelin-Stick:   [S0, d, f, lambda2, theta, phi]
%     bvals       [1 x K]         b-values, s/mm^2
%     qhat        [3 x K]         unit gradient directions
%     model_type  char            'ball-stick' | 'zeppelin-stick' | 'zep-tort'
%                                 (default: 'ball-stick')
%     title_str   char            plot title (optional)
%
%   Example:
%     params = [4257.9, 1.14e-3, 0.357, -0.981, 0.579];
%     plot_model_fit(Avox, params, bvals, qhat, 'ball-stick', ...
%                   'Q1.2 Constrained fit — voxel [92,65,72]');
%
%   See also: ball_stick_ssd, zeppelin_stick_residuals.

if nargin < 5 || isempty(model_type), model_type = 'ball-stick'; end
if nargin < 6 || isempty(title_str),  title_str  = model_type;   end

meas  = meas(:);   % ensure column
bvals = bvals(:)';
K     = length(meas);

% --- Synthesise model signal ---
S0    = params(1);
d     = params(2);
f     = params(3);

switch lower(model_type)

    case 'ball-stick'
        theta = params(4);
        phi   = params(5);
        n  = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
        dn = n * qhat;
        S_pred = S0 * (f * exp(-bvals .* d .* dn.^2) + (1-f) * exp(-bvals .* d));

    case 'zeppelin-stick'
        lam2  = params(4);
        theta = params(5);
        phi   = params(6);
        n  = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
        dn = n * qhat;
        S_I = exp(-bvals .* d .* dn.^2);
        S_E = exp(-bvals .* (lam2 + (d - lam2) .* dn.^2));
        S_pred = S0 * (f * S_I + (1-f) * S_E);

    case 'zep-tort'
        theta = params(4);
        phi   = params(5);
        n  = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
        dn = n * qhat;
        S_I = exp(-bvals .* d .* dn.^2);
        S_E = exp(-bvals .* d .* ((1-f) + f .* dn.^2));
        S_pred = S0 * (f * S_I + (1-f) * S_E);

    otherwise
        error('plot_model_fit:unknownModel', ...
              'Unknown model type ''%s''. Use ''ball-stick'', ''zeppelin-stick'', or ''zep-tort''.', ...
              model_type);
end

% --- Plot ---
figure('Color', 'w', 'Position', [100 100 800 400]);
plot(1:K, meas,      'bs', 'MarkerSize', 4, 'DisplayName', 'Measured');
hold on;
plot(1:K, S_pred(:), 'rx', 'MarkerSize', 4, 'LineWidth', 1.2, 'DisplayName', 'Model');
hold off;
xlabel('Measurement index k');
ylabel('Signal S');
title(title_str, 'Interpreter', 'none');
legend('Location', 'best');
grid on;
box off;
set(gca, 'TickDir', 'out', 'FontSize', 11);

end
