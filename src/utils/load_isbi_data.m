function [meas, bvals, qhat, protocol] = load_isbi_data(data_dir, voxel_idx)
% LOAD_ISBI_DATA  Load ISBI 2015 white matter benchmark data and protocol.
%
%   [MEAS, BVALS, QHAT, PROTOCOL] = LOAD_ISBI_DATA(DATA_DIR, VOXEL_IDX)
%
%   Loads the normalised diffusion signal and acquisition protocol from the
%   ISBI 2015 white matter microstructure challenge dataset (Ferizi et al. 2017).
%   The data contain 3612 measurements from 6 white matter voxels, acquired
%   with multiple b-values and gradient directions.
%
%   Data normalisation: each measurement is divided by the mean b=0 signal,
%   so the baseline signal is approximately 1.
%
%   b-values are computed from the Stejskal-Tanner formula:
%
%       b = (gamma * delta * G)^2 * (Delta - delta/3)
%
%   where gamma = 2.675987e8 rad/(s*T) is the proton gyromagnetic ratio,
%   G is the gradient amplitude (T/m), Delta is the pulse separation (s),
%   and delta is the pulse duration (s). Units are converted from s/m^2 to
%   s/mm^2 by dividing by 1e6.
%
%   Inputs:
%     data_dir    char     path to data directory (default: 'data/')
%     voxel_idx   integer  which of the 6 voxels to extract (default: 1)
%
%   Outputs:
%     meas      [1 x K]    normalised diffusion signal (row vector)
%     bvals     [1 x K]    b-values in s/mm^2 (row vector)
%     qhat      [3 x K]    unit gradient direction vectors
%     protocol  struct     raw protocol fields: grad_dirs, G, delta, smalldel, TE
%
%   Example:
%     [meas, bvals, qhat] = load_isbi_data('data/', 1);
%     sigma_noise = 0.04;  % dataset noise standard deviation
%     expected_RN = length(meas) * sigma_noise^2;
%
%   See also: load_hcp_data.
%
%   Reference:
%     Ferizi et al. (2017). NMR in Biomedicine.
%     Ferizi et al. (2014). MRM 72(6):1785-1792.

if nargin < 1 || isempty(data_dir),   data_dir   = 'data/'; end
if nargin < 2 || isempty(voxel_idx),  voxel_idx  = 1;       end

if data_dir(end) ~= filesep && data_dir(end) ~= '/'
    data_dir = [data_dir, '/'];
end

% --- Load signal ---
data_path = fullfile(data_dir, 'isbi2015_data_normalised.txt');
if ~isfile(data_path)
    error('load_isbi_data:fileNotFound', ...
          'Cannot find %s.\nSee data/README.md for download instructions.', data_path);
end

fid  = fopen(data_path, 'r', 'b');
fgetl(fid);                            % skip header
D    = fscanf(fid, '%f', [6, inf])';   % [K x 6]
fclose(fid);

meas = D(:, voxel_idx)';               % [1 x K] row vector

% --- Load protocol ---
prot_path = fullfile(data_dir, 'isbi2015_protocol.txt');
if ~isfile(prot_path)
    error('load_isbi_data:fileNotFound', ...
          'Cannot find %s.\nSee data/README.md for download instructions.', prot_path);
end

fid = fopen(prot_path, 'r', 'b');
fgetl(fid);                            % skip header
A   = fscanf(fid, '%f', [7, inf]);     % [7 x K]
fclose(fid);

protocol.grad_dirs = A(1:3, :);        % [3 x K] raw gradient directions
protocol.G         = A(4, :);          % gradient amplitude (T/m)
protocol.delta     = A(5, :);          % pulse separation (s)
protocol.smalldel  = A(6, :);          % pulse duration (s)
protocol.TE        = A(7, :);          % echo time (s)

% --- Compute b-values (Stejskal-Tanner) ---
GAMMA = 2.675987e8;                    % proton gyromagnetic ratio (rad/s/T)
bvals = ((GAMMA * protocol.smalldel .* protocol.G).^2) .* ...
        (protocol.delta - protocol.smalldel / 3);
bvals = bvals / 1e6;                   % convert s/m^2 -> s/mm^2

% --- Build unit gradient directions ---
K    = size(protocol.grad_dirs, 2);
qhat = zeros(3, K);

for k = 1:K
    gnorm = norm(protocol.grad_dirs(:, k));
    if gnorm > 1e-10
        qhat(:, k) = protocol.grad_dirs(:, k) / gnorm;
    else
        qhat(:, k) = [1; 0; 0];   % arbitrary unit vector for b=0 entries
    end
end

% Force row vectors
bvals = bvals(:)';
meas  = meas(:)';

fprintf('Loaded ISBI 2015 data: %d measurements, voxel %d of 6.\n', K, voxel_idx);
fprintf('  b-value range: [%.1f, %.1f] s/mm^2,  b=0 entries: %d\n', ...
        min(bvals), max(bvals), sum(bvals < 1));
fprintf('  Mean b=0 signal: %.4f (normalised, expect ~1)\n', ...
        mean(meas(bvals < 1)));

end
