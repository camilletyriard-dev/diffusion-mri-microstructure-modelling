function [dwis, bvals, qhat] = load_hcp_data(data_dir)
% LOAD_HCP_DATA  Load Human Connectome Project dMRI data and acquisition protocol.
%
%   [DWIS, BVALS, QHAT] = LOAD_HCP_DATA(DATA_DIR) loads the dMRI data array,
%   gradient directions, and computes b-values from the bvecs file.
%
%   Expected files in DATA_DIR:
%     data.mat   — MATLAB matrix file containing variable 'dwis'
%     bvecs      — text file of gradient directions, shape [3 x 108]
%
%   The raw data array is permuted from [nx x ny x nz x K] to [K x nx x ny x nz]
%   so that the first dimension indexes measurements (consistent with the
%   model functions in this codebase).
%
%   Acquisition protocol:
%     18 b=0 images + 90 diffusion-weighted images (K = 108 total)
%     b-value: 1000 s/mm^2 for diffusion-weighted volumes
%     Voxel size: 1.25 x 1.25 x 1.25 mm^3
%     Image dimensions: 145 x 174 x 145 voxels
%
%   b-values are computed from the gradient directions as:
%     bvals = 1000 * sum(qhat .* qhat, 1)
%   yielding bvals = 0 for b=0 images and bvals = 1000 for DW images
%   (since bvecs stores unit vectors for DW directions and [0,0,0] for b=0).
%
%   Inputs:
%     data_dir  char  path to data directory (e.g. 'data/')
%
%   Outputs:
%     dwis   [108 x 145 x 174 x 145]  double-precision dMRI data
%     bvals  [1 x 108]                b-values, s/mm^2
%     qhat   [3 x 108]                gradient directions (unit vectors for DW)
%
%   Example:
%     [dwis, bvals, qhat] = load_hcp_data('data/');
%     Avox = dwis(:, 92, 65, 72);   % single voxel time-series
%
%   See also: load_isbi_data, compute_brain_mask.

if nargin < 1 || isempty(data_dir)
    data_dir = 'data/';
end

% Ensure trailing separator
if data_dir(end) ~= filesep && data_dir(end) ~= '/'
    data_dir = [data_dir, '/'];
end

% --- Load dMRI volumes ---
mat_path = fullfile(data_dir, 'data.mat');
if ~isfile(mat_path)
    error('load_hcp_data:fileNotFound', ...
          'Cannot find %s.\nSee data/README.md for download instructions.', mat_path);
end

fprintf('Loading HCP data from %s ...\n', mat_path);
S    = load(mat_path, 'dwis');
dwis = double(S.dwis);
dwis = permute(dwis, [4, 1, 2, 3]);   % -> [K x nx x ny x nz]

% --- Load gradient directions ---
bvec_path = fullfile(data_dir, 'bvecs');
if ~isfile(bvec_path)
    error('load_hcp_data:fileNotFound', ...
          'Cannot find %s.\nSee data/README.md for download instructions.', bvec_path);
end

qhat  = load(bvec_path);          % [3 x K]
bvals = 1000 * sum(qhat .* qhat, 1);  % [1 x K]

fprintf('  Loaded: %d volumes, image size %dx%dx%d, voxel 1.25mm iso.\n', ...
        size(dwis, 1), size(dwis, 2), size(dwis, 3), size(dwis, 4));
fprintf('  b=0 volumes: %d,  DW volumes: %d\n', ...
        sum(bvals < 1), sum(bvals >= 1));

end
