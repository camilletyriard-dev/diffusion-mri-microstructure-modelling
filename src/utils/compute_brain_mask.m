function mask = compute_brain_mask(dwis, bvals, slice_z, threshold)
% COMPUTE_BRAIN_MASK  Intensity-threshold brain mask from mean b=0 image.
%
%   MASK = COMPUTE_BRAIN_MASK(DWIS, BVALS, SLICE_Z, THRESHOLD) constructs a
%   binary brain mask for axial slice SLICE_Z by thresholding the mean
%   b=0 signal. Voxels with mean b=0 signal above THRESHOLD are included.
%
%   Rationale:
%     Background voxels have near-zero signal in all volumes. The mean b=0
%     image provides a high-SNR intensity map that separates brain tissue from
%     background. A threshold of ~1300 (raw signal units) is appropriate for
%     the HCP dataset, excluding CSF and noise while retaining grey and white
%     matter. The optimal threshold should be verified visually (see diagnostic
%     figure generated in run_section1_parameter_estimation.m).
%
%   Inputs:
%     dwis       [K x nx x ny x nz]  4-D dMRI data
%     bvals      [1 x K]             b-values, s/mm^2
%     slice_z    integer             axial slice index
%     threshold  scalar              intensity threshold (default: 1300)
%
%   Output:
%     mask  [nx x ny]  logical array; true = voxel included in fitting
%
%   Usage:
%     mask = compute_brain_mask(dwis, bvals, 72, 1300);
%     fprintf('Brain voxels in slice: %d\n', sum(mask(:)));
%
%   See also: parameter_map_slice, load_hcp_data.

if nargin < 4 || isempty(threshold)
    threshold = 1300;
end

b0_idx  = find(bvals < 1);

if isempty(b0_idx)
    warning('compute_brain_mask:noB0', ...
            'No b=0 volumes found (bvals < 1). Using all volumes for mask.');
    b0_idx = 1:size(dwis, 1);
end

b0_mean = squeeze(mean(dwis(b0_idx, :, :, slice_z), 1));   % [nx x ny]
mask    = b0_mean > threshold;

fprintf('Brain mask: %d voxels above threshold %.0f (out of %d total).\n', ...
        sum(mask(:)), threshold, numel(mask));

end
