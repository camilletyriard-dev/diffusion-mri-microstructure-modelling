# Data

This directory holds the raw data used in the experiments. Data files are
excluded from version control (see `.gitignore`) and must be downloaded
separately.

---

## Section 1 & 2 — Human Connectome Project (HCP)

**Source:** WU-Minn Human Connectome Project  
**Download:**
```
https://www.dropbox.com/s/q6qeump8knlaav2/data_p1.zip
```

**Instructions:**
1. Download and unzip into this `data/` directory.
2. Expected files after extraction:
   - `data/data.mat`   — dMRI volumes, shape [108 × 145 × 174 × 145]
   - `data/bvecs`      — gradient directions, shape [3 × 108]

**Acquisition protocol:**
- 18 b=0 images + 90 diffusion-weighted volumes
- b-value: 1000 s/mm²
- Voxel size: 1.25 × 1.25 × 1.25 mm³
- Noise standard deviation: σ ≈ 200 (raw signal units)

---

## Section 3 — ISBI 2015 White Matter Benchmark

**Source:** International Symposium on Biomedical Imaging (ISBI) 2015  
**Reference:** Ferizi et al. (2017), *NMR in Biomedicine*  
**Files required:**
- `data/isbi2015_data_normalised.txt`  — normalised signals [3612 × 6 voxels]
- `data/isbi2015_protocol.txt`         — acquisition protocol [7 × 3612]

**Acquisition details:**
- 3612 measurements per voxel, 6 voxels
- Multi-shell: b-values up to ~50,000 s/mm²
- Noise standard deviation: σ ≈ 0.04 (normalised signal units)
- Data are normalised by S(b=0), so the baseline signal is ≈ 1

---

## Loading the Data

```matlab
% HCP data
[dwis, bvals, qhat] = load_hcp_data('data/');

% ISBI 2015 data
[meas, bvals_isbi, qhat_isbi] = load_isbi_data('data/');
```
