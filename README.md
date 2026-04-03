# Diffusion MRI Microstructure Modelling
---

## Overview

This repository implements a full computational pipeline for **diffusion MRI (dMRI) microstructure modelling**, covering:

- Non-linear parameter estimation and whole-brain mapping of the **ball-and-stick model**
- **Multi-start global optimisation** with physics-informed constraint encoding via surrogate reparameterisation
- **Uncertainty quantification** via the classical bootstrap and Markov Chain Monte Carlo (MCMC, Metropolis-Hastings)
- **Model selection** across a hierarchy of two-compartment models using AIC and BIC information criteria, applied to the ISBI 2015 white matter benchmark dataset

All numerical experiments are reproducible from publicly available data. The codebase is written in MATLAB and is structured for clarity, modularity, and scientific correctness.

---

## Scientific Background

Diffusion-weighted MRI encodes tissue microstructure through the attenuation of the MR signal by the self-diffusion of water molecules. The measured signal at b-value $b$ and gradient direction $\hat{\mathbf{q}}$ is modelled as:

$$S(b, \hat{\mathbf{q}}) = S_0 \left[ f \cdot S_I(b, \hat{\mathbf{q}}) + (1 - f) \cdot S_E(b, \hat{\mathbf{q}}) \right]$$

where $S_0$ is the baseline (b=0) signal, $f \in [0,1]$ is the intra-cellular volume fraction, and $S_I$, $S_E$ are the intra- and extra-cellular signal models, respectively.

### Models Implemented

| Model | $S_I$ | $S_E$ | Free Parameters |
|-------|-------|-------|-----------------|
| Diffusion Tensor (DT) | — | $\exp(-b \hat{\mathbf{q}}^\top \mathbf{D} \hat{\mathbf{q}})$ | 7 |
| Ball-and-Stick (BS) | $\exp(-bd(\hat{\mathbf{q}} \cdot \mathbf{n})^2)$ | $\exp(-bd)$ | 5 |
| Zeppelin-and-Stick (ZS) | $\exp(-bd(\hat{\mathbf{q}} \cdot \mathbf{n})^2)$ | $\exp\!\left(-b[\lambda_2 + (\lambda_1 - \lambda_2)(\hat{\mathbf{q}} \cdot \mathbf{n})^2]\right)$ | 6 |
| Zeppelin-Stick-Tortuosity (ZST) | $\exp(-bd(\hat{\mathbf{q}} \cdot \mathbf{n})^2)$ | $\exp\!\left(-bd[(1-f) + f(\hat{\mathbf{q}} \cdot \mathbf{n})^2]\right)$ | 5 |

The fibre direction is parameterised in spherical coordinates:
$$\mathbf{n}(\theta, \phi) = [\cos\phi\sin\theta,\ \sin\phi\sin\theta,\ \cos\theta]^\top$$

### Surrogate Reparameterisation

Physical constraints ($S_0 > 0$, $d > 0$, $f \in [0,1]$) are enforced via smooth surrogate variables, enabling unconstrained optimisation:

| Physical | Surrogate | Mapping |
|----------|-----------|---------|
| $S_0 > 0$ | $\alpha_1 \in \mathbb{R}$ | $S_0 = \alpha_1^2$ |
| $d > 0$ | $\alpha_2 \in \mathbb{R}$ | $d = \alpha_2^2$ |
| $f \in [0,1]$ | $\alpha_3 \in \mathbb{R}$ | $f = \cos^2(\alpha_3)$ |
| $\lambda_2 \in [0, d]$ | $\alpha_4 \in \mathbb{R}$ | $\lambda_2 = d \cdot \sin^2(\alpha_4)$ |

### Model Selection

Models are ranked using the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC):

$$\text{AIC} = 2N_p + K \log(\text{SSD}/K), \qquad \text{BIC} = N_p \log K + K \log(\text{SSD}/K)$$

where $N_p$ is the number of free parameters, $K$ is the number of measurements, and SSD is the sum of squared residuals at the optimum.

---

## Results Summary

### Section 1 — Parameter Estimation (HCP Data, Voxel [92,65,72])

| Method | RESNORM | RESNORM / Expected | $p_\text{global}$ | $N_{95}$ |
|--------|---------|--------------------|-------------------|----------|
| Unconstrained (fminunc) | $3.06 \times 10^7$ | 7.08 | — | — |
| Constrained (transform) | $5.87 \times 10^6$ | 1.36 | 0.61 | 4 |
| DT-informed start (M2) | $5.87 \times 10^6$ | 1.36 | 0.79 | 2 |

**Best-fit parameters:** $S_0 = 4257.9$, $d = 1.14 \times 10^{-3}\ \text{mm}^2/\text{s}$, $f = 0.357$

### Section 2 — Uncertainty Estimation (Voxel [92,65,72])

| Method | $S_0$ (mean ± 2σ) | $d$ (mean ± 2σ) | $f$ (mean ± 2σ) |
|--------|-------------------|-----------------|-----------------|
| Bootstrap (T=1000) | $4258 \pm 110$ | $(1.14 \pm 0.13) \times 10^{-3}$ | $0.355 \pm 0.093$ |
| MCMC (N=18,000) | $4258 \pm 94$ | $(1.14 \pm 0.06) \times 10^{-3}$ | $0.356 \pm 0.038$ |

### Section 3 — Model Selection (ISBI 2015 Data, K=3612)

| Model | $N_p$ | RESNORM | AIC | BIC | Rank |
|-------|--------|---------|-----|-----|------|
| Diffusion Tensor | 7 | 223.85 | −10029 | −9980 | 4 |
| Ball-and-Stick | 5 | 15.11 | −19771 | −19734 | 3 |
| **Zeppelin-and-Stick** | **6** | **10.82** | **−20975** | **−20932** | **1** |
| Zeppelin-Stick-Tortuosity | 5 | 11.61 | −20723 | −20686 | 2 |

The Zeppelin-and-Stick model is selected by both AIC and BIC, indicating that anisotropic extra-cellular diffusion is a statistically justified addition. The tortuosity constraint imposes $\lambda_2 = (1-f)d$ but achieves a worse fit than the unconstrained Zeppelin, suggesting the constraint is too rigid for this dataset.

---

## Repository Structure

```
diffusion-mri-microstructure/
│
├── README.md
│
├── data/
│   └── README.md                        # Data download instructions
│
├── src/
│   ├── models/
│   │   ├── ball_stick_ssd.m             # Unconstrained SSD objective
│   │   ├── ball_stick_ssd_constrained.m # Constrained SSD (surrogate space)
│   │   ├── ball_stick_residuals.m       # Residual vector (LM-compatible)
│   │   ├── zeppelin_stick_residuals.m   # Zeppelin-and-Stick residuals
│   │   └── zeppelin_stick_tort_residuals.m  # ZS + tortuosity residuals
│   │
│   ├── optimisation/
│   │   ├── multistart_fit.m             # Multi-start LM optimisation engine
│   │   └── constraint_transforms.m      # Surrogate ↔ physical transforms
│   │
│   ├── uncertainty/
│   │   ├── bootstrap_uncertainty.m      # Classical bootstrap (Algorithm 5.1)
│   │   └── mcmc_sampler.m               # Metropolis-Hastings MCMC
│   │
│   ├── mapping/
│   │   └── parameter_map_slice.m        # Whole-slice parameter mapping
│   │
│   ├── model_selection/
│   │   └── compute_information_criteria.m  # AIC, BIC, Akaike weights
│   │
│   └── utils/
│       ├── load_hcp_data.m              # Load HCP dMRI data + bvecs
│       ├── load_isbi_data.m             # Load ISBI 2015 protocol + data
│       ├── diffusion_tensor_fit.m       # Weighted linear DT estimation
│       ├── compute_brain_mask.m         # Intensity-threshold brain mask
│       └── plot_model_fit.m             # Data vs model visualisation
│
└── scripts/
    ├── run_section1_parameter_estimation.m
    ├── run_section2_uncertainty_estimation.m
    └── run_section3_model_selection.m
```

---

## Getting Started

### Requirements

- MATLAB R2019b or later (Optimization Toolbox required for `fminunc`, `fmincon`, `lsqnonlin`)
- No additional toolboxes required for core fitting

### Data

**Section 1 & 2 — Human Connectome Project (HCP) data:**
```
https://www.dropbox.com/s/q6qeump8knlaav2/data_p1.zip
```
Extract into `data/`. Expected files: `data.mat`, `bvecs`.

**Section 3 — ISBI 2015 white matter benchmark:**
Available from the ISBI 2015 challenge supplementary materials (Ferizi et al., 2017).  
Expected files: `isbi2015_data_normalised.txt`, `isbi2015_protocol.txt` — place in `data/`.

### Running the Pipeline

Add the source tree to your MATLAB path:
```matlab
addpath(genpath('src'));
```

Then run each section:
```matlab
run('scripts/run_section1_parameter_estimation.m')
run('scripts/run_section2_uncertainty_estimation.m')
run('scripts/run_section3_model_selection.m')
```

---

## Mathematical Notes

### Global Minimum Estimation

The probability of identifying the global minimum in a single random-start run is estimated empirically as:
$$\hat{p}_\text{global} = \frac{\#\{r : L_r < L^* + \epsilon\}}{N_\text{runs}}, \quad \epsilon = 10^{-4} \cdot L^*$$

The minimum number of runs required to find the global minimum with probability $\geq 0.95$ is:
$$N_{95} = \left\lceil \frac{\log(0.05)}{\log(1 - \hat{p}_\text{global})} \right\rceil$$

### Bootstrap

The classical (non-parametric) bootstrap resamples the $K$ measurements with replacement $T$ times and re-fits the model to each bootstrap dataset. The empirical distribution of bootstrap estimates approximates the sampling distribution of the MLE.

### MCMC (Metropolis-Hastings)

The posterior $p(\mathbf{x} | \mathbf{A}) \propto \exp(-L(\mathbf{x}) / 2\sigma^2)$ is sampled using coordinate-wise Gaussian proposals. Proposal widths are tuned to achieve an acceptance rate of 20–50%. A burn-in period is discarded before computing posterior statistics.

---

## References

1. Alexander, D.C. (2009). *Modelling, Fitting and Sampling in Diffusion MRI.* In: Visualisation and Processing of Tensor Fields, Springer.
2. Panagiotaki, E. et al. (2012). Compartment models of the diffusion MR signal in brain white matter: a taxonomy and comparison. *NeuroImage*, 59(3), 2241–2254.
3. Ferizi, U. et al. (2014). A ranking of diffusion MRI compartment models with in vivo human brain data. *Magnetic Resonance in Medicine*, 72(6), 1785–1792.
4. Ferizi, U. et al. (2017). Diffusion MRI microstructure models with in vivo human brain Connectome data: results from a multi-group comparison. *NMR in Biomedicine*.
5. Zhang, H. et al. (2012). NODDI: practical in vivo neurite orientation dispersion and density imaging of the human brain. *NeuroImage*, 61(4), 1000–1016.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{tyriard2026dmri,
  author       = {Tyriard, Camille},
  title        = {Diffusion MRI Microstructure Modelling},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/camille-tyriard/diffusion-mri-microstructure-modelling}
}
```
