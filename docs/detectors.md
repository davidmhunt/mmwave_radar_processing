# CFAR Detector Guide

Implementation-focused guide for:
- 1D & 2D **CA-CFAR** (Cell Averaging)
- 1D & 2D **GO-CFAR** (Greatest-Of)
- 1D & 2D **SO-CFAR** (Smallest-Of)
- 1D & 2D **OS-CFAR** (Ordered Statistics)
## **NOTE: THE CURRENT FORMULATION IS A FIRST CUT DEVLEPED WITH THE AID OF GPT, IT WILL BE VERIFIED SOON, BUT HAS NOT YET BEEN VERIFIED FOR MATHEMATICAL CORRECTNESS JUST YET**:

## 1. Notation and Setup

We assume a post-detection radar magnitude (or power) domain.

- 1D signal (range profile or Doppler cut): $x[i]$, $i = 0, \dots, L-1$.
- 2D signal (range–Doppler map): $X[r, d]$, $r = 0,\dots,R-1$, $d = 0,\dots,D-1$.
- CUT (Cell Under Test): the sample being tested for a target.
- $N_G$: guard cells per side (1D) or guard half-width in each dimension (2D).
- $N_T$: training cells per side (1D) or training half-width (2D).
- $P_{FA}$: desired probability of false alarm.
- $\alpha$: CFAR scaling factor.

Classical CFAR assumptions:
- Noise after envelope detection is exponential and i.i.d. across training cells.
- Under $H_0$ (no target), the CUT distribution matches the training cells.

---

## 2. CA-CFAR (Cell Averaging CFAR)

### 2.1 1D CA-CFAR: Math

For a CUT at index $i$:

Left training region:
$$
\mathcal{T}_L = \{ x[k] \mid k \in [i - N_G - N_T,\; i - N_G - 1] \}
$$
Right training region:
$$
\mathcal{T}_R = \{ x[k] \mid k \in [i + N_G + 1,\; i + N_G + N_T] \}
$$

Training set and size:
$$
\mathcal{T}_i = \mathcal{T}_L \cup \mathcal{T}_R, \qquad N = 2 N_T
$$

Noise estimate:
$$
\hat{Z}_i = \frac{1}{N} \sum_{x[k] \in \mathcal{T}_i} x[k]
$$

Threshold and scaling factor:
$$
T_i = \alpha \hat{Z}_i, \qquad
\alpha = N \left( P_{FA}^{-1/N} - 1 \right)
$$

Decision rule:
$$
x[i] > T_i \quad \Rightarrow \quad \text{declare target at } i
$$

### 2.2 1D CA-CFAR: Python-Style Pseudocode

```python
def ca_cfar_1d(x, num_guard, num_train, pfa):
    """
    1D CA-CFAR detector.

    Args:
        x: 1D sequence (list/np.array) of length L
        num_guard: guard cells on each side of CUT
        num_train: training cells on each side (beyond guards)
        pfa: desired probability of false alarm

    Returns:
        detections: list[bool] of length L (True = detection)
        thresholds: list[float] of length L (CFAR threshold)
    """
    L = len(x)
    N = 2 * num_train  # total training cells
    alpha = N * (pfa ** (-1.0 / N) - 1.0)

    detections = [False] * L
    thresholds = [0.0] * L

    # only indices where full window fits
    start = num_guard + num_train
    end = L - (num_guard + num_train)

    for i in range(start, end):
        left_start = i - num_guard - num_train
        left_end = i - num_guard
        right_start = i + num_guard + 1
        right_end = i + num_guard + 1 + num_train

        left = x[left_start:left_end]
        right = x[right_start:right_end]

        train_cells = list(left) + list(right)
        noise_est = sum(train_cells) / len(train_cells)

        T = alpha * noise_est
        thresholds[i] = T
        detections[i] = (x[i] > T)

    return detections, thresholds
```

### 2.3 2D CA-CFAR: Math

Consider a 2D map $X[r, d]$.

Half-window sizes:
$$
W_r = N_{G,r} + N_{T,r}, \qquad W_d = N_{G,d} + N_{T,d}
$$

Window sizes:
$$
N_{\text{win}} = (2 W_r + 1)(2 W_d + 1)
$$
$$
N_{\text{guard}} = (2 N_{G,r} + 1)(2 N_{G,d} + 1)
$$
$$
N = N_{\text{win}} - N_{\text{guard}}
$$

Training set for CUT at $(r, d)$: all cells in
$[r - W_r, r + W_r] \times [d - W_d, d + W_d]$
excluding the guard + CUT block
$[r - N_{G,r}, r + N_{G,r}] \times [d - N_{G,d}, d + N_{G,d}]$.

Noise estimate:
$$
\hat{Z}_{r,d} = \frac{1}{N} \sum_{(r', d') \in \mathcal{T}_{r,d}} X[r', d']
$$

Threshold and scaling factor (same CA-CFAR formula):
$$
T_{r,d} = \alpha \hat{Z}_{r,d}, \qquad
\alpha = N \left( P_{FA}^{-1/N} - 1 \right)
$$

Decision rule:
$$
X[r, d] > T_{r,d} \quad \Rightarrow \quad \text{declare target at } (r, d)
$$

### 2.4 2D CA-CFAR: Python-Style Pseudocode

```python
def ca_cfar_2d(X, num_guard_r, num_guard_d, num_train_r, num_train_d, pfa):
    """
    2D CA-CFAR over a range-Doppler map.

    Args:
        X: 2D array-like of shape (R, D)
        num_guard_r: guard half-width in range
        num_guard_d: guard half-width in Doppler
        num_train_r: training half-width in range
        num_train_d: training half-width in Doppler
        pfa: desired probability of false alarm

    Returns:
        detections: 2D list[bool] of shape (R, D)
        thresholds: 2D list[float] of shape (R, D)
    """
    R = len(X)
    D = len(X[0]) if R > 0 else 0

    Wr = num_guard_r + num_train_r
    Wd = num_guard_d + num_train_d

    detections = [[False for _ in range(D)] for _ in range(R)]
    thresholds = [[0.0 for _ in range(D)] for _ in range(R)]

    total_window_cells = (2 * Wr + 1) * (2 * Wd + 1)
    guard_cells = (2 * num_guard_r + 1) * (2 * num_guard_d + 1)
    N = total_window_cells - guard_cells

    alpha = N * (pfa ** (-1.0 / N) - 1.0)

    # iterate only where full window fits
    for r in range(Wr, R - Wr):
        for d in range(Wd, D - Wd):

            train_vals = []
            for rr in range(r - Wr, r + Wr + 1):
                for dd in range(d - Wd, d + Wd + 1):

                    # skip guard + CUT region
                    if (abs(rr - r) <= num_guard_r and
                        abs(dd - d) <= num_guard_d):
                        continue

                    train_vals.append(X[rr][dd])

            noise_est = sum(train_vals) / len(train_vals)
            T = alpha * noise_est

            thresholds[r][d] = T
            detections[r][d] = (X[r][d] > T)

    return detections, thresholds
```

---

## 3. GO-CFAR and SO-CFAR (1D)

GO-CFAR and SO-CFAR improve robustness near clutter edges or strong interferers by treating the two sides of the window separately and combining the estimates with $\max$ (GO) or $\min$ (SO).

For a CUT at index $i$:

Left and right training regions:
$$
\mathcal{T}_L = \{ x[k] \mid k \in [i - N_G - N_T,\; i - N_G - 1] \}, \quad
\mathcal{T}_R = \{ x[k] \mid k \in [i + N_G + 1,\; i + N_G + N_T] \}
$$

Side averages:
$$
Z_L = \frac{1}{N_T} \sum_{x[k] \in \mathcal{T}_L} x[k], \qquad
Z_R = \frac{1}{N_T} \sum_{x[k] \in \mathcal{T}_R} x[k]
$$

Noise estimate:
$$
\hat{Z}_i^{\text{GO}} = \max(Z_L, Z_R), \qquad
\hat{Z}_i^{\text{SO}} = \min(Z_L, Z_R)
$$

Threshold (approximate $\alpha$ using one side for calibration):
$$
T_i = \alpha \hat{Z}_i, \qquad
\alpha \approx N_T \left( P_{FA}^{-1/N_T} - 1 \right)
$$

### 3.1 1D GO/SO-CFAR: Python-Style Pseudocode

```python
def go_so_cfar_1d(x, num_guard, num_train, pfa, mode="GO"):
    """
    1D GO-CFAR / SO-CFAR.

    Args:
        x: 1D sequence
        num_guard: guard cells each side
        num_train: training cells each side
        pfa: desired false alarm probability
        mode: "GO" or "SO"

    Returns:
        detections, thresholds
    """
    L = len(x)
    N_side = num_train
    alpha = N_side * (pfa ** (-1.0 / N_side) - 1.0)

    detections = [False] * L
    thresholds = [0.0] * L

    start = num_guard + num_train
    end = L - (num_guard + num_train)

    for i in range(start, end):
        Lvals = x[i - num_guard - num_train : i - num_guard]
        Rvals = x[i + num_guard + 1 : i + num_guard + 1 + num_train]

        ZL = sum(Lvals) / len(Lvals)
        ZR = sum(Rvals) / len(Rvals)

        if mode.upper() == "GO":
            noise_est = max(ZL, ZR)
        elif mode.upper() == "SO":
            noise_est = min(ZL, ZR)
        else:
            raise ValueError("mode must be 'GO' or 'SO'")

        T = alpha * noise_est
        thresholds[i] = T
        detections[i] = (x[i] > T)

    return detections, thresholds
```

Note: 2D GO-/SO-CFAR variants exist (e.g., splitting windows into near/far or left/right halves) but are less standard. Adapt the same idea by computing $Z_1, Z_2$ over two subwindows and using $\max$ or $\min$.

---

## 4. OS-CFAR (Ordered Statistic CFAR)

OS-CFAR is robust in non-homogeneous clutter and multi-target scenarios. Instead of averaging, it uses an order statistic of the training samples.

For CUT at index $i$ with training set $\mathcal{T}_i$ of size $N$:

Sort the training magnitudes:
$$
z_{(1)} \le z_{(2)} \le \dots \le z_{(N)}
$$

Pick a rank $k$ (e.g., mid or high rank) and set
$$
\hat{Z}_i^{\text{OS}} = z_{(k)}
$$

Threshold:
$$
T_i = \alpha \hat{Z}_i^{\text{OS}}
$$

In practice $k$ and $\alpha$ are tuned (often via simulation) to hit the target $P_{FA}$. A common heuristic is $k \approx \lceil \rho N \rceil$ with $\rho \in [0.6, 0.9]$.

### 4.1 1D OS-CFAR: Python-Style Pseudocode

```python
def os_cfar_1d(x, num_guard, num_train, k_rank, alpha):
    """
    1D OS-CFAR.

    Args:
        x: 1D sequence
        num_guard: guard cells per side
        num_train: training cells per side
        k_rank: 1-based rank index into sorted training cells
        alpha: scale factor (tuned to get desired PFA)

    Returns:
        detections, thresholds
    """
    L = len(x)
    detections = [False] * L
    thresholds = [0.0] * L

    start = num_guard + num_train
    end = L - (num_guard + num_train)

    for i in range(start, end):
        left = x[i - num_guard - num_train : i - num_guard]
        right = x[i + num_guard + 1 : i + num_guard + 1 + num_train]
        train = sorted(list(left) + list(right))

        idx = max(0, min(len(train) - 1, k_rank - 1))
        noise_est = train[idx]

        T = alpha * noise_est
        thresholds[i] = T
        detections[i] = (x[i] > T)

    return detections, thresholds
```

### 4.2 2D OS-CFAR: Python-Style Pseudocode

Same principle as 1D, but $\mathcal{T}_{r,d}$ is the 2D training window (excluding guard + CUT) flattened to a list.

```python
def os_cfar_2d(X, num_guard_r, num_guard_d,
               num_train_r, num_train_d,
               k_rank, alpha):
    """
    2D OS-CFAR on a range-Doppler map.

    Args:
        X: 2D array-like, shape (R, D)
        num_guard_r: guard half-width (range)
        num_guard_d: guard half-width (Doppler)
        num_train_r: training half-width (range)
        num_train_d: training half-width (Doppler)
        k_rank: 1-based rank index
        alpha: scaling factor (tuned)

    Returns:
        detections, thresholds
    """
    R = len(X)
    D = len(X[0]) if R > 0 else 0

    Wr = num_guard_r + num_train_r
    Wd = num_guard_d + num_train_d

    detections = [[False for _ in range(D)] for _ in range(R)]
    thresholds = [[0.0 for _ in range(D)] for _ in range(R)]

    for r in range(Wr, R - Wr):
        for d in range(Wd, D - Wd):

            vals = []
            for rr in range(r - Wr, r + Wr + 1):
                for dd in range(d - Wd, d + Wd + 1):

                    # skip guard + CUT
                    if (abs(rr - r) <= num_guard_r and
                        abs(dd - d) <= num_guard_d):
                        continue

                    vals.append(X[rr][dd])

            vals_sorted = sorted(vals)
            idx = max(0, min(len(vals_sorted) - 1, k_rank - 1))
            noise_est = vals_sorted[idx]

            T = alpha * noise_est
            thresholds[r][d] = T
            detections[r][d] = (X[r][d] > T)

    return detections, thresholds
```

---

## 5. Practical Implementation Notes

- **Linear vs dB:** CFAR formulas assume linear power. Convert dB data to linear, run CFAR, then convert back if needed.
- **Edges:** The pseudocode skips indices where the full window does not fit. Alternatives: pad, mirror, or shrink windows near edges.
- **2D performance:** Nested loops are slow for large maps. Vectorize via convolution/integral images for CA-CFAR or use GPU (PyTorch/CuPy) for scale.
- **Parameter tuning:** Typical starts: $P_{FA} \in [10^{-2}, 10^{-5}]$, guard cells 1–4/side, training cells 8–32/side (1D). For OS-CFAR, pick $k$ around 60–80% of $N$ and tune $\alpha$ via Monte Carlo.
- **Sequential vs full 2D:** Applying 1D CFAR along range then Doppler is cheaper but approximate; full 2D CFAR is more robust but heavier.

---

## 6. Summary

- CA-CFAR: average of all training cells; optimal in homogeneous clutter.
- GO-CFAR: uses the larger side; good near clutter edges.
- SO-CFAR: uses the smaller side; helps near multiple targets.
- OS-CFAR: rank-ordered statistic; robust in non-homogeneous clutter.

---

## 7. Implementation Plan (Python, `mmwave_radar_processing/detectors`)

### 7.1 Base detector scaffolding
- Create `BaseCFAR1D` and `BaseCFAR2D` classes (assume magnitude input) handling window geometry, valid-region iteration (no padding/mirroring/shrinking), and storing per-cut thresholds/noise estimates.
- Expose a `detect(x, **params)` that returns detection index lists (1D: `[i]`, 2D: `[(r, d), ...]`); cache thresholds/noise maps on the instance for later inspection.
- Implement vectorized/sliding-window extraction: NumPy with `sliding_window_view` or SciPy convolution for sums; compute guard/training masks once per configuration to reuse across calls.
- Provide common helpers: input validation, guard/training size checks, PFA-to-alpha computation (shared formulas), conversion between threshold maps and index lists, and optional batch mode for 2D if trivially supported.
- Add optional plotting helpers (matplotlib, behind a separate `plot_detections`/`plot_thresholds` method) that operate on stored masks/thresholds without re-computation.

### 7.2 Child detector classes (compute thresholds/noise, reuse base)
- **CA-CFAR 1D/2D**: use mean of training cells; compute $\alpha$ from $P_{FA}$ and training count; thresholds via base helpers.
- **GO-CFAR 1D**: split left/right training, take `max`; $\alpha$ via single-side approximation; extend to 2D variant only if needed later.
- **SO-CFAR 1D**: split left/right training, take `min`; same $\alpha$ approach as GO; optional 2D variant similar to GO.
- **OS-CFAR 1D/2D**: sort training window (or use `np.partition` for efficiency) and pick rank $k$; apply supplied $\alpha$; reuse base windowing/mask logic.
- Each child defines `compute_thresholds(...)` (or `_estimate_noise(...)`) overriding a base abstract; base handles comparison and output formatting.

### 7.3 Examples in Readme/Docs
- Provide usage snippets in the docstring or README of the detectors module showing typical parameters (guard/train counts, $P_{FA}$) and how to retrieve detection indices and plots.\

### 7.4 Validation (planned, but not implemented)
- Add lightweight unit tests (pytest) for window sizing, valid-region masking, and numerical thresholds against small synthetic arrays.


# References (From GPT, need to check/confirm these)

## Classical CFAR Papers (checked)

- Finn, H.M. and Johnson, R.S., **"Adaptive Detection Mode with Threshold Control as a Function of Spatially Sampled Clutter-Level Estimates,"**  
  *IEEE Transactions on Aerospace and Electronic Systems*, 1968.

- Rohling, H., **"Radar CFAR Thresholding in Clutter and Multiple Target Situations,"**  
  *IEEE Transactions on Aerospace and Electronic Systems*, vol. AES-19, no. 4, pp. 608–621, 1983.  
  (Introduces GO-CFAR and SO-CFAR)

- Weiss, M., **"An Improved Detection Algorithm for Non-Homogeneous Clutter Environments,"**  
  *IEEE Transactions on Aerospace and Electronic Systems*, 1982.  
  (Original Ordered-Statistic CFAR)

- Gandhi, P.P. and Kassam, S.A., **"Analysis of CFAR Processors in Nonhomogeneous Background,"**  
  *IEEE Transactions on Aerospace and Electronic Systems*, vol. AES-24, no. 4, 1988.  
  (Seminal 2D CFAR analysis)

## Radar Signal Processing Textbooks (checked)

- Richards, M.A., **"Fundamentals of Radar Signal Processing,"**  
  2nd Edition, McGraw-Hill, 2014.  
  (Clear CFAR derivations + practical examples)

- Richards, M.A., Scheer, J.A., Holm, W.A. (eds.),  
  **"Principles of Modern Radar: Basic Principles,"**  
  SciTech Publishing, 2010.  
  (A standard reference; good chapters on detection theory)

- Skolnik, M., **"Introduction to Radar Systems,"**  
  McGraw-Hill, multiple editions.  
  (Classic; CFAR appears in detection chapters)

## Modern Tutorials and Application Papers (to check)

- Rohling, H., **"Radar CFAR Thresholding for Automotive Radar,"**  
  *IEEE AES Systems Magazine*, 2011.  
  (Useful for FMCW + automotive radar systems)

- Kay, S.M. and Marple, S.L., **"Spectrum Analysis—A Modern Perspective,"**  
  (Contains useful sections on detection and CFAR in spectral estimation contexts)

- Fa, G. and Kassam, S., **"CFAR Detection in Nonhomogeneous Backgrounds,"**  
  *IEEE Transactions on Aerospace and Electronic Systems*, 1984.  
  (Deep analysis of CFAR challenges)


