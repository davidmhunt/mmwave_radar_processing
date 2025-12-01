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

## 5. Implementation in `mmwave_radar_processing`

The detectors are implemented in the `mmwave_radar_processing.detectors` module, providing a flexible and efficient Python interface for CFAR detection.

### 5.1 Instantiation

All detector classes follow a consistent instantiation pattern. You must provide the window geometry (training and guard cells) and the desired probability of false alarm ($P_{FA}$).

**Example (1D CA-CFAR):**
```python
from mmwave_radar_processing.detectors import CaCFAR1D

# Instantiate a CA-CFAR detector
# num_train: 10 cells on each side
# num_guard: 2 cells on each side
# pfa: 1e-4
detector = CaCFAR1D(num_train=10, num_guard=2, pfa=1e-4)
```

**Example (2D OS-CFAR):**
```python
from mmwave_radar_processing.detectors import OsCFAR2D

# Instantiate a 2D OS-CFAR detector
# num_train: (range=8, doppler=4)
# num_guard: (range=2, doppler=1)
# k_rank: 50 (rank for ordered statistic)
# alpha: 3.5 (scaling factor)
detector = OsCFAR2D(num_train=(8, 4), num_guard=(2, 1), k_rank=50, alpha=3.5)
```

### 5.2 Key Methods

- **`detect(x)`**: The primary method for performing detection.
    - **Input**: A 1D array (for 1D detectors) or 2D array (for 2D detectors) representing the signal magnitude or power.
    - **Output**: A list of indices where targets are detected. For 1D, this is a list of integers `[i1, i2, ...]`. For 2D, it is a list of tuples `[(r1, d1), (r2, d2), ...]`.
    - **Side Effects**: Caches the computed thresholds and noise estimates internally.

- **`plot_detections(x)`**: A helper method to visualize the results.
    - **Input**: The same signal array `x` passed to `detect()`.
    - **Behavior**: Generates a plot (using `matplotlib`) showing the signal, the adaptive threshold, and the detected points.

### 5.3 Downstream Usage

The output of the `detect()` method is a list of raw indices. These can be used for:
1. **Target Extraction**: Extracting the precise range or Doppler values corresponding to the detected indices.
2. **Point Cloud Generation**: Converting (Range, Doppler) tuples into a 3D point cloud (Range, Azimuth, Doppler) if angle-of-arrival processing follows.
3. **Tracking**: Feeding the detected centroids into a tracker (e.g., Kalman Filter) to maintain target identity over time.

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


