# CFAR Detector Guide (`detectors_full.md`)

Implementation-focused guide for:

- 1D & 2D **CA-CFAR** (Cell Averaging)
- 1D & 2D **GO-CFAR** (Greatest-Of)
- 1D & 2D **SO-CFAR** (Smallest-Of)
- 1D & 2D **OS-CFAR** (Ordered Statistics)

All equations are written in LaTeX math syntax (`$...$`, `$$...$$`) so math-aware renderers can display them nicely.

---

## 1. Notation and Setup

We assume a post-detection radar magnitude (or power) domain.

- 1D signal (range profile or Doppler cut):  
  - $x[i]$: value at index $i$, $i = 0, \dots, L-1$.
- 2D signal (range–Doppler map):  
  - $X[r, d]$: value at range bin $r$ and Doppler bin $d$, $r = 0,\dots,R-1$, $d = 0,\dots,D-1$.
- CUT (Cell Under Test): the sample being tested for a target.
- $N_G$: number of **guard cells per side** (1D) or half-width in each dimension (2D).
- $N_T$: number of **training cells per side** (1D) or half-width (2D).
- $P_{FA}$: desired probability of false alarm.
- $\alpha$: CFAR scaling factor.

Assumption (classical CFAR model):
- Noise after envelope detection is **exponential** (i.i.d. in training cells).
- Under $H_0$ (no target), CUT distribution matches training cells.

---

## 2. CA-CFAR (Cell Averaging CFAR)

### 2.1 1D CA-CFAR: Math

For CUT at index $i$:

- Left training region:
  $$
  \mathcal{T}_L =
  \{ x[k] \mid k \in [i - N_G - N_T,\; i - N_G - 1] \}
  $$
- Right training region:
  $$
  \mathcal{T}_R =
  \{ x[k] \mid k \in [i + N_G + 1,\; i + N_G + N_T] \}
  $$

Total training set:
$$
\mathcal{T}_i = \mathcal{T}_L \cup \mathcal{T}_R
$$

Number of training cells:
$$
N = 2 N_T
$$

Noise estimate (cell averaging):
$$
\hat{Z}_i = \frac{1}{N} \sum_{x[k] \in \mathcal{T}_i} x[k]
$$

Threshold:
$$
T_i = \alpha \hat{Z}_i
$$

To achieve a given $P_{FA}$ with exponential noise and CA-CFAR, the scaling factor is:
$$
\alpha = N \left( P_{FA}^{-1/N} - 1 \right)
$$

Decision rule:
$$
x[i] > T_i \Rightarrow \text{declare target at } i
$$

---

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
2.3 2D CA-CFAR: Math
We now consider a 2D map $X[r,d]$.

Define:

Range half-window: $W_r = N_{G,r} + N_{T,r}$

Doppler half-window: $W_d = N_{G,d} + N_{T,d}$

Total CFAR window size:

𝑁
win
=
(
2
𝑊
𝑟
+
1
)
(
2
𝑊
𝑑
+
1
)
N 
win
​
 =(2W 
r
​
 +1)(2W 
d
​
 +1)
Guard + CUT region size:

𝑁
guard
=
(
2
𝑁
𝐺
,
𝑟
+
1
)
(
2
𝑁
𝐺
,
𝑑
+
1
)
N 
guard
​
 =(2N 
G,r
​
 +1)(2N 
G,d
​
 +1)
Thus number of training cells:

𝑁
=
𝑁
win
−
𝑁
guard
N=N 
win
​
 −N 
guard
​
 
For a CUT at $(r, d)$, the training set is:

All cells in the rectangle:
$[r - W_r, r + W_r] \times [d - W_d, d + W_d]$

excluding the guard + CUT rectangle:
$[r - N_{G,r}, r + N_{G,r}] \times [d - N_{G,d}, d + N_{G,d}]$

Noise estimate:

𝑍
^
𝑟
,
𝑑
=
1
𝑁
∑
(
𝑟
′
,
𝑑
′
)
∈
𝑇
𝑟
,
𝑑
𝑋
[
𝑟
′
,
𝑑
′
]
Z
^
  
r,d
​
 = 
N
1
​
  
(r 
′
 ,d 
′
 )∈T 
r,d
​
 
∑
​
 X[r 
′
 ,d 
′
 ]
Threshold:

𝑇
𝑟
,
𝑑
=
𝛼
𝑍
^
𝑟
,
𝑑
T 
r,d
​
 =α 
Z
^
  
r,d
​
 
Scaling factor (same CA-CFAR formula):

𝛼
=
𝑁
(
𝑃
𝐹
𝐴
−
1
/
𝑁
−
1
)
α=N(P 
FA
−1/N
​
 −1)
2.4 2D CA-CFAR: Python-Style Pseudocode
python
Copy code
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

    # Iterate only where full window fits
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
3. GO-CFAR and SO-CFAR (1D)
GO-CFAR and SO-CFAR improve robustness near clutter edges and strong interferers by using two separate training regions and combining them with max or min.

For a CUT at index $i$:

Leading (left) training:

𝑇
𝐿
=
𝑥
[
𝑖
−
𝑁
𝐺
−
𝑁
𝑇
:
𝑖
−
𝑁
𝐺
]
T 
L
​
 =x[i−N 
G
​
 −N 
T
​
 :i−N 
G
​
 ]
Lagging (right) training:

𝑇
𝑅
=
𝑥
[
𝑖
+
𝑁
𝐺
+
1
:
𝑖
+
𝑁
𝐺
+
1
+
𝑁
𝑇
]
T 
R
​
 =x[i+N 
G
​
 +1:i+N 
G
​
 +1+N 
T
​
 ]
Averages:

𝑍
𝐿
=
1
𝑁
𝑇
∑
𝑇
𝐿
,
𝑍
𝑅
=
1
𝑁
𝑇
∑
𝑇
𝑅
Z 
L
​
 = 
N 
T
​
 
1
​
 ∑T 
L
​
 ,Z 
R
​
 = 
N 
T
​
 
1
​
 ∑T 
R
​
 
GO-CFAR (Greatest-Of):

𝑍
^
𝑖
GO
=
max
⁡
(
𝑍
𝐿
,
𝑍
𝑅
)
Z
^
  
i
GO
​
 =max(Z 
L
​
 ,Z 
R
​
 )
SO-CFAR (Smallest-Of):

𝑍
^
𝑖
SO
=
min
⁡
(
𝑍
𝐿
,
𝑍
𝑅
)
Z
^
  
i
SO
​
 =min(Z 
L
​
 ,Z 
R
​
 )
Threshold:

𝑇
𝑖
=
𝛼
𝑍
^
𝑖
T 
i
​
 =α 
Z
^
  
i
​
 
In theory, $\alpha$ depends on the effective distribution (since you are using max or min of two estimates). In practice, a common approximation is to treat one side as representative for the $P_{FA}$ calibration:

𝛼
≈
𝑁
𝑇
(
𝑃
𝐹
𝐴
−
1
/
𝑁
𝑇
−
1
)
α≈N 
T
​
 (P 
FA
−1/N 
T
​
 
​
 −1)
and then tune if needed.

3.1 1D GO/SO-CFAR: Python-Style Pseudocode
python
Copy code
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
Note: 2D GO-/SO-CFAR variants exist (e.g., splitting windows into near/far or left/right halves), but are less standard. You can adapt the same idea by computing $Z_1, Z_2$ over two subwindows and using $\max$ or $\min$.

4. OS-CFAR (Ordered Statistic CFAR)
OS-CFAR is more robust in non-homogeneous clutter and multi-target scenarios. Instead of averaging, it uses an order statistic of the training samples.

For CUT at index $i$, with training set $\mathcal{T}_i$ of size $N$:

Sort training magnitudes:

𝑧
(
1
)
≤
𝑧
(
2
)
≤
⋯
≤
𝑧
(
𝑁
)
z 
(1)
​
 ≤z 
(2)
​
 ≤⋯≤z 
(N)
​
 
Pick a rank index $k$ (e.g., mid or high rank).

Use:

𝑍
^
𝑖
OS
=
𝑧
(
𝑘
)
Z
^
  
i
OS
​
 =z 
(k)
​
 
Threshold:

𝑇
𝑖
=
𝛼
𝑍
^
𝑖
OS
T 
i
​
 =α 
Z
^
  
i
OS
​
 
In practice:

$k$ and $\alpha$ are typically tuned via simulation to achieve a target $P_{FA}$.

A common heuristic is $k \approx \lceil \rho N \rceil$ with $\rho \in [0.6, 0.9]$.

4.1 1D OS-CFAR: Python-Style Pseudocode
python
Copy code
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
4.2 2D OS-CFAR: Python-Style Pseudocode
Same principle, but $\mathcal{T}_{r,d}$ is the 2D training window (excluding guard + CUT), flattened to a list.

python
Copy code
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
5. Practical Implementation Notes
Linear vs dB:

CFAR formulas assume linear power.

If your data is in dB: convert to linear, run CFAR, optionally convert results back.

Edges:

The pseudocode simply skips indices where the full window does not fit.

Alternatives: pad, mirror, or use smaller windows near edges.

2D Performance:

2D nested loops can be slow for large RD maps.

Possible optimizations:

Vectorized sums via convolution or integral images (for CA-CFAR).

GPU (PyTorch / CuPy) for large-scale processing.

Parameter Tuning:

Typical starting points:

$P_{FA} \in [10^{-2}, 10^{-5}]$

Guard cells: 1–4 per side

Training cells: 8–32 per side (1D); smaller half-widths in 2D but many total cells.

For OS-CFAR:

Choose $k$ around 60–80% of the sorted list.

Tune $\alpha$ empirically via Monte Carlo.

Sequential vs Full 2D CFAR:

You can apply CFAR along range and then along Doppler (1D+1D) as an approximation.

Full 2D CFAR is more robust but more expensive.

6. Summary
CA-CFAR: uses average of all training cells; optimal in homogeneous clutter.

GO-CFAR: uses the larger side; good near clutter edges.

SO-CFAR: uses the smaller side; can help in multi-target scenarios.

OS-CFAR: uses a rank-ordered statistic; robust in non-homogeneous clutter.