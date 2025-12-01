
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmwave_radar_processing.detectors import (
    CaCFAR1D, CaCFAR2D,
    GoCFAR1D, SoCFAR1D,
    OsCFAR1D, OsCFAR2D
)

def verify_1d():
    print("Verifying 1D Detectors...")
    # Create synthetic signal: Noise + Target
    np.random.seed(42)
    L = 100
    noise = np.random.exponential(scale=1.0, size=L)
    x = noise.copy()
    x[50] = 10.0 # Strong target
    
    # Params
    num_train = 10
    num_guard = 2
    pfa = 1e-3
    
    # CA-CFAR
    print("  Testing CA-CFAR 1D...")
    ca = CaCFAR1D(num_train, num_guard, pfa)
    dets = ca.detect(x)
    print(f"    Detections: {dets}")
    assert 50 in dets, "CA-CFAR failed to detect target at 50"
    
    # GO-CFAR
    print("  Testing GO-CFAR 1D...")
    go = GoCFAR1D(num_train, num_guard, pfa)
    dets = go.detect(x)
    print(f"    Detections: {dets}")
    assert 50 in dets, "GO-CFAR failed to detect target at 50"
    
    # SO-CFAR
    print("  Testing SO-CFAR 1D...")
    so = SoCFAR1D(num_train, num_guard, pfa)
    dets = so.detect(x)
    print(f"    Detections: {dets}")
    assert 50 in dets, "SO-CFAR failed to detect target at 50"
    
    # OS-CFAR
    print("  Testing OS-CFAR 1D...")
    # k = 3/4 of 2*N_T = 15
    k_rank = 15
    alpha = 5.0 # Arbitrary alpha for test
    os_det = OsCFAR1D(num_train, num_guard, k_rank, alpha)
    dets = os_det.detect(x)
    print(f"    Detections: {dets}")
    assert 50 in dets, "OS-CFAR failed to detect target at 50"

def verify_2d():
    print("\nVerifying 2D Detectors...")
    # Create synthetic signal
    R, D = 50, 50
    noise = np.random.exponential(scale=1.0, size=(R, D))
    X = noise.copy()
    X[25, 25] = 15.0 # Target
    
    # Params
    num_train = (5, 5)
    num_guard = (2, 2)
    pfa = 1e-4
    
    # CA-CFAR 2D
    print("  Testing CA-CFAR 2D...")
    ca = CaCFAR2D(num_train, num_guard, pfa)
    dets = ca.detect(X)
    # dets is list of tuples
    print(f"    Detections count: {len(dets)}")
    assert (25, 25) in dets, "CA-CFAR 2D failed to detect target at (25, 25)"
    
    # OS-CFAR 2D
    print("  Testing OS-CFAR 2D...")
    # Total train cells = (2*5+1)*(2*5+1) - (2*2+1)*(2*2+1) = 11*11 - 5*5 = 121 - 25 = 96
    k_rank = 80
    alpha = 5.0
    os_det = OsCFAR2D(num_train, num_guard, k_rank, alpha)
    dets = os_det.detect(X)
    print(f"    Detections count: {len(dets)}")
    assert (25, 25) in dets, "OS-CFAR 2D failed to detect target at (25, 25)"

if __name__ == "__main__":
    verify_1d()
    verify_2d()
    print("\nAll verifications passed!")
