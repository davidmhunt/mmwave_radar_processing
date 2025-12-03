
import numpy as np

def reproduce():
    val1 = 10.0
    val2 = 5.0
    
    print(f"Testing np.min({val1}, {val2})...")
    try:
        # This is the suspected buggy call: np.min(float, float)
        # The second argument is interpreted as 'axis', which expects an integer.
        result = np.min(val1, val2)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    reproduce()
