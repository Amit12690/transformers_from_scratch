import numpy as np 

def softmax(x: np.ndarray) -> np.ndarray:
    den_sum = np.sum(np.exp(x))
    return np.array([np.exp(xi)/den_sum for xi in x])

def dot_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b)

def triangular_mask(r: int, c: int) -> np.ndarray:
    """
    Used to prevent a position from attending to subsequent positions (future positions) in the sequence. Usually used in decoder
    In the context of language modeling or sequence-to-sequence tasks, you want each position to attend to positions that come before it, but not after it during training.
    The future mask is applied by setting the attention scores for future positions to a large negative value or zero during the softmax operation.

    Example of a future mask : 
    array([[0, -1.e+09, -1.e+09],
           [0,  0,      -1.e+09],
           [0,  0,       0]])
    """
    huge_neg_val = -1e9 
    future_mask = (np.tri(r, c) == 0) * huge_neg_val
    return future_mask

