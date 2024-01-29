from typing import Tuple
import numpy as np 
from utils import softmax, dot_product, triangular_mask

class SelfAttention:
    def __init__(self, L: int, d_k: int, d_v: int):
        """
        Implements the SelfAttention class which calculates the scaled dot product self attention
        Inputs
            L : len of the input sequence
            d_k  : len of the key vector
            d_v  : len of the value vector
        """
        self.L = L
        self.d_k = d_k
        self.d_q = d_k
        self.d_v = d_v

        self.Q = np.random.randn(self.L, self.d_q)
        self.K = np.random.randn(self.L, self.d_k)
        self.V = np.random.randn(self.L, self.d_v)
    
    def apply(self, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of scaled dot product self attention

        """
        scaled_dot_pdt = dot_product(self.Q, self.K.T) / np.sqrt(self.d_k)
        if mask is not None:
            scaled_dot_pdt += mask
        attention = softmax(scaled_dot_pdt)
        out = dot_product(attention, self.V)
        return out , attention

if __name__ == "__main__":
    L = 4
    d_k, d_v = 3, 2
    out1, attention1 = SelfAttention(L, d_k, d_v).apply()

    #with mask (usually used for decoders)
    mask = triangular_mask(L, L) 
    out2, attention2 = SelfAttention(L, d_k, d_v).apply(mask)
    import pdb;pdb.set_trace()