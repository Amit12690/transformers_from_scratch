import torch 
import torch.nn.functional as F
from typing import Optional
import math

class MultiHeadAttention:
    def __init__(self, input_seq_len: int, d_in: int, d_model: int, num_heads: int, batch_size: int=1):
        """
        input_seq_len : Input sequence length
        d_in          : Input embeddings size
        d_model       : Size of key matrix combined across all heads. Same size holds for query and value matrices as well
        num_heads     : Number of attention heads
        batch_size    : batch_size
        """
        self.input_seq_len = input_seq_len
        self.d_in = d_in
        self.d_model = d_model
        self.num_heads = num_heads
        self.batch_size = batch_size

    def forward(self, input_emb: torch.Tensor,  mask: bool) -> torch.Tensor:
        qkv_layer = torch.nn.Linear(self.d_in, self.d_model*3)
        
        # Dims of qkv combined projection [bs, input_seq_len, self.d_model*3]
        qkv_combined = qkv_layer(input_emb)

        # Split the projections based on num_heads
        dim_per_head = self.d_model//self.num_heads
        
        # Dims after splitting based on num_heads =  [bs, input_seq_len, num_heads, dim_per_head*3]
        qkv_combined = qkv_combined.reshape(self.batch_size, self.input_seq_len, self.num_heads, dim_per_head*3)

        # Reorganising the dims to [bs, num_heads, input_seq_len, dim_per_head*3]]
        qkv_combined = qkv_combined.permute(0, 2, 1, 3)

        # Split in to q, k, v tensors. Each tensor dims [bs, num_heads, input_seq_len, dim_per_head]
        query, key , value = qkv_combined.chunk(3, dim=-1) 

        # Scaled dot product : dims = [bs, num_heads, input_seq_len, input_seq_len]
        scaled_dot_pdt = torch.matmul(query, key.transpose(-1, -2))
        if mask:
            mask = torch.full(scaled_dot_pdt.size(), fill_value=float("-inf")) #-inf because , -inf results in zero when we apply softmax
            mask = torch.triu(mask, diagonal = 1)
            scaled_dot_pdt = scaled_dot_pdt + mask
        scaled_dot_pdt = scaled_dot_pdt/math.sqrt(dim_per_head)

        # Take softmax to calculate attention whose dims are [bs, num_heads, input_seq_len, input_seq_len]
        attention = F.softmax(scaled_dot_pdt, dim=-1)

        # Transform the value using attention . dims will be same as dimension of value :  [bs, num_heads, input_seq_len, dim_per_head]
        transformed_value = torch.matmul(attention, value)

        return attention, transformed_value 
        

if __name__ == "__main__":
    input_seq_len = 4
    d_in = 512
    d_model = 512
    num_heads = 8
    batch_size = 1

    attn_layer = MultiHeadAttention(input_seq_len, d_in, d_model, num_heads, batch_size)

    inp_embedding = torch.randn((batch_size, input_seq_len, d_in))

    attention1, transformed_value1  = attn_layer.forward(inp_embedding, mask=False)
    attention2, transformed_value2  = attn_layer.forward(inp_embedding, mask=True)

    print(attention1[0,0], transformed_value1[0, 0, 0])
    print(attention2[0,0], transformed_value2[0, 0, 0])    

    #import pdb;pdb.set_trace()







