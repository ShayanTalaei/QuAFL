import torch
import numpy.linalg as LA

class Quantizer:
    
    def encode(self, x):
        pass
    
    def decode(self, x, key = None):
        pass
    
class Identity_Quantizer(Quantizer):

    def encode(self, x):
        return x

    def decode(self, x, key):
        return x
         
class Lattice_Quantizer(Quantizer):

    def __init__(self, q, s):
        self.q = q  ## quantization level
        self.s = s  ## hypercube side length

    def encode(self, input_vec):
        scaled_input = input_vec / self.s ## devide by s
        scaled_input = torch.round(scaled_input).type(torch.int32) ## make it integer
        encoded_vector = torch.remainder(scaled_input, self.q) ## mod q
        return encoded_vector

    def decode(self, quantized_vector, key):
        part1 = self.q * self.s * torch.round((key / (self.q * self.s)) - (quantized_vector / self.q))
        part2 = self.s * quantized_vector
        decoded_vec = part1 + part2
        return decoded_vec
    
class QSGD_Quantizer(Quantizer):
    
    def __init__(self, k, L2 = False):
        self.q_levels = k
        self.L2 = L2
        
    def encode(self, x):
        fmin = 0 if self.L2 else x.min() 
        fmax = LA.norm(x) if self.L2 else x.max()
        if fmax - fmin == 0:
            return fmin * torch.ones_like(x)
        
        unit = (fmax - fmin) / (self.q_levels - 1)
        v = torch.floor((x - fmin) / unit + torch.rand_like(x))#.cuda()
        res = fmin + v * unit
        return res
    
    def decode(self, x, key = None):
        return x
    
def get_quantizer(**kwargs):
    quantization_method = kwargs["method"]
    if quantization_method == "identity":
        quantizer = Identity_Quantizer()
    elif quantization_method == 'lattice':
        quant_q = kwargs['quant_q']
        quant_s = kwargs['quant_s']
        quantizer = Lattice_Quantizer(2 ** quant_q, quant_s)
    elif quantization_method == 'qsgd':
        k = kwargs['k']
        L2 = False
        if "L2" in kwargs.keys():
            L2 = kwargs['L2']
        quantizer = QSGD_Quantizer(k, L2)
    return quantizer
