import torch

def quantize_torch_int8(val, reduce_axis):
    # val is (batch, heads, seqlen, dim)
    scale = torch.amax(val.abs(), axis=reduce_axis, keepdim=True)
    scale = scale / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (val / scale).to(torch.int8), scale


def dequantize_torch_int8(val, scale):
    return val * scale