

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

def calcute_lags(x_enc, k=5):
    q_fft = torch.fft.rfft(x_enc, dim=-1)
    k_fft = torch.fft.rfft(x_enc, dim=-1)
    res = q_fft * torch.conj(k_fft)
    corr = torch.fft.irfft(res, dim=-1)
    mean_value = torch.mean(corr, dim=1)
    _, lags = torch.topk(mean_value, k, dim=-1)
    return lags

