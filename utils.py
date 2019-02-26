import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch.nn.modules.batchnorm import _BatchNorm

import numpy as np
import pdb
import os


# ------------------------------------------------------------------------------
# Utility Methods
# ------------------------------------------------------------------------------

Log2PI = float(np.log(2 * np.pi))

def flatten_sum(logps):
    #while len(logps.size()) > 1: 
    #    logps = logps.sum(dim=-1)
    logps = logps.view(logps.shape[0], -1).sum(-1)
    return logps


# ------------------------------------------------------------------------------
# Distributions
# ------------------------------------------------------------------------------

def standard_normal(shape, device='cuda'):
    class Normal(object):
        def logps(x):
            return -0.5 * (Log2PI + x**2)

        def sample():
            s = torch.empty(shape, device=device).normal_(mean=0, std=1)

        def logp(x):
            flatten_sum(self.logps(x))

    return Normal


def standard_gaussian(shape):
    mean, logsd = [torch.cuda.FloatTensor(shape).fill_(0.) for _ in range(2)]
    return gaussian_diag(mean, logsd)


def gaussian_diag(mean, logsd):
    class o(object):
        Log2PI = float(np.log(2 * np.pi))
        pass

        def logps(x):
            return  -0.5 * (o.Log2PI + 2. * logsd + ((x - mean) ** 2) / torch.exp(2. * logsd))

        def sample():
            eps = torch.zeros_like(mean).normal_()
            return mean + torch.exp(logsd) * eps

    o.logp    = lambda x: flatten_sum(o.logps(x))
    return o


