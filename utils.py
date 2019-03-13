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
    logps = logps.view(logps.shape[0], -1).sum(-1)
    return logps


@torch.no_grad()
def onehot(x, n):
    v = torch.zeros(x.size()[0], n, device=x.device)
    v.scatter_(1, x.view(-1, 1), 1)
    return v

# ------------------------------------------------------------------------------
# Distributions
# ------------------------------------------------------------------------------

@torch.no_grad()
def standard_normal_logp(x):
    '''Calculate probability of x under a zero mean, identity covariance 
    multivariate normal distribution'''
    return flatten_sum(-0.5 * (Log2PI + x**2))


@torch.no_grad()
def standard_normal_sample(shape, std=1, device='cuda'):
    '''Sample from a multivariate normal distribution with zero mean and
    identity covariance'''
    s = torch.empty(shape, device=device).normal_(mean=0, std=std)
    return s


@torch.no_grad()
def gaussian_shift_logp(x, mean):
    '''Calculate the probability of x under a multivariate normal with
    mean `mean` and identity covariance'''
    return flatten_sum(-0.5 * (Log2PI + ((x - mean)**2)))


@torch.no_grad()
def gaussian_shift_sample(mean, std=1):
    '''Sample from  a multivariate normal with mean `mean` and identity
    covariance'''
    s = torch.empty_like(mean)
    s = torch.normal(mean=mean, std=std, out=s)
    return s


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


