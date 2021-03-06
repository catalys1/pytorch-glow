import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn

import numpy as np
import pdb

from .layers import * 
from .utils import * 

# ------------------------------------------------------------------------------
# Abstract Classes to define common interface for invertible functions
# ------------------------------------------------------------------------------

# Abstract Class for bijective functions
class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward_(self, x, objective, y=None, **kwargs):
        raise NotImplementedError

    def reverse_(self, x, objective, y=None, **kwargs):
        raise NotImplementedError

# Wrapper for stacking multiple layers 
class LayerList(Layer):
    def __init__(self, list_of_layers=None):
        super(LayerList, self).__init__()
        self.layers = nn.ModuleList(list_of_layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward_(self, x, objective, y=None, **kwargs):
        for layer in self.layers:
            x, objective = layer.forward_(x, objective, y=y, **kwargs)
        return x, objective

    def reverse_(self, x, objective, y=None, **kwargs):
        for layer in reversed(self.layers): 
            x, objective = layer.reverse_(x, objective, y=y, **kwargs)
        return x, objective


# ------------------------------------------------------------------------------
# Permutation Layers 
# ------------------------------------------------------------------------------

# Invertible 1x1 convolution
class Invertible1x1Conv(Layer, nn.Conv2d):
    def __init__(self, num_channels):
        self.num_channels = num_channels
        nn.Conv2d.__init__(self, num_channels, num_channels, 1, bias=False)

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = np.linalg.qr(np.random.randn(self.num_channels, self.num_channels))[0]
        w_init = torch.from_numpy(w_init.astype('float32'))
        w_init = w_init.unsqueeze(-1).unsqueeze(-1)
        self.weight.data.copy_(w_init)

    def forward_(self, x, objective, y=None, **kwargs):
        dlogdet = torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1) 
        objective += dlogdet
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, \
            self.dilation, self.groups)
 
        return output, objective

    def reverse_(self, x, objective, y=None, **kwargs):
        dlogdet = torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1) 
        objective -= dlogdet
        weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
        output = F.conv2d(x, weight_inv, self.bias, self.stride, self.padding, \
                    self.dilation, self.groups)
        return output, objective


# ------------------------------------------------------------------------------
# Layers involving squeeze operations defined in RealNVP / Glow. 
# ------------------------------------------------------------------------------

# Trades space for depth and vice versa
class Squeeze(Layer):
    def __init__(self, input_shape, factor=2):
        super(Squeeze, self).__init__()
        assert factor > 1 and isinstance(factor, int), 'no point of using this if factor <= 1'
        self.factor = factor
        self.input_shape = input_shape

    def squeeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert h % self.factor == 0 and w % self.factor == 0, pdb.set_trace()
        
        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c, h // self.factor, self.factor, w // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(bs, c * self.factor * self.factor, h // self.factor, w // self.factor)

        return x
 
    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // self.factor ** 2, self.factor, self.factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x
    
    def forward_(self, x, objective, y=None, **kwargs):
        if len(x.size()) != 4: 
            raise NotImplementedError # Maybe ValueError would be more appropriate

        return self.squeeze_bchw(x), objective
        
    def reverse_(self, x, objective, y=None, **kwargs):
        if len(x.size()) != 4: 
            raise NotImplementedError

        return self.unsqueeze_bchw(x), objective


# ------------------------------------------------------------------------------
# Layers involving prior
# ------------------------------------------------------------------------------

# Split Layer for multi-scale architecture. Factor of 2 hardcoded.
class Split(Layer):
    def __init__(self, input_shape):
        super(Split, self).__init__()
        bs, c, h, w = input_shape
        self.conv_zero = Conv2dZeroInit(c // 2, c, 3, padding=(3 - 1) // 2)

    def split2d_prior(self, x, y=None, **kwargs):
        h = self.conv_zero(x)
        mean, logs = h[:, 0::2], h[:, 1::2]
        return gaussian_diag(mean, logs)

    def forward_(self, x, objective, y=None, **kwargs):
        bs, c, h, w = x.size()
        z1, z2 = torch.chunk(x, 2, dim=1)
        objective += standard_normal_logp(z2)
        self.sample = z2
        return z1, objective

    def reverse_(self, x, objective, y=None, **kwargs):
        std = kwargs.get('std', 1)
        use_stored_sample = kwargs.get('use_stored_sample', False)
        if use_stored_sample:
            z2 = self.sample
        else:
            z2 = standard_normal_sample(x.size(), std=std, device=x.device)
        z = torch.cat([x, z2], dim=1)
        objective -= standard_normal_logp(z2)
        return z, objective

# Gaussian Prior that's compatible with the Layer framework
class GaussianPrior(Layer):
    def __init__(self, input_shape, ydim=None):
        super(GaussianPrior, self).__init__()
        self.input_shape = input_shape
        self.ydim = ydim

        if ydim is not None:
            c, h, w = self.input_shape[1:]
            self.mean_select = LinearZeroInit(ydim, c * h * w)

    def forward_(self, x, objective, y=None, **kwargs):
        if y is not None:
            y = onehot(y, self.ydim)
            mean = self.mean_select(y).view(-1, *self.input_shape[1:])
            objective += gaussian_shift_logp(x, mean)
        else:
            objective += standard_normal_logp(x)

        # this way, you can encode and decode back the same image. 
        return x, objective

    def reverse_(self, x, objective, y=None, **kwargs):
        std = kwargs.get('std', 1)
        if y is not None:
            y = onehot(y, self.ydim)
            mean = self.mean_select(y).view(-1, *self.input_shape[1:])
            z = gaussian_shift_sample(mean, std=std)
            objective -= gaussian_shift_logp(z, mean)
        elif x is None:
            z = standard_normal_sample(self.input_shape, std=std)
            objective -= standard_normal_logp(z)
        else:
            z = x
            objective -= standard_normal_logp(z)

        # this way, you can encode and decode back the same image. 
        return z, objective
         

# ------------------------------------------------------------------------------
# Coupling Layers
# ------------------------------------------------------------------------------

# Affine Coupling Layer
class AffineCoupling(Layer):
    def __init__(self, num_features):
        super(AffineCoupling, self).__init__()
        # assert num_features % 2 == 0
        self.NN = NN(num_features // 2, channels_out=num_features)

    def forward_(self, x, objective, y=None, **kwargs):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = torch.sigmoid(h[:, 1::2] + 2.)
        z2 += shift
        z2 *= scale
        objective += flatten_sum(torch.log(scale))

        return torch.cat([z1, z2], dim=1), objective

    def reverse_(self, x, objective, y=None, **kwargs):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = torch.sigmoid(h[:, 1::2] + 2.)
        z2 /= scale
        z2 -= shift
        objective -= flatten_sum(torch.log(scale))
        return torch.cat([z1, z2], dim=1), objective


# ------------------------------------------------------------------------------
# Normalizing Layers
# ------------------------------------------------------------------------------

# ActNorm Layer with data-dependant init
class ActNorm(Layer):
    def __init__(self, num_features, logscale_factor=1., scale=1.):
        super(Layer, self).__init__()
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_buffer('initialized', torch.ByteTensor([0]))

    def forward_(self, input, objective, y=None, **kwargs):
        input_shape = input.size()
        input = input.view(*input.shape[:2], -1)

        if not self.initialized[0]: 
            with torch.no_grad():
                self.initialized[0] = 1
                t = True
                b = input.mean(0, keepdim=t).mean(-1, keepdim=t)
                vars = ((input - b)**2).mean(0, keepdim=t).mean(-1, keepdim=t)
                logs = torch.log(self.scale / torch.sqrt(vars) + 1e-6)
                logs[vars == 0] = 0
                logs = logs / self.logscale_factor 
          
            self.b.data.copy_(-b.data)  # negative so it centers initially
            self.logs.data.copy_(logs.data)

        logs = self.logs * self.logscale_factor
        b = self.b
        
        output = (input + b) * torch.exp(logs)
        dlogdet = torch.sum(logs) * input.size(-1)

        return output.view(input_shape), objective + dlogdet

    def reverse_(self, input, objective, y=None, **kwargs):
        assert self.initialized[0] == 1
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        output = input * torch.exp(-logs) - b
        dlogdet = torch.sum(logs) * input.size(-1)

        return output.view(input_shape), objective - dlogdet

# (Note: a BatchNorm layer can be found in previous commits)


# ------------------------------------------------------------------------------
# Stacked Layers
# ------------------------------------------------------------------------------

# 1 step of the flow (see Figure 2 a) in the original paper)
class RevNetStep(LayerList):
    def __init__(self, num_channels, norm):
        super(RevNetStep, self).__init__()
        layers = []
        if norm == 'actnorm': 
            layers += [ActNorm(num_channels)]
        else: 
            assert not norm               
 
        layers += [Invertible1x1Conv(num_channels)]
        layers += [AffineCoupling(num_channels)]

        self.layers = nn.ModuleList(layers)


class ClassPredict(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(ClassPredict, self).__init__()
        self.linear = nn.Linear(in_channels, out_dim)
        torch.nn.init.normal_(self.linear.weight, std=0.03)

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


# Full model
class Glow_(LayerList, nn.Module):
    def __init__(self, input_shape, n_levels=4, depth=4, batch_size=32,
                 ydim=None, norm='actnorm'):
        super(Glow_, self).__init__()
        layers = []
        output_shapes = []
        _, C, H, W = input_shape
        
        for i in range(n_levels):
            # Squeeze Layer 
            layers += [Squeeze(input_shape)]
            C, H, W = C * 4, H // 2, W // 2
            output_shapes += [(-1, C, H, W)]
            
            # RevNet Block
            layers += [RevNetStep(C, norm) for _ in range(depth)]
            output_shapes += [(-1, C, H, W) for _ in range(depth)]

            if i < n_levels - 1: 
                # Split Layer
                layers += [Split(output_shapes[-1])]
                C = C // 2
                output_shapes += [(-1, C, H, W)]

        layers += [GaussianPrior((batch_size, C, H, W), ydim=ydim)]
        output_shapes += [output_shapes[-1]]

        if ydim is None:
            self.classifier = lambda x: None
        else:
            self.classifier = ClassPredict(output_shapes[-1][1], ydim)
        
        self.layers = nn.ModuleList(layers)
        self.output_shapes = output_shapes
        self.flatten()

    def forward(self, *inputs, **kwargs):
        x, obj = self.forward_(*inputs, **kwargs)
        clss = self.classifier(x)
        return x, obj, clss

    def sample(self, y=None, std=None):
        with torch.no_grad():
            kwargs = {'std': std} if std is not None else {}
            samples = self.reverse_(None, 0., y, **kwargs)[0]
            return samples

    def sample_with_grad(self, y, std=None):
        kwargs = {'std': std} if std is not None else {}
        samples = self.reverse_(None, 0., y, **kwargs)[0]
        return samples

    def flatten(self):
        # flattens the list of layers to avoid recursive call every time. 
        processed_layers = []
        to_be_processed = [self]
        while len(to_be_processed) > 0:
            current = to_be_processed.pop(0)
            if isinstance(current, LayerList):
                to_be_processed = [x for x in current.layers] + to_be_processed
            elif isinstance(current, Layer):
                processed_layers += [current]
        
        self.layers = nn.ModuleList(processed_layers)

