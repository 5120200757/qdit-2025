import torch
from torch import nn
import logging
from quant.quant_layer import UniformAffineQuantizer, round_ste

logger = logging.getLogger(__name__)
CLIPMIN = 1e-8

def floor_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for floor operation.
    """
    return (x.floor() - x).detach() + x

class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, round_mode='learned_round_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        self.lwc = uaq.lwc
        if self.lwc:
            self.upbound_factor, self.lowbound_factor, self.x_max, self.x_min = uaq.upbound_factor, uaq.lowbound_factor, uaq.x_max, uaq.x_min
        self.sigmoid = uaq.sigmoid

        self.t_out = uaq.t_out
        if self.t_out:
            self.x_outlier = uaq.x_outlier

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

    def init_scale(self,x):
        device = self.x_max.device
        xmax = self.sigmoid(self.upbound_factor).to(device) * self.x_max
        xmin = self.sigmoid(self.lowbound_factor).to(device) * self.x_min

        if self.sym:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / self.n_levels
            delta = scale.clamp(min=CLIPMIN, max=1e4)
            # 量化范围的中间值
            zero_point = torch.zeros_like(delta)
        else:
            bound = xmax - xmin
            scale = bound / (self.n_levels - 1)
            delta = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (-(xmin) / (delta)).round()

        if len(x.shape) == 4:
            delta = delta.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
        elif len(x.shape) == 2:
            delta = delta.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        elif len(x.shape) == 3:
            delta = delta.view(1, -1, 1)
            zero_point = zero_point.view(1, -1, 1)
        else:
            raise NotImplementedError

        delta = nn.Parameter(delta.detach())
        zero_point = nn.Parameter(zero_point.detach())
        return delta, zero_point

    def forward(self, x):
        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.delta)
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(x / self.delta)
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(x / self.delta)
            rest = (x / self.delta) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            logger.info('Draw stochastic sample')
        elif self.round_mode == 'learned_hard_sigmoid':
            if self.lwc:
                self.delta, self.zero_point = self.init_scale(x)

            # 减去离群点
            x_work= x.clone()   

            if self.t_out and self.x_outlier is not None:
                xo = self.x_outlier.coalesce()
                idx = tuple(xo.indices())
                vals = xo.values()
                x_work[idx] -= vals

            x_floor = torch.floor(x_work / self.delta)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError('Wrong rounding mode')

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta

        # 加回离群点
        if self.t_out and self.x_outlier is not None:
            xo = self.x_outlier.coalesce()
            idx = tuple(xo.indices())
            vals = xo.values()
            x_float_q[idx] = vals

        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            # logger.info('Init alpha to be FP32')
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def extra_repr(self):
        s = 'bit={n_bits}, symmetric={sym}, round_mode={round_mode}' 
        return s.format(**self.__dict__)
