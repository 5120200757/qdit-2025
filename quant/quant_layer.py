import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as st
import time

logger = logging.getLogger(__name__)
CLIPMIN = 1e-8


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


def save_tensor_hist(tensor, save_dir, name, bins=50):
    os.makedirs(save_dir, exist_ok=True)
    plt.hist(tensor.detach().cpu().numpy().flatten(), bins=bins)
    plt.title(name)
    plt.savefig(os.path.join(save_dir, f"{name}.png"))
    plt.close()


def save_tensor_txt(tensor, file_path):
    t = tensor.detach().cpu().numpy()
    t_flat = t.flatten()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 自动创建 x 目录

    np.savetxt(file_path, t_flat, fmt='%.8f')


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, always_zero: bool = False, lwc: bool = False, t_out: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.leaf_param = leaf_param
        self.scale_method = scale_method
        self.running_stat = False
        self.always_zero = always_zero
        self.x_outlier = None
        if self.leaf_param:
            self.x_min, self.x_max = None, None
        self.lwc = lwc
        self.t_out = t_out
        if self.lwc:
            self.upbound_factor, self.lowbound_factor, self.x_max, self.x_min = None, None, None, None
        self.sigmoid = nn.Sigmoid()

    def __repr__(self):
        s = super(UniformAffineQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "x_outlier"
        if key in state_dict:

            tensor = state_dict[key]
            if hasattr(self, "x_outlier") and "x_outlier" not in self._buffers:
                delattr(self, "x_outlier")
            if "x_outlier" in self._buffers:
                self._buffers["x_outlier"] = tensor.clone()
            else:
                self.register_buffer("x_outlier", tensor.clone(), persistent=True)
            state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point, self.x_outlier = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta)
            else:
                self.delta, self.zero_point, self.x_outlier = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        if self.running_stat:
            self.act_momentum_update(x)


        # x去离群点
        if self.t_out and self.x_outlier is not None:
            x_clone = x.clone()
            self.x_outlier = self.x_outlier.to(x_clone.device)
            xo = self.x_outlier.coalesce()
            idx = tuple(xo.indices())
            vals = xo.values()
            x_clone[idx] -= vals
        else:
            x_clone = x

        # start quantization
        x_int = round_ste(x_clone / self.delta) + self.zero_point

        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        # 加回离群点
        if self.t_out and self.x_outlier is not None:
                xo = self.x_outlier.coalesce()
                idx = tuple(xo.indices())
                vals = xo.values()
                x_dequant[idx] = vals

        return x_dequant

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert (self.inited)
        assert (self.leaf_param)

        x_min = x.data.min()
        x_max = x.data.max()
        self.x_min = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        if self.sym:
            delta = torch.max(self.x_min.abs(), self.x_max.abs()) / self.n_levels
        else:
            delta = (self.x_max - self.x_min) / (self.n_levels - 1) if not self.always_zero \
                else self.x_max / (self.n_levels - 1)

        delta = torch.clamp(delta, min=1e-8)
        if not self.sym:
            self.zero_point = (-self.x_min / delta).round() if not (self.sym or self.always_zero) else 0
        self.delta = torch.nn.Parameter(delta)

    def init_bound(self, x: torch.Tensor):
        x_clone = x.clone().detach()
        x_max = x_clone.max()
        x_min = x_clone.min()
        mean_val = x_clone.mean()
        std_val = x_clone.std()
        best_score = 1e+10
        for pct in [0.999, 0.9999, 0.99999]:
            try:
                new_max = torch.quantile(x_clone.reshape(-1), pct)
                new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
            except:
                new_max = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
                new_min = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
            x_q = self.quantize(x_clone, new_max, new_min)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                x_max = new_max
                x_min = new_min
        # for pct in [0.999, 0.995, 0.99]:
        #     z_values = st.norm.ppf(pct, loc=mean_val.detach().cpu(), scale=std_val.detach().cpu())
        #     new_max = mean_val + z_values * std_val
        #     new_min = mean_val - z_values * std_val
        #
        #     x_q = self.quantize(x_clone, new_max, new_min)
        #     score = lp_loss(x_clone, x_q, p=2, reduction='all')
        #     if score < best_score:
        #         best_score = score
        #         x_max = new_max
        #         x_min = new_min

        return x_max, x_min

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point, x_outlier = None, None, None
        x_clone = x.clone().detach()
        n_channels = x_clone.shape[-2] if len(x.shape) == 3 else x_clone.shape[0]
        if channel_wise:
            if self.lwc:
                init_value = 8.0
                self.upbound_factor = nn.Parameter(torch.ones(n_channels) * init_value)
                self.lowbound_factor = nn.Parameter(torch.ones(n_channels) * init_value)

                # x_max和x_min记录每个通道的最大值和最小值, 固定不变
                device_x = x_clone.device
                self.x_max = torch.zeros(n_channels, device=device_x)
                self.x_min = torch.zeros(n_channels, device=device_x)

                if self.t_out:
                    # 离群点矩阵
                    x_outlier = torch.zeros_like(x_clone)

                for c in range(n_channels):

                    if len(x.shape) == 3:
                        self.x_max[c], self.x_min[c] = self.init_bound(x_clone[:, c, :])
                    else:
                        self.x_max[c], self.x_min[c] = self.init_bound(x_clone[c])

                    if self.t_out:
                        if len(x.shape) == 3:
                            x_c = x_clone[:, c, :]
                        else:
                            x_c = x_clone[c]
                        mask = (x_c > self.x_max[c]) | (x_c < self.x_min[c])

                        if len(x.shape) == 3:
                            # mask = true的位置为原值，mask = false的位置为0
                            x_outlier[:, c, :] = torch.where(mask, x_c, torch.zeros_like(x_c))
                        else:
                            x_outlier[c] = torch.where(mask, x_c, torch.zeros_like(x_c))


                # 如果有提取离群点，则减去离群点后计算max和min的参数
                if self.t_out:
                    x_outlier = x_outlier.to_sparse()
                    xo = x_outlier.coalesce()
                    idx = tuple(xo.indices())
                    vals = xo.values()
                    x_clone[idx] -= vals

                    for c in range(n_channels):
                        if len(x.shape) == 3:
                            self.x_max[c], self.x_min[c] = self.init_bound(x_clone[:, c, :])
                        else:
                            self.x_max[c], self.x_min[c] = self.init_bound(x_clone[c])

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

            else:
                if len(x.shape) == 4:
                    x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:
                    x_max = x_clone.abs().max(dim=-1)[0]
                elif len(x.shape) == 3:
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=-1)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()

                # 如果是max，会加载检查点文件，就直接跳过
                if self.t_out and 'max' not in self.scale_method:
                    # 离群点矩阵
                    x_outlier = torch.zeros_like(x_clone)

                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c], _ = self.init_quantization_scale(x_clone[:, c, :], channel_wise=False)
                    else:
                        delta[c], zero_point[c], _ = self.init_quantization_scale(x_clone[c], channel_wise=False)

                    if self.t_out and 'max' not in self.scale_method:
                        if len(x.shape) == 3:
                            x_max, x_min = self.init_bound(x_clone[:, c, :])
                            x_c = x_clone[:, c, :]
                        else:
                            x_max, x_min = self.init_bound(x_clone[c])
                            x_c = x_clone[c]

                        mask = (x_c > x_max) | (x_c < x_min)

                        if len(x.shape) == 3:
                            # mask = true的位置为原值，mask = false的位置为0
                            x_outlier[:, c, :] = torch.where(mask, x_c, torch.zeros_like(x_c))
                        else:
                            x_outlier[c] = torch.where(mask, x_c, torch.zeros_like(x_c))
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
        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    delta = x_absmax / self.n_levels
                else:
                    delta = float(x.max().item() - x.min().item()) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
                delta = torch.tensor(delta).type_as(x)
            else:
                """
                4.结合
                """
                x_clone = x.clone().detach()
                x_max = x_clone.max()
                x_min = x_clone.min()
                mean_val = x_clone.mean()
                std_val = x_clone.std()
                best_score = 1e+10
                for pct in [0.999, 0.9999, 0.99999]:
                    try:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                    except:
                        new_max = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), pct * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                        new_min = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                    x_q = self.quantize(x_clone, new_max, new_min)
                    score = lp_loss(x_clone, x_q, p=2, reduction='all')
                    if score < best_score:
                        best_score = score
                        x_max = new_max
                        x_min = new_min
                # for pct in [0.999, 0.995, 0.990]:
                #     z_values = st.norm.ppf(pct, loc=mean_val.detach().cpu(), scale=std_val.detach().cpu())
                #     new_max = mean_val + z_values * std_val
                #     new_min = mean_val - z_values * std_val
                #
                #     x_q = self.quantize(x_clone, new_max, new_min)
                #     score = lp_loss(x_clone, x_q, p=2, reduction='all')
                #     if score < best_score:
                #         best_score = score
                #         x_max = new_max
                #         x_min = new_min

                delta = (x_max - x_min) / (2 ** self.n_bits - 1)
                zero_point = (- x_min / delta).round()


        if self.t_out and x_outlier is not None and  not x_outlier.is_sparse:
            x_outlier = x_outlier.to_sparse()

        return delta, zero_point, x_outlier

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False):
        super(QuantModule, self).__init__()
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight.data
        if org_module.bias is not None:
            self.bias = org_module.bias.data
        else:
            self.bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**self.weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**self.act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor, split: int = 0):
        if not self.disable_act_quant and self.use_act_quant:
            input = self.act_quantizer(input)
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_running_stat(self, running_stat: bool):
        self.act_quantizer.running_stat = running_stat
