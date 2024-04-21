import math
import torch
import torch.nn as nn


from torch import Tensor
from typing import Union
from torch.nn.common_types import _size_2_t
from biotorch.autograd.tfa.conv import Conv2dGrad
from biotorch.layers.metrics import compute_matrix_angle
import einops

class Conv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            layer_config: dict = None
    ):

        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode
        )

        self.layer_config = layer_config

        if self.layer_config is None:
            self.layer_config = {
                "type": "tfa"
            }

        if "options" not in self.layer_config:
            self.layer_config["options"] = {
                "constrain_weights": False,
                "gradient_clip": False,
                "init": "xavier"
            }

        self.options = self.layer_config["options"]
        self.type = self.layer_config["type"]
        self.init = self.options["init"]
        self.expand_ratio = self.options["expand_ratio"]
        if isinstance(self.expand_ratio, str):
            assert self.expand_ratio == "id"
            self.expand_ratio = 1
            self.identity_B = True
        else:
            self.identity_B = False
        N_out, N_in, K1, K2 = self.weight.size()
        N_mid = N_out * self.expand_ratio
        self.weight_backward_R = nn.Parameter(torch.Tensor(N_mid, N_in, K1, K2), requires_grad=True)
        if self.identity_B:
            self.weight_backward_B = nn.Parameter(torch.eye(N_out, N_mid).view(N_out, N_mid, 1, 1), requires_grad=False)
        else:
            self.weight_backward_B = nn.Parameter(torch.Tensor(N_out, N_mid, 1, 1), requires_grad=False)
        if self.bias is not None:
            N_b = self.bias.size()
            assert len(N_b) == 1
            self.bias_backward = nn.Parameter(torch.Tensor(N_b[0]*self.expand_ratio), requires_grad=False)
        else:
            self.register_parameter("bias", None)
            self.bias_backward = None

        self.init_parameters()

        if "constrain_weights" in self.options and self.options["constrain_weights"]:
            self.norm_initial_weights = torch.linalg.norm(self.weight)

        self.alignment = 0
        self.weight_ratio = 0

        if "gradient_clip" in self.options and self.options["gradient_clip"]:
            self.register_backward_hook(self.gradient_clip)

    def init_parameters(self) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # Xavier initialization
        if self.init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.weight_backward_R)
            if not self.identity_B:
                nn.init.xavier_uniform_(self.weight_backward_B,gain=math.sqrt((1+1/self.expand_ratio)/2))
            # Scaling factor is the standard deviation of xavier init.
            self.scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
                nn.init.constant_(self.bias_backward, 0)
        # Pytorch Default (Kaiming)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_backward_R, a=math.sqrt(5))
            if not self.identity_B:
                nn.init.kaiming_uniform_(self.weight_backward_B, a=math.sqrt(5))
            # Scaling factor is the standard deviation of Kaiming init.
            self.scaling_factor = 1 / math.sqrt(3 * fan_in)
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_backward, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            # Based on "Feedback alignment in deep convolutional networks" (https://arxiv.org/pdf/1812.06488.pdf)
            # Constrain weight magnitude
            if "constrain_weights" in self.options and self.options["constrain_weights"]:
                self.weight = torch.nn.Parameter(self.weight * self.norm_initial_weights /
                                                 torch.linalg.norm(self.weight))

        return Conv2dGrad.apply(x,
                                self.weight,
                                self.weight_backward_B,
                                self.weight_backward_R,
                                self.bias,
                                self.bias_backward,
                                self.stride,
                                self.padding,
                                self.dilation,
                                self.groups)

    def compute_alignment(self):
        with torch.no_grad():
            weight_backward = einops.einsum(self.weight_backward_B, self.weight_backward_R, 'N_out N_mid N_single1 N_single2, N_mid N_in K1 K2 -> N_out N_in N_single1 N_single2 K1 K2')
            weight_backward = einops.rearrange(weight_backward, 'N_out N_in 1 1 K1 K2 -> N_out N_in K1 K2')
            # print('weight_backward', weight_backward.shape, 'weight', self.weight.shape)
            self.alignment = compute_matrix_angle(weight_backward, self.weight)
        return self.alignment

    def compute_weight_ratio(self):
        with torch.no_grad():
            weight_backward = einops.einsum(self.weight_backward_B, self.weight_backward_R, 'N_out N_mid N_single1 N_single2, N_mid N_in K1 K2 -> N_out N_in N_single1 N_single2 K1 K2')
            weight_backward = einops.rearrange(weight_backward, 'N_out N_in 1 1 K1 K2 -> N_out N_in K1 K2')
            # print('weight_backward', weight_backward.shape, 'weight', self.weight.shape)
            self.weight_diff = torch.linalg.norm(weight_backward) / torch.linalg.norm(self.weight)
        return self.weight_diff

    @staticmethod
    def gradient_clip(module, grad_input, grad_output):
        grad_input = list(grad_input)
        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                grad_input[i] = torch.clamp(grad_input[i], -1, 1)
        return tuple(grad_input)
