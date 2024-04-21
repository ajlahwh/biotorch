import torch

from torch import autograd


class Conv2dGrad(autograd.Function):
    """
    Autograd Function that Does a backward pass using the weight_backward matrix of the layer
    """
    @staticmethod
    def forward(context, input, weight, weight_backward_B, weight_backward_R, bias, bias_backward, stride, padding, dilation, groups):
        context.stride, context.padding, context.dilation, context.groups = stride, padding, dilation, groups
        # obtain the output of the R layer
        mid_input = torch.nn.functional.conv2d(input,
                                            weight_backward_R,
                                            bias=bias_backward,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=groups)
        context.save_for_backward(input, mid_input, weight, weight_backward_B, weight_backward_R, bias, bias_backward)
        output = torch.nn.functional.conv2d(input,
                                            weight,
                                            bias=bias,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=groups)
        return output

    @staticmethod
    def backward(context, grad_output_B):
        input, mid_input, weight, weight_backward_B, weight_backward_R, bias, bias_backward = context.saved_tensors
        grad_input = grad_weight = grad_weight_backward_B = grad_weight_backward_R = grad_bias = grad_bias_backward = None

        # grad output for R
        grad_output_R = torch.nn.grad.conv2d_input(input_size=mid_input.shape,
                                                    weight=weight_backward_B,
                                                    grad_output=grad_output_B,
                                                    stride=1,
                                                    padding=0,
                                                    dilation=1,
                                                    groups=1)
        # Gradient input
        if context.needs_input_grad[0]:
            # Use the FA constant weight matrix to compute the gradient
            grad_input = torch.nn.grad.conv2d_input(input_size=input.shape,
                                                    weight=weight_backward_R,
                                                    grad_output=grad_output_R,
                                                    stride=context.stride,
                                                    padding=context.padding,
                                                    dilation=context.dilation,
                                                    groups=context.groups)

        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input=input,
                                                      weight_size=weight.shape,
                                                      grad_output=grad_output_B,
                                                      stride=context.stride,
                                                      padding=context.padding,
                                                      dilation=context.dilation,
                                                      groups=context.groups)

        # Gradient weights for backward B
        assert not context.needs_input_grad[2]

        # Gradient weights for backward R
        if context.needs_input_grad[3]:
            grad_weight_backward_R = torch.nn.grad.conv2d_weight(input=input,
                                                      weight_size=weight_backward_R.shape,
                                                      grad_output=grad_output_R,
                                                      stride=context.stride,
                                                      padding=context.padding,
                                                      dilation=context.dilation,
                                                      groups=context.groups)
        # Gradient bias
        if bias is not None and context.needs_input_grad[4]:
            grad_bias = grad_output_B.sum(0).sum(2).sum(1)

        # Return the same number of parameters
        return grad_input, grad_weight, grad_weight_backward_B, grad_weight_backward_R, grad_bias, grad_bias_backward, None, None, None, None
