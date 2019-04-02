"""
Activation Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function


class BinarizeSigF(Function):
    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0.5] = 1
        output[input < 0.5] = 0
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarizeTanhF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class TernarizeTanhF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output.data = input.data
        output.round_()
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BernolliSampleBinarizeF(Function):
    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output = torch.bernoulli(output)
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarySigmoid(nn.Module):
    def __init__(self):
        super(BinarySigmoid, self).__init__()
        self.hardsigmoid = nn.Sigmoid()

    def forward(self, input, stochastic=False):
        output = self.hardsigmoid(input)
        if not stochastic:
            output = binarizeSig(output)
        else:
            output = bernolliSample(output)
        return output


class BinaryTanh(nn.Module):
    """
    reference: https://github.com/DingKe/pytorch_workplace/blob/master/binary/modules.py#L10
    """

    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Tanh()

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarizeTanh(output)
        return output


class TernaryTanh(nn.Module):
    """
    reference: https://r2rt.com/beyond-binary-ternary-and-one-hot-neurons.html
    """

    def __init__(self):
        super(TernaryTanh, self).__init__()

    def forward(self, input):
        output = 1.5 * F.tanh(input) + 0.5 * F.tanh(-3 * input)
        output = ternarizeTanh(output)
        return output


def _sample_gumbel(input):
    noise = torch.rand(input.size())
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise)


def gumbel_softmax_sample(input, hard=False):
    temperature = 0.8
    noise = _sample_gumbel(input)
    x = (input + noise) / temperature
    x = F.softmax(x, dim=1)

    if hard:
        max_val, _ = torch.max(x, x.dim() - 1)
        x = x == max_val.expand_as(x)

    return x.view_as(input)


class DynamicGNoise(nn.Module):
    def __init__(self, shape, mean=0, std=0.05, cuda=False):
        super(DynamicGNoise).__init__()
        self.noise = torch.zeros(shape, shape)
        self.noise = self.noise.cuda() if cuda else self.noise
        self.noise = Variable(self.noise)
        self.std = std
        self.mean = mean

    def forward(self, x):
        if self.training:
            self.noise.data.normal_(mean=self.mean, std=self.std)
            print(x.size(), self.noise.size())
            x += self.noise.expand()
        return x

# aliases
binarizeSig = BinarizeSigF.apply
binarizeTanh = BinarizeTanhF.apply
ternarizeTanh = TernarizeTanhF.apply
bernolliSample = BernolliSampleBinarizeF.apply
