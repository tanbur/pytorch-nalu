import math

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module


class NAC(Module):
    r"""Neural Accumulator: :math:`y = Wx` where :math:`W = \tanh(\hat{W}) * \sigma(\hat{M})`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight_tanh: the learnable weights of the module of shape
            `(out_features x in_features)`
        weight_sigma: the learnable weights of the module of shape
            `(out_features x in_features)`

    Examples:
        >>> m = NAC(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])

        >>> m = NAC(2, 1)
        >>> _ = m.weight_tanh.data.fill_(4), m.weight_sigma.data.fill_(4)
        >>> m.weight
        tensor([[0.9814, 0.9814]], grad_fn=<ThMulBackward>)
        >>> input = torch.Tensor([[0, 1], [2, 5], [-1, 4]])
        >>> output = m(input)
        >>> output
        tensor([[0.9814],
                [6.8695],
                [2.9441]], grad_fn=<MmBackward>)
    """

    def __init__(self, in_features, out_features):
        super(NAC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_tanh = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight_tanh.data.uniform_(-stdv, stdv)
        self.weight_sigma.data.uniform_(-stdv, stdv)

    @property
    def weight(self):
        """
        Effective weight of NAC
        :return:
        """
        return torch.tanh(self.weight_tanh) * torch.sigmoid(self.weight_sigma)

    def forward(self, input):
        return F.linear(input, weight=self.weight)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features,
        )

class NALU(Module):
    r"""Neural Arithmetic Logic Unit: :math:`y = g * a + (1-g) * m` where :math:`g` is a sigmoidal gate,
    :math:`a` is a NAC and :math:`m` is a log-space NAC enabling multiplication.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        addition_cell: a NAC with learnable weights with the same input and output shapes as the NALU
        multiplication_cell: a NAC with learnable weights with the same input and output shapes as the NALU
        gate_weights: the learnable weights of the gate that interpolates between addition and multiplication of shape
            `(out_features x in_features)`

    Examples:
        >>> m = NALU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])


        Check we can multiply using the multiplication cell:
        >>> m = NALU(2, 1)
        >>> _ = (m.multiplication_cell.weight_tanh.data.fill_(4), m.multiplication_cell.weight_sigma.data.fill_(4),
        ...     m.gate_weights.data.fill_(-4))
        >>> m.multiplication_cell.weight
        tensor([[0.9814, 0.9814]], grad_fn=<ThMulBackward>)
        >>> input = torch.Tensor([[0, 10], [2, 5], [-1, 4]])
        >>> output = m(input)
        >>> torch.round(output)
        tensor([[ 0.],
                [10.],
                [ 4.]], grad_fn=<RoundBackward>)
    """

    def __init__(self, in_features, out_features):
        super(NALU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.addition_cell = NAC(in_features=in_features, out_features=out_features)
        self.multiplication_cell = NAC(in_features=in_features, out_features=out_features)
        self.gate_weights = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.gate_weights.data.uniform_(-stdv, stdv)
        self.addition_cell.reset_parameters()
        self.multiplication_cell.reset_parameters()

    def forward(self, input):
        summation = self.addition_cell(input)
        multiplication = torch.exp(self.multiplication_cell(torch.log(torch.abs(input) + 0.001)))
        gate = torch.sigmoid(F.linear(input=input, weight=self.gate_weights))

        return gate * summation + (1 - gate) * multiplication

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features,
        )


if __name__ == '__main__':
    import doctest
    doctest.testmod()
