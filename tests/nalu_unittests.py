import unittest
from functools import partial

import torch
from torch.utils.data import TensorDataset, DataLoader

from nalu.nalu import NALU, NAC
from tests.common import train, validate


def get_int_func_dataloader(n_samples, range_, func, xdim=1):
    """
    Get data on [0, range] and apply func to find the target. Return a DataLoader.

    Args:
        n_samples: number of samples
        range_: range of the values the samples can take.
        func: function to apply to each sample
        xdim: number of dimensions of the samples

    Returns:
        torch.DataLoader: pytorch DataLoader of the generated dataset, where y = func(x).

    Examples:
        >>> _ = torch.manual_seed(1)
        >>> next(iter(get_int_func_dataloader(4, 2, lambda x: x + 1, xdim=1)))
        [tensor([[1.],
                [1.],
                [0.],
                [0.]]), tensor([[2.],
                [2.],
                [1.],
                [1.]])]
        >>> next(iter(get_int_func_dataloader(4, 10, lambda s: s[:, 0] + s[:, 1] + 1, xdim=2)))
        [tensor([[1., 1.],
                [9., 2.],
                [8., 9.],
                [6., 3.]]), tensor([ 3., 12., 18., 10.])]

    """
    data_x = torch.Tensor(n_samples, xdim).random_(to=range_)
    data_y = func(data_x)
    dataset = TensorDataset(data_x, data_y)
    dataloader = DataLoader(dataset, batch_size=16)
    return dataloader

class TestScalarOperation(unittest.TestCase):

    def _test_scalar_operation(self, func, model, verbose=True):
        """
        Test a networks ability to learn scalar addition a+b that generalises.
        """

        training_data = get_int_func_dataloader(10000, 100, func, xdim=2)
        test_data = get_int_func_dataloader(10000, 100000, func, xdim=2)

        loss_function = torch.nn.L1Loss()

        # Test the model
        model = model(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        losses = train(model=model, loss_function=loss_function, optimizer=optimizer, train_loader=training_data,
                       val_loader=test_data, verbose=verbose, max_epochs=10)

        training_loss = validate(model, loss_function, training_data)
        test_loss = validate(model, loss_function, test_data)

        return training_loss, test_loss

    def test_scalar_addition(self, verbose=False):
        """
            Test addition of 2 numbers.
        """
        torch.manual_seed(1)
        training_loss, test_loss = self._test_scalar_operation(partial(torch.sum, dim=1), model=NAC, verbose=verbose)
        self.assertAlmostEqual(training_loss, 0.004, places=3)
        self.assertAlmostEqual(test_loss, 4.053, places=3)

        training_loss, test_loss = self._test_scalar_operation(partial(torch.sum, dim=1), model=NALU, verbose=verbose)
        self.assertAlmostEqual(training_loss, 0.004, places=3)
        self.assertAlmostEqual(test_loss, 4.0211, places=3)


    def test_scalar_multiplication(self, verbose=False):
        """
            Test multiplication of 2 numbers.
        """
        torch.manual_seed(1)
        training_loss, test_loss = self._test_scalar_operation(partial(torch.prod, dim=1), model=NAC, verbose=verbose)
        self.assertAlmostEqual(training_loss, 2304.0562, places=3)
        self.assertAlmostEqual(test_loss, 2468774400.0, places=3)
        training_loss, test_loss = self._test_scalar_operation(partial(torch.prod, dim=1), model=NALU, verbose=verbose)
        self.assertAlmostEqual(training_loss, 0.040132813, places=3)
        self.assertAlmostEqual(test_loss, 198006.06, places=0)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    unittest.main()
