import torch
from nalu import NALU
from functools import partial


def get_func_dataloader(n, range_, func, xdim=1):
    """
    Get data on [0, range] and apply func to find the target. Return a DataLoader.
    :return:
    """
    data_x = torch.Tensor(n, xdim).random_(to=range_)
    print(data_x)
    data_y = func(data_x)
    print(data_y)

    return
    dataset = TensorDataset(data_x, data_y)
    dataloader = DataLoader(dataset, batch_size=16)
    return dataloader

def test_addition():
    """
    Test a networks ability to learn a+b that generalises.
    :return:
    """

    training_data = get_func_dataloader(10000, 100, partial(torch.sum, dim=1), xdim=2)
    test_data = get_func_dataloader(10000, 100000, partial(torch.sum, dim=1), xdim=2)

    net = NALU(2, 1)

