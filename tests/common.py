import numpy
import torch


def validate(model, loss_function, validation_loader):
    """
    Given a model, loss function and a dataset, calculate the loss.

    Args:
        model: PyTorch model.
        loss_function: Loss function.
        data_loader: pytorch DataLoader.

    Returns:
        float: Average error over the dataset.

    """
    validation_losses = []
    # We don't need the gradient, so speed things up
    with torch.no_grad():
        for val_x, val_target in validation_loader:
            output = model(val_x)
            val_loss = loss_function(output, val_target.unsqueeze(1))
            # val_loss = loss_function(output, val_target.squeeze(1))
            validation_losses.append(val_loss)
    return numpy.mean(validation_losses)


def train(model, loss_function, optimizer, train_loader, val_loader=None, max_epochs=100, verbose=False, debug=False):
    """
    Helper method for training a model.

    Args:
        model: PyTorch model.
        loss_function: Loss function.
        optimizer: Loss function.
        train_loader: pytorch DataLoader for the training set.
        val_loader: pytorch DataLoader for the validation set.
        max_epochs: maximum number of iterations over the training set.
        verbose: print training information
        debug: print debugging information, if verbose is True

    Returns:
        list: List of losses per epoch.
    """
    # Stores the loss at the end of each epoch
    losses = []

    # If we have validation data, use it
    if val_loader is not None:
        losses.append(validate(model=model, loss_function=loss_function, validation_loader=val_loader))
    # Else use an EWMA of the batch losses
    loss_eav = 0
    eav_lambda = 0.2

    for epoch in range(max_epochs):
        for batch_num, sample_batch in enumerate(train_loader):
            batch_x, batch_target = sample_batch
            output = model(batch_x)
            loss = loss_function(output, batch_target.unsqueeze(1))

            # Calculate the eav of batch loss
            # loss += 0.05 * sum(l1_loss(params, torch.zeros_like(params)) for params in net.parameters()) / total_params
            loss_eav += eav_lambda * loss - eav_lambda * loss_eav

            # zero the gradient buffers
            optimizer.zero_grad()
            # Store the gradient of loss function on the leaf nodes
            loss.backward()
            # Does the update
            optimizer.step()

        if val_loader is not None:
            epoch_loss = validate(model=model, loss_function=loss_function, validation_loader=val_loader)
        else:
            epoch_loss = loss_eav
        losses.append(epoch_loss)

        # epoch_print = DEBUG and (not (epoch % int(num_epochs / 10)))
        if verbose:
            print('\n')
            print('Epoch {}: loss: {}'.format(epoch, epoch_loss))
            print('Loss: {}'.format(losses[-1]))

        if verbose and debug:
            layer = model.fc3
            print('Weight:')
            print(layer.weight)
            # w = layer.weight.detach().numpy()
            # print(w.shape)
            # print(numpy.sum(w, axis=1))
            # plt.hist(w.flatten())
            # plt.show()
            print(layer.bias.data)
            print('Gradient:')
            print(layer.weight.grad)
            print(layer.bias.grad)

    if verbose:
        print('Initial loss: {:.3f}'.format(losses[0]))
        print('Final loss: {:.3f}'.format(losses[-1]))
    return losses


if __name__ == '__main__':
    import doctest
    doctest.testmod()
