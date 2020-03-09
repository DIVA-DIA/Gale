"""
Here are defined the basic methods to initialize a CNN in a data driven way.
For initializing complex architecture or using more articulated stuff (e.g LDA
has two functions) one should implement his own init function.
"""

# Utils
import logging
import sys
import time
from itertools import count
from threading import Thread

import gc
import numpy as np
import psutil
from sklearn.feature_extraction.image import extract_patches_2d
from torch import nn

from init import advanced_init
from template.runner.base.base_routine import BaseRoutine


def init_model(model, data_loader, num_samples, init_function, max_patches, **kwargs):
    """
    Initialize a standard CNN composed by convolutional layer followed by fully
    connected layers.

    Parameters
    ----------
    model : DataParallel
        The model to initialize
    data_loader : torch.utils.data.dataloader.DataLoader
        The dataloader to take the data from
    num_samples : int
        Specifies how many points should be used to compute the data-driven
        initialization
    init_function : string
        Name of the function to use to init the model
    max_patches : int or double [0:0.99]
        Number of patches to extract. Exact if int, ratio of all possible patches if double.
    """
    # Collect initial data
    logging.info(f'Collect initial data #samples={num_samples}')
    X, y = _collect_initial_data(data_loader, num_samples)

    ###############################################################################################
    # Iterate over all layers
    logging.info('Iterate over all layers')
    memory = psutil.virtual_memory().used
    for index, layer in enumerate(list(list(model.children())[0].children()), start=1):

        if psutil.virtual_memory().used > memory:
            logging.info(f"[MEMORY] Higher memory usage: {psutil.virtual_memory().used:,}")
            memory = psutil.virtual_memory().used
            deadlock_counter = 0
            while psutil.virtual_memory().percent > 85:
                logging.info(f"[MEMORY] Memory usage is abouve 85%({psutil.virtual_memory().percent}). Sleeping 10 min")
                deadlock_counter += 1
                if deadlock_counter > 36:
                    logging.error(f"[MEMORY] It is 6h that the memory is above 85%. I quit!")
                    sys.exit(-1)
                time.sleep(600)

        # Get module from layer
        module = get_module_from_sequential_layer(layer)
        compute_parameters = False
        logging.info(f'\nLayer: {index} - layer: {type(module)}')

        # CONV LAYER
        if type(module) is nn.Conv2d:
            compute_parameters = True
            # Get kernel size of current layer
            kernel_size = module.kernel_size
            # Fix max patches if is not a ratio
            if max_patches > 1.0:
                max_patches = int(max_patches)
            logging.info(f'Get {max_patches} patches of kernel size {kernel_size} from {num_samples} samples')
            # Get the patches in a matrix form
            init_input, init_labels = get_patches(X=X, y=y, kernel_size=kernel_size, max_patches=max_patches)

        # LINEAR LAYER
        if type(module) is nn.Linear:
            compute_parameters = True
            # Reshape mini-batches into a matrix form
            init_input = minibatches_to_matrix(X)
            init_labels = minibatches_to_matrix(y)

        #######################################################################
        # Compute data-driven parameters (if a module with weights has been detected in this layer earlier)
        if compute_parameters:
            logging.info(f'Compute data-driven parameters with {init_function}')
            W, B = getattr(advanced_init, init_function)(
                layer_index=index, init_input=init_input, init_labels=init_labels, model=model, module=module, **kwargs
            )

            # Check parameters shape (better safe than sorry...)
            if module.weight.data.shape != W.shape:
                logging.error(f"Weight matrix dimension mis-match. Expected {module.weight.data.shape} got {W.shape}")
                sys.exit(-1)
            if module.bias is not None and module.bias.data.shape != B.shape:
                logging.error(f"Bias matrix dimension mis-match. Expected {module.bias.data.shape} got {B.shape}")
                sys.exit(-1)

            # Assign parameters in-place s.t. the hooks are not broken (e.g. for wandb)
            logging.info('Assign parameters')
            W, B = BaseRoutine().move_to_device(W, B, **kwargs)
            module.weight.data.copy_(W)
            if module.bias is not None:
                module.bias.data.copy_(B)

        #######################################################################
        # Forward pass of this layer
        logging.info('Forward pass')
        for i, _ in enumerate(X):
            # Move data to GPU if desired
            X[i] = BaseRoutine().move_to_device(X[i], **kwargs)[0]
            # Forward pass
            X[i] = layer(X[i])
            # Bring data back to CPU for further computing data-driven inits
            X[i] = BaseRoutine().move_to_device(X[i], no_cuda=True)[0]

    # Free some resources, just in case
    del X
    del y
    del init_input
    del init_labels
    gc.collect()
    pass


def _collect_initial_data(data_loader, num_samples):
    """Randomly samples the training set with the number of samples requested
    The precision of the number of samples collected is data_loader.batch_size: so one either gets the exact num_samples
    value of that plus at most data_loader.batch_size - 1 samples.
    This is necessary because we need for practical reason the structure of X,y to be mini-batches


    Examples
    --------
    This means that if 1000 samples are desired with a data_loader.batch_size of 12, the final
    amount of samples collected will be 1008, computed as (1000%12 + 1) * 12


    Parameters
    ----------
    data_loader : torch.utils.data.dataloader.DataLoader
        The dataloader to take the data from
    num_samples : int
        Specifies how many points should be used to compute the data-driven
        initialization

    Returns
    -------
    X : list(FloatTensor)
        Input samples structured in batches
    y : list(IntTensor)
        Target samples structured in batches (corresponding to X input samples)
    """
    X = []
    y = []
    # This is necessary because last batches might not have size mini-batch but smaller!
    collected_so_far = 0
    # Iterate troughs the dataset as many times as necessary to collect the amount samples required
    for j in count(1):
        for i, (input, target) in enumerate(data_loader, start=1):
            X.append(input)
            y.append(target['category_id'])
            collected_so_far += len(input)
            # If you collected enough samples leave. This makes a precision of +-data_loader.batch_size. See documentation
            if collected_so_far >= num_samples:
                return X, y
        logging.warning(f"Iterated trough the entire dataset {j} times already. "
                        f"Have {collected_so_far} samples. Keeping collecting...")


def get_module_from_sequential_layer(layer):
    """ Extract the module with weights from the sequential layer passed as parameter

    Parameters
    ----------
    layer : nn.Sequential
        The sequential layer in which we scan for a known module

    Returns
    -------
    module : nn.Module
        The module which we want to initialize i.e. the first child of the layer which is either a conv2d or a linear
        layer. This can be None is there aren't any of them.
    """
    assert isinstance(layer, nn.Sequential)
    for module in layer.children():
        module_type = type(module)
        if module_type is nn.Conv2d or module_type is nn.Linear:
            return module
    return None



def minibatches_to_matrix(A):
    """Flattens the a list of matrices of shape[[minibatch, dim_1, ..., dim_n], [minibatch, dim_1, ..., dim_n] ...] such
    that it becomes [minibatch size * len(list), dim_1 * dim_2 ... * dim_n]

    Parameters
    ----------
    A : list(FloatTensor)
        Input samples structured in batches


    Returns
    -------
    A : ndarray([number of elements, dimensionality of elements flattened]) or ndarray(number of elements)
        Flattened matrix
    """
    A = np.array([sample.data.view(-1).numpy() for minibatch in A for sample in minibatch])
    if A.shape[1] == 1:
        A = np.squeeze(A)
    return A


def get_patches(X, y, kernel_size, max_patches):
    """
    Extract patches out of a set X of tensors passed as parameter. Additionally returns the relative set of labels
    corresponding to each patch

    Parameters
    ----------
    X : list(FloatTensor)
        Input samples structured in batches
    y : list(IntTensor)
        Target samples structured in batches (corresponding to X input samples)
    kernel_size: tuple(width,height)
        size of the kernel to use to extract the patches.
    max_patches : int or double [0:0.99]
        number of patches to extract. Exact if int, ratio of all possible patches if double.

    Returns
    -------
    all_patches : ndarray(N*patch_per_image,depth*width*height)
        List of patches flattened into a matrix form
    labels : ndarray(N*patch_per_image,)
        List of labels for each of the elements of 'all_patches'
    """
    # Prepare the input data into a ndarray form
    tmp_X = np.array([e.data.numpy() for minibatch in X for e in minibatch])
    tmp_y = minibatches_to_matrix(y)

    # Init the return values
    all_patches, all_labels = [], []

    # For all images in X
    for image, label in zip(tmp_X, tmp_y):
        # Transform the image in the right format for extract_patches_2d(). Needed as channels are not in same order
        image = np.transpose(image, axes=[1, 2, 0])
        # Extract the patches
        extracted_patches = extract_patches_2d(image=image, patch_size=kernel_size, max_patches=max_patches)
        # Append the patches to the list of all patches extracted and "restore" the order of the channels.
        all_patches.append(np.transpose(extracted_patches, axes=[0, 3, 1, 2]))
        # Append the labels to the list of labels, by replicating the current one for as many patches has been extracted
        all_labels.append(np.repeat(label, len(extracted_patches)))

    # Flatten everything (here 'all_patches' and 'all_labels' are a list of lists. Each element of the outer list is
    # a list with all the patches (or corresponding labels) extracted from a single sample
    all_patches = np.array([patch
                            for patches in all_patches
                            for patch in patches])
    all_labels = np.array([label
                           for labels in all_labels
                           for label in labels])
    assert len(all_patches) == len(extracted_patches) * len(tmp_X)
    assert len(all_labels) == len(extracted_patches) * len(tmp_y)

    # Reshape the patches into a matrix form where one patch is one row and thus each pixel a column
    all_patches = all_patches.reshape(all_patches.shape[0], -1)
    assert len(all_patches.shape) == 2
    assert all_patches.shape[0] == len(extracted_patches) * len(tmp_X)
    assert all_patches.shape[1] == kernel_size[0] * kernel_size[1] * X[0].shape[1]  # X[0].shape[1] is the num channels

    return all_patches, all_labels
