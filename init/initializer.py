"""
Here are defined the basic methods to initialize a CNN in a data driven way.
For initializing complex architecture or using more articulated stuff (e.g LDA
has two functions) one should implement his own init function.
"""

# Utils
import copy
import logging
import sys
import time
from itertools import count

import gc
import numpy as np
import psutil
from sklearn.feature_extraction.image import extract_patches_2d
from torch import nn
from tqdm import tqdm

from init import advanced_init
from init.advanced_init import minibatches_to_matrix
from template.runner.base.base_routine import BaseRoutine


def init_model(**kwargs):
    """ Initialize a standard CNN composed by convolutional layer followed by fully connected layers."""
    if 'random' in kwargs['init_function']:
        return
    # Collect initial data
    X, y = _collect_initial_data(**kwargs)
    # Init the model
    logging.info('Initializing the model...')
    _init_module(X=X, y=y, prefix="", **kwargs)


def _init_module(X, y, model, prefix, sub_model=None, **kwargs):
    """Initialize a model passed by argument with the init function chosen.

    This function is used recursively for going deep on those blocks which have children such as nn.Sequential or
    ResNet-like blocks. When not in the main call, `sub_model` will be not None and will be processed as a whole
    new model.

    Parameters
    ----------
    X : list(FloatTensor)
        Input samples structured in batches
    y : list(IntTensor)
        Target samples structured in batches (corresponding to X input samples)
    model : DataParallel
        The model to initialize
    prefix : str
        String to pre-pend to logging the layer number
    sub_model : nn.Module
        The block of the network to initialize when performing a recursive call of this function
    """
    # Get the list of children to iterate on
    if sub_model is None:
        # Main call
        assert type(model) is nn.DataParallel
        children_modules = list(list(model.children())[0].children())
    else:
        # Recursive calls
        children_modules = list(sub_model.children())
    assert children_modules
    # This is used to slip the downsampling module in ResNet-like architectures as it would fail
    if hasattr(sub_model, 'downsample') and sub_model.downsample:
        children_modules = children_modules[:-1]

    for index, module in enumerate(children_modules, start=1):
        logging.info(f'\nLayer: {prefix}{index} - module: {type(module)}')
        _check_memory_usage()

        # CONV LAYER
        if type(module) is nn.Conv2d:
            # Get the patches in a matrix form
            init_input, init_labels = get_patches(X=X, y=y, kernel_size=module.kernel_size, **kwargs)
            _compute_and_assign_parameters(
                init_input=init_input, init_labels=init_labels, model=model, module=module, **kwargs
            )

        # LINEAR LAYER
        if type(module) is nn.Linear:
            init_input = X
            init_labels = y
            _compute_and_assign_parameters(
                init_input=init_input, init_labels=init_labels, model=model, module=module, **kwargs
            )

        # RECURSIVE CALL ON LAYER WITH CHILDREN
        if list(module.children()):
            logging.info(f"Deep copying X")
            tmpx = copy.deepcopy(X)
            logging.info(f"Done copying X")
            _init_module(
                X=tmpx,
                y=y,
                model=model,
                sub_model=module,
                prefix=f"{prefix}{index}.",
                **kwargs
            )

        #######################################################################
        # Forward pass of this module
        logging.info(f'Forward pass {prefix}{index}')
        for i, _ in tqdm(enumerate(X), total=len(X), unit='batch', ncols=130, leave=False):
            # Move data to GPU if desired
            X[i] = BaseRoutine().move_to_device(X[i], **kwargs)[0]
            # Forward pass
            X[i] = module(X[i])
            # Bring data back to CPU for further computing data-driven inits
            X[i] = BaseRoutine().move_to_device(X[i], no_cuda=True)[0]

    # Free resources
    del X, y
    gc.collect()


def _compute_and_assign_parameters(module, init_function, **kwargs):
    """Compute data-driven parameters

    Parameters
    ----------
    module : torch.nn.Module
        The module in which we'll put the weights
    init_function : string
        Name of the function to use to init the model
    """
    logging.info(f'Compute data-driven parameters with {init_function}')
    W, B = getattr(advanced_init, init_function)(module=module, **kwargs)
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


def _check_memory_usage():
    """ Checks if the memory usage is below a threshold; otherwise sleeps until it is"""
    MEMORY_THRESHOLD = 80
    deadlock_counter = 0
    while psutil.virtual_memory().percent > MEMORY_THRESHOLD:
        logging.info(f"[MEMORY] Memory usage is above {MEMORY_THRESHOLD}%({psutil.virtual_memory().percent}). Sleeping 10 min")
        deadlock_counter += 1
        if deadlock_counter > 36:
            logging.error(f"[MEMORY] It is 6h that the memory is above {MEMORY_THRESHOLD}%. I quit!")
            sys.exit(-1)
        time.sleep(600)


def _collect_initial_data(data_loader, num_samples, **kwargs):
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
    # If not specified take the entire dataset
    if num_samples is None:
        num_samples = len(data_loader.dataset)

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


def get_patches(X, y, kernel_size, patches_cap, **kwargs):
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
    patches_cap : int
        Upper bound on the number of patches to extract.

    Returns
    -------
    all_patches : ndarray(N*patch_per_image,depth*width*height)
        List of patches flattened into a matrix form
    labels : ndarray(N*patch_per_image,)
        List of labels for each of the elements of 'all_patches'
    """
    logging.info("Preparing data for collecting patches out of samples")
    # Prepare the input data into a list and ndarray form
    tmp_X = [e.data.numpy() for minibatch in X for e in minibatch]
    tmp_y = minibatches_to_matrix(y)

    # Compute the total amount of possible patches to be extracted from one sample
    possible_patches_per_sample = (tmp_X[0].shape[1] - kernel_size[0] + 1) * (tmp_X[0].shape[2] - kernel_size[1] + 1)
    # Set max_patches accordingly in order to either meet the cap or take all the samples
    num_samples = len(tmp_y)
    if patches_cap >= possible_patches_per_sample * num_samples:
        max_patches = possible_patches_per_sample
    else:
        max_patches = int(np.max((np.ceil(patches_cap / num_samples), 1)))  # At least one patch per sample

    logging.info(
        f'Get {max_patches} patches of kernel size {kernel_size}'
        f' from {num_samples} samples of size ({tmp_X[0].shape[0]}x{tmp_X[0].shape[1]}x{tmp_X[0].shape[2]})'
    )

    # Init the return values
    all_patches, all_labels = [], []
    # For all images in X
    for i, (image, label) in enumerate(zip(tmp_X, tmp_y)):
        if len(all_patches) >= patches_cap:
            # If you collected enough samples, stop iterating
            break
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
    all_patches = np.array([patch for patches in all_patches for patch in patches])
    all_labels = np.array([label for labels in all_labels for label in labels])

    # Reshape the patches into a matrix form where one patch is one row and thus each pixel a column
    all_patches = all_patches.reshape(all_patches.shape[0], -1)
    # Subset the selection if the cap was set smaller than number of samples received
    if patches_cap < len(all_patches):
        all_patches = all_patches[:patches_cap, :]
        all_labels = all_labels[:patches_cap]
    logging.info(f'Got {len(all_labels)} patches')
    assert len(all_patches.shape) == 2
    assert all_patches.shape[0] == patches_cap or all_patches.shape[0] == possible_patches_per_sample * num_samples
    assert all_patches.shape[1] == kernel_size[0] * kernel_size[1] * X[0].shape[1]  # X[0].shape[1] is the num channels

    return all_patches, all_labels
