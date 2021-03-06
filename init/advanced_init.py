"""
Here are defined the different versions of advanced initialization techniques.
"""
import datetime
import logging
import math
import sys
import time

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import nn, optim

from evaluation.metrics import accuracy
from init.util import lda, pca
from template.runner.base.base_routine import BaseRoutine
from template.runner.base.base_setup import BaseSetup
from util.TB_writer import TBWriter


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

def _normalize_weights(w, b):
    """
    Given both matrices of bias and weights this function return them normalized between [-1, 1]
    Parameters
    ----------
    w: ndarray 2D
        Weight matrix
    b : ndarray 1D
        Bias matrix

    Returns
    -------
        B and W normalized between [-1, 1]
    """
    max_value = max(np.max(np.abs(b)), np.max(np.abs(w)))
    w = w / max_value
    b = b / max_value
    return w, b


def _standardize_weights(w, b):
    """
    Given both matrices of bias and weights this function return them standardized with 0 mean
    and unit variance

    Parameters
    ----------
    w : ndarray 2D
        Weight matrix
    b : ndarray 1D
        Bias matrix

    Returns
    -------
        B and W standard normalized
    """
    logging.debug(f"w={w.shape} b={b.shape}")
    # The flattening&transposition is used for the random init which has the W matrix still in conv-layer shape
    w_tmp = w if len(w.shape) == 2 else _flatten_conv_filters(w).T
    joint_matrices = np.concatenate((w_tmp, np.expand_dims(b, axis=1)), axis=1)
    mean = np.mean(joint_matrices)
    std = np.std(joint_matrices)
    w = (w - mean) / std
    b = (b - mean) / std
    return w, b


def _scale_weights(w, b):
    """
    Given both matrices of bias and weights this function return them scaled such that the
    standard deviation is scaled accordging to Kaiming He observation, which works best with ReLU
    See more: https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

    Parameters
    ----------
    w: ndarray 2D
        Weight matrix
    b : ndarray 1D
        Bias matrix

    Returns
    -------
        B and W standardized
    """
    w = w * math.sqrt(2 / w.shape[0])
    b = b * math.sqrt(2 / w.shape[0])
    return w, b


def _adapt_magnitude(w, b, normalize, standardize, scale):
    """Scale the magnitude of w and b depending on the choice of the parameters

    Parameters
    ----------
    w : ndarray2d
        Weight matrix
    b : ndarray2d
        Bias array
    normalize : bool
        Flag for normalizing weights
    standardize : bool
        Flag for standardizing weights
    scale : bool
        Flag for scaling weights

    Returns
    -------
    w : ndarray2d
        Weight matrix scaled depending on the choice of the parameters
    b : ndarray2d
        Bias array scaled depending on the choice of the parameters
    """
    if normalize:
        w, b = _normalize_weights(w, b)
    if standardize:
        w, b = _standardize_weights(w, b)
    if scale:
        w, b = _scale_weights(w, b)
    return w, b


def _fit_weights_size(w, module):
    """Correct the sizes of W  according to the expected size of the module

    Parameters
    ----------
    w : ndarray 2D
        Weight matrix
    module : torch.nn.Module
        The module in which we'll put the weights
    Returns
    -------
    W : ndarray 2D
        Weights with theirs size fit to expected values of the module
    """
    out_size = module.out_channels if type(module) is nn.Conv2d else module.out_features

    # Add default values columns when num_columns < num_desired_dimensions
    if out_size > w.shape[1]:
        logging.warning(f"Init weight matrix smaller than the number of neurons (too few columns). "
                        f"Expected {out_size} got {w.shape[1]}. "
                        f"Filling missing dimensions with default values. ")
        # Get default values (the existing ones) from the network
        default_values = module.weight.data.cpu().numpy()
        if type(module) is nn.Conv2d:
            default_values = _flatten_conv_filters(default_values) # Brings it in the same shape as w
        else:
            default_values = default_values.T  # Linear Layers have neurons x input_dimensions
        assert len(default_values.shape) == len(w.shape)
        assert default_values.shape[0] == w.shape[0]
        w = np.hstack((w, default_values[:, w.shape[1]:]))

    # Discard extra columns when num_columns > num_desired_dimensions
    if out_size < w.shape[1]:
        logging.warning(f"Init weight matrix bigger than the number of neurons. "
                        f"Expected {out_size} got {w.shape[1]}. "
                        f"Removing extra columns. ")
        w = w[:, :out_size]

    return w

def _flatten_conv_filters(filters):
    """
    Takes the conv filters and flatten them.
    E.g. filters(24x3x5x5) turn into filters(75x24)

    Parameters:
    -----------
    filters : ndarray
        Conv filters in their natural shape 4D: [num_filters x in_channels x kernel_size x kernel_size]

    Returns
    -------
    ndarray
        Filters flattened in shape 2D: [num_filters x (in_channels * kernel_size * kernel_size)]
    """
    assert len(filters.shape) > 2
    return filters.T.reshape(-1, filters.shape[0])


def _reshape_flattened_conv_filters(filters, kernel_filter_size):
    """
    Reshape flattened conv filters back to a conv shape
    E.g. filter(75x24) turn into filters(25x3x5x5) with the kernel_size param as 5

    Parameters:
    -----------
    filters : ndarray
        Filters flattened in shape 2D: [num_filters x (in_channels * kernel_size * kernel_size)]
    kernel_filter_size: int
        Size of the square filters kernel e.g. 3x3 or 5x5

    Returns
    -------
    ndarray
        Conv filters in their natural shape 4D: [num_filters x in_channels x kernel_size x kernel_size]
    """
    return filters.T.reshape(filters.shape[1], -1, kernel_filter_size, kernel_filter_size)


def _basic_conv_procedure(w, b, module, sn_ratio, **kwargs):
    """
    Add missing column or remove extra one, set the bias to be the mathematical mean that makes sense and finally
    reshape the matrix W to match the expected shape of the module.
         
    Parameters
    ----------
    w : ndarray2d
        Weight matrix
    b : ndarray2d
        Bias array which should contain the means of the data X
    module : torch.nn.Module
        The module in which we'll put the weights
    sn_ratio : float
        Ratio of noise to be added on the conv weights

    Returns
    -------
    w : ndarray 4d
        The weight matrix in the same shape and size as expected by the module
    b : ndarray
        The bias vector multiplied by the transposed weights (see math formulation)
    """
    assert type(module) is nn.Conv2d
    # Correct the sizes of W according to the expected size of the module
    w = _fit_weights_size(w, module)

    # Set B to be -W*mean(X) such that it centers the data. At this point in B there are the means of the data
    b = -np.matmul(w.T, b)
    if module.bias is not None:  # Conv layers might have no bias
        assert b.shape == module.bias.shape

    # Reshape W to match the expected shape of the module
    w = _reshape_flattened_conv_filters(w, module.kernel_size[0])
    assert w.shape == module.weight.shape

    # Set W by adding it to the current random values
    if sn_ratio > 0:
        w += module.weight.data.cpu().numpy() / math.pow(10, 5 - sn_ratio)

    return w, b


def _filter_points_trimlda(init_input, init_labels, solver, iterations=5, **kwargs):
    """Given a set of points with their label it fits an LDA classifier and predicts on them.
    Then, only the samples positively classified are returned. This procedure can be done iteratively
    multiple times, specifying the amount by the parameter

    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    init_labels :  ndarray 2d
        Labels corresponding to the input data. The size is same as `init_input`
    solver : str
        Either 'eigen' or 'svd'
    iterations : int
        Number of iterations for trimming the LDA. If set to 0 nothing is performed
        
    Returns
    -------
    init_input : ndarray 2d
        Input data, trimmed
    init_labels :  ndarray 2d
        Labels corresponding to the input data.
    """
    assert iterations >= 0

    # Create the solver
    clf = LinearDiscriminantAnalysis(solver=solver)

    # Initialize to full list
    input = init_input
    labels = init_labels

    for i in range(1, iterations+1):
        logging.info('Filter points with trim-lda')
        start_time = time.time()
        logging.info(f'\titeration {i} of {iterations} #samples={len(input)}')
        clf.fit(X=input, y=labels)
        # Predict on the FULL LIST
        predictions = clf.predict(init_input)
        # Keep the input data where it is CORRECTLY predicted
        locs = np.where(predictions == init_labels)
        input = init_input[locs]
        labels = init_labels[locs]
        logging.info(
            f'\tAcc={(len(input)) / len(init_input):.2f} '
            f'Time taken: {datetime.timedelta(seconds=time.time() - start_time)}'
        )

    return input, labels


def _lda_discriminants(init_input, init_labels, lin_normalize, lin_scale, lin_standardize, trim_lda, **kwargs):
    """Compute LDA discriminants and relative bias and return them

    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    init_labels :  ndarray 2d
        Labels corresponding to the input data. The size is same as `init_input`
    lin_normalize : bool
    lin_standardize : bool
    lin_scale : bool
        Flags for adapting the magnitude of the weights of linear layers
    trim_lda : int
        Flag denoting if the samples for last layer will be trimmed

    Returns
    -------
    w : ndarray2d
        Weight matrix to compute LDA discriminants
    b : ndarray2d
        Bias array relative to W
    """
    # Reshape mini-batches into a matrix form
    init_input = minibatches_to_matrix(init_input)
    init_labels = minibatches_to_matrix(init_labels)

    logging.info('LDA Discriminants')
    if trim_lda:
        init_input, init_labels = _filter_points_trimlda(init_input=init_input, init_labels=init_labels, **kwargs)

    W, B = lda.discriminants(X=init_input, y=init_labels, **kwargs)
    # Adapt the size of the weights
    return _adapt_magnitude(w=W, b=B, normalize=lin_normalize, standardize=lin_standardize, scale=lin_scale)


def _initialize_classifier(
        *,
        module,
        init_input,
        init_labels,
        retrain,
        retrain_normalize,
        retrain_standardize,
        retrain_scale,
        **kwargs):
    """ Initialize the classifier layer with LDA and if specified further train it with XE

    Parameters
    ----------
    module : torch.nn.Module
        The module in which we'll put the weights
    retrain : bool
        Flag for retraining the classifier
    retrain_normalize : bool
    retrain_standardize : bool
    retrain_scale : bool
        Flags for adapting the magnitude of the weights of linear layer after retraining

    Returns
    -------
    w : ndarray2d
        Weight matrix to compute LDA discriminants
    b : ndarray2d
        Bias array relative to W
    """
    # Compute LDA discriminants
    W, B = _lda_discriminants(init_input, init_labels, **kwargs)

    # If requested, further train the classifier with LDA
    if retrain:
        # Create classifier
        classifier = nn.Linear(in_features=module.weight.shape[1], out_features=module.weight.shape[0])
        # Init it with the LDA numbers
        classifier.weight.data.copy_(torch.from_numpy(W))
        classifier.bias.data.copy_(torch.from_numpy(B))
        classifier = classifier.cuda()
        # Further train it
        _train_classifier(classifier=classifier, init_input=init_input, init_labels=init_labels, **kwargs)
        # Copy the final weights and adapt their magnitude
        W = classifier.weight.data.cpu().numpy()
        B = classifier.bias.data.cpu().numpy()
        W, B = _adapt_magnitude(w=W, b=B, normalize=retrain_normalize, standardize=retrain_standardize, scale=retrain_scale)
    return torch.from_numpy(W), torch.from_numpy(B)


def _train_classifier(classifier, init_input, init_labels, retrain_epochs, retrain_lr, retrain_wd, **kwargs):
    """ Train the classifier passed as parameters with XE

    Parameters
    ----------
    classifier : torch.nn.Module
        The classifier  to train
    init_input : list(FloatTensor)
        Input samples structured in batches
    init_labels : list(FloatTensor)
        Input labels structured in batches
    retrain_lr : float
        Learning rate for the last layer
    retrain_wd : float
        Weight decay for the last layer
    retrain_epochs : int 
        Number of epochs for the last layer

    Returns
    -------
    w : ndarray2d
        Weight matrix to compute LDA discriminants
    b : ndarray2d
        Bias array relative to W
    """
    # Measure initial accuracy
    acc = 0
    for i, (input, target) in enumerate(zip(init_input, init_labels), 0):
        input, target = BaseRoutine.move_to_device(input=input, target=target, **kwargs)
        output = classifier(input)
        acc += accuracy(output.data, target.data, topk=(1,))[0]
    acc /= i
    lda_accuracy = acc.data.cpu().numpy()
    print(f'\t[-1] {lda_accuracy}')
    # Further train it
    optimizer = optim.SGD(classifier.parameters(), lr=retrain_lr, weight_decay=retrain_wd, momentum=0.9, nesterov=True)
    criterion = BaseSetup().get_criterion(**kwargs)

    best_acc = 0
    # for e in count(1):
    for e in range(retrain_epochs):
        # if lr < 0.0001:
        #     print(f"LR is now {lr} -> Exiting!")
        #     break
        acc = 0
        for i, (input, target) in enumerate(zip(init_input, init_labels), 0):
            input, target = BaseRoutine.move_to_device(input=input, target=target, **kwargs)
            optimizer.zero_grad()
            output = classifier(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            acc += accuracy(output.data, target.data, topk=(1,))[0]
        acc /= i
        acc = acc.data.cpu().numpy()
        print(f'\t[{e}] {acc}')
        TBWriter().add_scalar(tag='init_lda_accuracy', scalar_value=lda_accuracy, global_step=e)
        TBWriter().add_scalar(tag='init_retrain', scalar_value=acc, global_step=e)

    print(f'\t[BEST] {best_acc}')
    TBWriter().add_scalar(tag='init_best_acc', scalar_value=best_acc)




#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def orthogonal(
        module,
        **kwargs
):
    """
    Initialize the layer with orthonormal matrices lt values from:
    https://github.com/ducha-aiki/LSUV-pytorch/blob/master/LSUV.py

   Orthogonal init code is taken (and adapted) from:
   Lasagne https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py

    Parameters
    ----------
    model : torch.nn.parallel.data_parallel.DataParallel
        The actual model we're initializing. It is used to infer the depth and possibly other information

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array
    """
    shape = module.weight.data.shape
    flat_shape = shape[0], np.prod(shape[1:])
    a = np.random.normal(0.0, 1.0, flat_shape)#w;
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v

    W = q.reshape(shape)
    B = np.zeros(module.weight.shape[0])

    return torch.from_numpy(W), torch.from_numpy(B)

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def random(
        module,
        **kwargs
):
    """Initialize the layer default values from the network. If left untouched, the default values are as follows:

        https://pytorch.org/docs/stable/nn.html#linear-layers
        https://pytorch.org/docs/stable/nn.html#convolution-layers

    Parameters
    ----------
    model : torch.nn.parallel.data_parallel.DataParallel
        The actual model we're initializing. It is used to infer the depth and possibly other information

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array
    """
    # Init W and B with the default values
    W = module.weight.data.cpu().numpy()
    B = module.bias.data.cpu().numpy() if module.bias is not None else np.zeros(module.weight.shape[0])
    return torch.from_numpy(W), torch.from_numpy(B)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def randisco(        
        init_input,
        init_labels,        
        module,
        **kwargs
):
    """Initialize the layer default values from the network, then uses LDA discriminants.
     If left untouched, the default values are as follows:

        https://pytorch.org/docs/stable/nn.html#linear-layers
        https://pytorch.org/docs/stable/nn.html#convolution-layers

    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    init_labels :  ndarray 2d
        Labels corresponding to the input data. The size is same as `init_input`
    module : torch.nn.Module
        The module in which we'll put the weights

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array
    """
    if type(module) is nn.Conv2d:
        # Init W and B with the default values
        W = module.weight.data.cpu().numpy()
        B = module.bias.data.cpu().numpy() if module.bias is not None else np.zeros(module.weight.shape[0])
        return torch.from_numpy(W), torch.from_numpy(B)

    ##################################################################
    if type(module) is nn.Linear:
        return _initialize_classifier(module=module, init_input=init_input, init_labels=init_labels, **kwargs)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def pure_lda(
        init_input,
        init_labels,
        module,
        conv_normalize,
        conv_standardize,
        conv_scale,
        **kwargs
):
    """Initialize the layer with pure LDA function and pure LDA discriminants for the last layer

    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    init_labels :  ndarray 2d
        Labels corresponding to the input data. The size is same as `init_input`
    module : torch.nn.Module
        The module in which we'll put the weights
    conv_normalize : bool
    conv_standardize : bool
    conv_scale : bool
        Flags for adapting the magnitude of the weights of convolutional layers

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array
    """
    if type(module) is nn.Conv2d:
        logging.info('Pure LDA Transform')
        W, B = lda.transform(X=init_input, y=init_labels, **kwargs)
        # Adapt the size of the weights
        W, B = _adapt_magnitude(w=W, b=B, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)
        W, B = _basic_conv_procedure(W, B, module, **kwargs)
        return torch.from_numpy(W), torch.from_numpy(B)

    ##################################################################
    if type(module) is nn.Linear:
        return _initialize_classifier(module=module, init_input=init_input, init_labels=init_labels, **kwargs)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def mirror_lda(
        init_input,
        init_labels,
        module,
        conv_normalize,
        conv_standardize,
        conv_scale,
        **kwargs
):
    """Initialize the layer with LDA function, but it duplicates the columns with non-zero eigenvalue and mirrors them
    and pure LDA discriminants for the last layer

    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    init_labels :  ndarray 2d
        Labels corresponding to the input data. The size is same as `init_input`
    module : torch.nn.Module
        The module in which we'll put the weights
    conv_normalize : bool
    conv_standardize : bool
    conv_scale : bool
        Flags for adapting the magnitude of the weights of convolutional layers

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array
    """
    if type(module) is nn.Conv2d:
        logging.info('Mirror LDA Transform')
        W, B = lda.transform(X=init_input, y=init_labels, **kwargs)
        # Compute available columns
        available_columns = module.weight.shape[0]
        # Discard dimensions as necessary
        if W.shape[1] * 2 > available_columns:
            W = W[:, 0:int(available_columns / 2)]
        # Mirror W. You don't mirror B because it has to be size of mean(X)
        W = np.hstack((W, -W))
        # Adapt the size of the weights
        W, B = _adapt_magnitude(w=W, b=B, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)
        W, B = _basic_conv_procedure(W, B, module, **kwargs)
        return torch.from_numpy(W), torch.from_numpy(B)

    ##################################################################
    if type(module) is nn.Linear:
        return _initialize_classifier(module=module, init_input=init_input, init_labels=init_labels, **kwargs)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def highlander_lda(
        init_input,
        init_labels,
        module,
        conv_normalize,
        conv_standardize,
        conv_scale,
        **kwargs
):
    """
    Initialize the layer with LDA by making a 1 vs ALL comparison for all classes. Thus we can initialize more columns
    with meaningful numbers. In a regular LDA setting we have C-1 columns where C is the number of classeses. Here we
    get 2*C because for each class there is a 1vsALL setting which gives us two columns.

    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    init_labels :  ndarray 2d
        Labels corresponding to the input data. The size is same as `init_input`
    module : torch.nn.Module
        The module in which we'll put the weights
    conv_normalize : bool
    conv_standardize : bool
    conv_scale : bool
        Flags for adapting the magnitude of the weights of convolutional layers

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array
    """
    if type(module) is nn.Conv2d:
        logging.info('Highlander LDA Transform')
        classes = np.unique(init_labels)

        # Check if size of model allows (has enough neurons)
        if module.weight.shape[0] < len(classes) * 2:
            logging.error(
                f"Model does not have enough neurons. Expected at least |C|*2 got {module.weight.shape[0]}"
            )
            sys.exit(-1)

        # Compute available columns per class. We need 2 minimum but we can use more
        available_columns_per_class = int(module.weight.shape[0] / len(classes))

        # Init W with the default values
        # -> If number of neuron is not a multiple of classes we leave random values in those columns
        W = _flatten_conv_filters(module.weight.data.cpu().numpy())

        # |C| times
        for i, l in enumerate(classes):
            logging.info(f'LDA Transform for class {l}')
            # Make a new set of labels of 1 vs ALL
            tmp_labels = init_labels.copy()
            tmp_labels[np.where(init_labels == l)] = 0
            tmp_labels[np.where(init_labels != l)] = 1
            w, b = lda.transform(X=init_input, y=tmp_labels, **kwargs)
            # Adapt the size of the weights
            w, b = _adapt_magnitude(w=w, b=b, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)

            start_index = i * available_columns_per_class
            end_index = start_index + np.min([available_columns_per_class, w.shape[1]])
            W[:, start_index:end_index] = w[:, 0:available_columns_per_class]
            # Mean of the data is always the same regardless of the labels arrangement
            B = b

        # Adapt the size of the weights
        # W, B = _adapt_magnitude(w=W, b=B, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)
        W, B = _basic_conv_procedure(W, B, module, **kwargs)
        return torch.from_numpy(W), torch.from_numpy(B)

    ##################################################################
    if type(module) is nn.Linear:
        return _initialize_classifier(module=module, init_input=init_input, init_labels=init_labels, **kwargs)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def pure_pca(
        init_input,
        module,
        conv_normalize,
        conv_standardize,
        conv_scale,
        **kwargs
):
    """Initialize the layer with pure PCA and leave the final layer "as it"

    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    module : torch.nn.Module
        The module in which we'll put the weights
    conv_normalize : bool
    conv_standardize : bool
    conv_scale : bool
        Flags for adapting the magnitude of the weights of convolutional layers

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array
    """
    if type(module) is nn.Conv2d:
        logging.info('PCA Transform')
        W, B = pca.transform(init_input)
        # Adapt the size of the weights
        W, B = _adapt_magnitude(w=W, b=B, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)
        W, B = _basic_conv_procedure(W, B, module, **kwargs)
        return torch.from_numpy(W), torch.from_numpy(B)

    ##################################################################
    if type(module) is nn.Linear:
        # Init W and B with the default values
        W = module.weight.data.cpu().numpy()
        B = module.bias.data.cpu().numpy()
        return torch.from_numpy(W), torch.from_numpy(B)

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def pcdisc(
        init_input,
        init_labels,
        module,
        conv_normalize,
        conv_standardize,
        conv_scale,
        **kwargs
):
    """Initialize the layer with pure PCA and the final layer with linear discriminants

    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    init_labels :  ndarray 2d
        Labels corresponding to the input data. The size is same as `init_input`
    module : torch.nn.Module
        The module in which we'll put the weights
    conv_normalize : bool
    conv_standardize : bool
    conv_scale : bool
        Flags for adapting the magnitude of the weights of convolutional layers

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array
    """
    if type(module) is nn.Conv2d:
        logging.info('PCA Transform')
        W, B = pca.transform(init_input)
        # Adapt the size of the weights
        W, B = _adapt_magnitude(w=W, b=B, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)
        W, B = _basic_conv_procedure(W, B, module, **kwargs)
        return torch.from_numpy(W), torch.from_numpy(B)

    ##################################################################
    if type(module) is nn.Linear:
        return _initialize_classifier(module=module, init_input=init_input, init_labels=init_labels, **kwargs)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def lpca(
        init_input,
        init_labels,
        module,
        conv_normalize,
        conv_standardize,
        conv_scale,
        **kwargs
):
    """Initialize the layer with both PCA and LDA. The amount of columns could be an hyper-parameter

    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    init_labels :  ndarray 2d
        Labels corresponding to the input data. The size is same as `init_input`
    module : torch.nn.Module
        The module in which we'll put the weights
    conv_normalize : bool
    conv_standardize : bool
    conv_scale : bool
        Flags for adapting the magnitude of the weights of convolutional layers

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array
    """
    if type(module) is nn.Conv2d:
        logging.info('LDA Transform')
        w, b = lda.transform(X=init_input, y=init_labels, **kwargs)
        w, B = _adapt_magnitude(w=w, b=b, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)
        logging.info('PCA Transform')
        p, c = pca.transform(init_input)
        p, c = _adapt_magnitude(w=p, b=c, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)

        # Compute available columns
        half_available_columns = int(module.weight.shape[0]/2)
        if module.weight.shape[0] % 2 != 0:
            logging.info(
                f"Not all columns will be initialized as the shape {module.weight.shape[0]} is not divisible by 2"
            )

        # Init W with the default values
        W = _flatten_conv_filters(module.weight.data.cpu().numpy())
        # Add LDA columns in the first part
        # In case of eigen solver each part will be half-half. Since svf svd solver only gives |C|-1 columns the pca
        # part takes all - |C|-1 columns (the rest after the lda has been set basically)
        end_index_first_part = np.min([half_available_columns, w.shape[1]])
        W[:, 0:end_index_first_part] = w[:, 0:end_index_first_part]
        # Add PCA columns in the second part
        W[:, end_index_first_part:] = p[:, 0:W.shape[1]-end_index_first_part]
        W, B = _basic_conv_procedure(W, B, module, **kwargs)
        return torch.from_numpy(W), torch.from_numpy(B)

    ##################################################################
    if type(module) is nn.Linear:
        return _initialize_classifier(module=module, init_input=init_input, init_labels=init_labels, **kwargs)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def reverse_pca(
        init_input,
        init_labels,
        module,
        conv_normalize,
        conv_standardize,
        conv_scale,
        **kwargs
):
    """Initialize the layer with the reverse PCA procedure. The basic idea is to leverage labels to find which dimensions
    would minimize the variance within one class i.e. we select the last columns of the PCA matrix L as candidate for
    the given class. Intuitively these columns are those who express the least variance for the selected class. This
    approach might be detrimental if two classes share common non-expressive set of features as these will be selected
    for both classes

    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    init_labels :  ndarray 2d
        Labels corresponding to the input data. The size is same as `init_input`
    module : torch.nn.Module
        The module in which we'll put the weights
    conv_normalize : bool
    conv_standardize : bool
    conv_scale : bool
        Flags for adapting the magnitude of the weights of convolutional layers

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array
    """
    if type(module) is nn.Conv2d:

        # Check if size of model allows (has enough neurons)
        if module.weight.shape[0] < len(np.unique(init_labels)) * 2:
            logging.error(
                f"Model does not have enough neurons. Expected at least |C|*2 got {module.weight.shape[0]}"
            )
            sys.exit(-1)

        # Compute available columns per class
        available_columns = int(module.weight.shape[0] / len(np.unique(init_labels)))
        bias_available_columns = int(init_input.shape[1] / len(np.unique(init_labels)))

        # Init W with the default values and B with zeros
        W = _flatten_conv_filters(module.weight.data.cpu().numpy())
        B = np.zeros(init_input.shape[1])  # Bias should be size of input because its later multiplied by -Wx
        classes = np.unique(init_labels)

        # |C| times
        for i, l in enumerate(classes):
            logging.info('Iteration of class {}'.format(l))
            # Select only the samples corresponding to a specific class
            p, c = pca.transform(init_input[np.where(init_labels == l)])

            start_index = i * available_columns
            end_index = start_index + available_columns
            W[:, start_index:end_index] = p[:, -available_columns:]

            start_index = i * bias_available_columns
            end_index = start_index + bias_available_columns
            B[start_index:end_index] = c[-bias_available_columns:]

        # Adapt the size of the weights
        W, B = _adapt_magnitude(w=W, b=B, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)
        W, B = _basic_conv_procedure(W, B, module, **kwargs)
        return torch.from_numpy(W), torch.from_numpy(B)

    ##################################################################
    if type(module) is nn.Linear:
        return _initialize_classifier(module=module, init_input=init_input, init_labels=init_labels, **kwargs)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def relda(
        init_input,
        init_labels,
        module,
        conv_normalize,
        conv_standardize,
        conv_scale,
        **kwargs
):
    """Initialize the layer with repreated LDA (reLDA). The core idea is to split the available columns into a number
    of iterations s.t. each iteration has the same amount of columns. At each iteration a LDA classifier is fit to the
    set of samples (the first time on the full set) and then used to make predictions on it. We keep from this LDA the
    amount of columns at our disposal for the iteration and then take only those samples which have been misclassified.
    At this point we repeat the process, thus, iteratively, fitting more and more classifiers to being able to
    collectively correctly classify more samples. At the end of the process, the large part of the set should be able to
    be correctly classified but not by the same columns. Each set of columns will be responsible to classify a different
    set of points.

    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    init_labels :  ndarray 2d
        Labels corresponding to the input data. The size is same as `init_input`
    module : torch.nn.Module
        The module in which we'll put the weights
    conv_normalize : bool
    conv_standardize : bool
    conv_scale : bool
        Flags for adapting the magnitude of the weights of convolutional layers

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array

    Notes
    -----

    CINIC10 , InitBaseline:

        SOLVER=EIGEN always fails!
    """
    if type(module) is nn.Conv2d:
        classes = np.unique(init_labels)

        # Check if size of model allows (has enough neurons)
        if module.weight.shape[0] < len(classes) * 2:
            logging.error(
                f"Model does not have enough neurons to make sense.Expected at least |C|*2 got {module.weight.shape[0]}"
            )
            sys.exit(-1)

        # Init W with the default values and B with zeros
        W = _flatten_conv_filters(module.weight.data.cpu().numpy())
        B = np.zeros(init_input.shape[1])  # Bias should be size of input because its later multiplied by -Wx

        # Compute available columns per class
        available_columns_per_iteration = len(classes) - 1
        # Compute number of iterations
        N = int(module.weight.shape[0] / available_columns_per_iteration)

        # N times
        logging.info('LDA Transform')
        clf = LinearDiscriminantAnalysis(solver=kwargs['solver'])
        initial_size = len(init_input)
        for i in range(N):
            logging.info(f'Iteration {i+1} of {N} #samples={len(init_input)}')
            if len(init_input) < 1:
                logging.info('No more wrong samples -> exiting loop')
                break
            if len(init_input) == len(np.unique(init_labels)):
                logging.info('Number of samples is equal to the number of classes -> exiting loop')
                break
            logging.info(f'\tfitting...')
            clf.fit(X=init_input, y=init_labels)
            w = -clf.scalings_
            b = np.mean(init_input, axis=0)
            # Adapt the size of the weights
            w, b = _adapt_magnitude(w=w, b=b, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)

            # Filter the input data with the wrong predictions i.e. keep those location where it is WRONGLY predicted
            logging.info(f'\tpredicting...')
            predictions = clf.predict(init_input)
            locs = np.where(predictions != init_labels)
            logging.info(f'\tcurrent acc={1 - float(len(locs[0]))/len(init_input):.2f} ...')
            init_input = init_input[locs]
            init_labels = init_labels[locs]
            logging.info(f'\tglobal integrated acc={(initial_size - len(init_input)) / initial_size:.2f} ...')

            # Set main weights
            start_index = i * available_columns_per_iteration
            end_index = start_index + np.min([available_columns_per_iteration, w.shape[1]])
            W[:, start_index:end_index] = w[:, 0:available_columns_per_iteration]

            # Set bias
            # start_index = i * bias_available_columns_per_iteration
            # end_index = start_index + bias_available_columns_per_iteration
            # B[start_index:end_index] = b[0:bias_available_columns_per_iteration]
            # TODO better global bias or bias for each iteration? Same for the other after highlander
            # Mean of the data is always the same regardless of the labels arrangement
            B = b

        # Adapt the size of the weights
        # TODO done above, what is better?
        # W, B = _adapt_magnitude(w=W, b=B, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)
        W, B = _basic_conv_procedure(W, B, module, **kwargs)
        return torch.from_numpy(W), torch.from_numpy(B)

    ##################################################################
    if type(module) is nn.Linear:
        return  _initialize_classifier(module=module, init_input=init_input, init_labels=init_labels, **kwargs)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def greedya(
        init_input,
        init_labels,
        module,
        conv_normalize,
        conv_standardize,
        conv_scale,
        **kwargs
):
    """


    Parameters
    ----------
    init_input : ndarray 2d
        Input data, either from images or feature space. Format is 2D: [data_size x dimensionality]
        data_size can be computed as (num_samples * patches_x_image), whereas dimensionality is typically
        (in_channels * kernel_size * kernel_size)
    init_labels :  ndarray 2d
        Labels corresponding to the input data. The size is same as `init_input`
    module : torch.nn.Module
        The module in which we'll put the weights
    conv_normalize : bool
    conv_standardize : bool
    conv_scale : bool
        Flags for adapting the magnitude of the weights of convolutional layers

    Returns
    -------
    w : torch.Tensor
        Weight matrix
    b : torch.Tensor
        Bias array
    """
    if type(module) is nn.Conv2d:
        logging.info('Greedya Transform')
        classes = np.unique(init_labels)

        # Check if size of model allows (has enough neurons)
        if module.weight.shape[0] < len(classes) * 3:
            logging.error(
                f"Model does not have enough neurons. Expected at least |C|*3 got {module.weight.shape[0]}"
            )
            sys.exit(-1)

        # Init W with the default values
        W = _flatten_conv_filters(module.weight.data.cpu().numpy())
        # W = np.zeros_like(_flatten_conv_filters(module.weight.data.cpu().numpy()))
        start_index = 0

        # ----------------------------------------------------------------------------------------------
        # Mirror LDA (note that the normal LDA is the first iteration of ReLDA
        logging.info('Mirror LDA Transform')
        w, b = lda.transform(X=init_input, y=init_labels, **kwargs)
        w, B = _adapt_magnitude(w=w, b=b, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)
        W[:, start_index:w.shape[1]] = -w
        start_index += w.shape[1]

        # ----------------------------------------------------------------------------------------------
        logging.info('Highlander LDA')
        assert kwargs['solver'] == 'svd'
        # |C| times
        for i, l in enumerate(classes):
            logging.info(f'Highlander, class {l}')
            # Make a new set of labels of 1 vs ALL
            tmp_labels = init_labels.copy()
            tmp_labels[np.where(init_labels == l)] = 0
            tmp_labels[np.where(init_labels != l)] = 1
            w, b = lda.transform(X=init_input, y=tmp_labels, **kwargs)
            # Adapt the size of the weights
            w, b = _adapt_magnitude(w=w, b=b, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)
            W[:, start_index:start_index + 1] = w
            start_index += 1

        # ----------------------------------------------------------------------------------------------
        logging.info('ReLDA')
        # Compute available columns per iteration
        available_columns_per_iteration = len(classes) - 1
        # Compute number of iterations
        N = int((module.weight.shape[0] - start_index) / available_columns_per_iteration)
        # Leave some room for PCA, if there is some
        if N > 4:
            N -= 2

        # N times
        tmp_input = init_input.copy()
        tmp_labels = init_labels.copy()
        clf = LinearDiscriminantAnalysis(solver=kwargs['solver'])
        initial_size = len(init_input)
        for i in range(N):
            logging.info(f'Iteration {i + 1} of {N} #samples={len(tmp_input)}')
            if len(tmp_input) < 1:
                logging.info('No more wrong samples -> exiting loop')
                break
            if len(tmp_input) == len(np.unique(tmp_labels)):
                logging.info('Number of samples is equal to the number of classes -> exiting loop')
                break
            logging.info(f'\tfitting...')
            clf.fit(X=tmp_input, y=tmp_labels)
            w = -clf.scalings_
            b = np.mean(tmp_input, axis=0)
            # Adapt the size of the weights
            w, b = _adapt_magnitude(w=w, b=b, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)

            # Filter the input data with the wrong predictions i.e. keep those location where it is WRONGLY predicted
            logging.info(f'\tpredicting...')
            predictions = clf.predict(tmp_input)
            locs = np.where(predictions != tmp_labels)
            logging.info(f'\tcurrent acc={1 - float(len(locs[0])) / len(tmp_input):.2f} ...')
            tmp_input = tmp_input[locs]
            tmp_labels = tmp_labels[locs]
            logging.info(f'\tglobal integrated acc={(initial_size - len(tmp_input)) / initial_size:.2f} ...')

            # Set main weights
            W[:, start_index:start_index + w.shape[1]] = w  # w could be smaller than |C|-1 as the init samples shrink
            start_index += w.shape[1]

        # ----------------------------------------------------------------------------------------------
        # PCA Part
        logging.info('PCA Transform')
        p, c = pca.transform(init_input)
        p, c = _adapt_magnitude(w=p, b=c, normalize=conv_normalize, standardize=conv_standardize, scale=conv_scale)

        # Compute available columns
        available_columns = module.weight.shape[0] - start_index

        # Add PCA columns in the second part
        W[:, start_index:] = p[:, 0:available_columns]
        start_index += available_columns

        # ----------------------------------------------------------------------------------------------

        # Adapt the size of the weights
        W, B = _basic_conv_procedure(W, B, module, **kwargs)
        return torch.from_numpy(W), torch.from_numpy(B)

    ##################################################################
    if type(module) is nn.Linear:
        return _initialize_classifier(module=module, init_input=init_input, init_labels=init_labels, **kwargs)



