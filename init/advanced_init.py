"""
Here are defined the different versions of advanced initialization techniques.
"""


import logging
import math
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import nn

from init.util import lda


def _normalize_weights(w, b):
    """
    Given both matrices of bias and weights this function return them normalized between [-1, 1]
    Parameters
    ----------
    w: nd.array 2D
        Weight matrix
    b : nd.array 1D
        Bias matrix

    Returns
    -------
        B and W normalized between [-1, 1]
    """
    max_value = max(np.max(np.abs(b)), np.max(np.abs(w)))
    w = w / max_value
    b = b / max_value
    return w, b


def _scale_weights(w, b):
    """
    Given both matrices of bias and weights this function return them scaled such that the
    standard deviation is scaled accordging to Kaiming He observation, which works best with ReLU
    See more: https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

    Parameters
    ----------
    w: nd.array 2D
        Weight matrix
    b : nd.array 1D
        Bias matrix

    Returns
    -------
        B and W normalized between [-1, 1]
    """
    w = w * math.sqrt(2 / w.shape[0])
    b = b * math.sqrt(2 / w.shape[0])
    return w, b


def _standardize_weights(w, b):
    """
    Given both matrices of bias and weights this function return them standardized with 0 mean
    and unit variance

    Parameters
    ----------
    w: nd.array 2D
        Weight matrix
    b : nd.array 1D
        Bias matrix

    Returns
    -------
        B and W standard normalized
    """
    joint_matrices = np.concatenate((w, np.expand_dims(b, axis=1)), axis=1)
    mean = np.mean(joint_matrices)
    std = np.std(joint_matrices)
    w = (w - mean) / std
    b = (b - mean) / std
    return w, b

def _fit_weights_size(w, module):
    """
    Correct the sizes of W  according to the expected size of the module
    :param w: nd.array 2D
        Weight matrix
    :return:
        W with theirs size fit to expected values of the module
    """

    out_size = module.out_channels if 'conv' in str(type(module)) else module.out_features

    # Add default values columns when num_columns < num_desired_dimensions
    if out_size > w.shape[1]:
        logging.warning("Init weight matrix smaller than the number of neurons (too few columns). "
                        "Expected {} got {}. "
                        "Filling missing dimensions with default values. "
                        .format(out_size, w.shape[1]))
        default_values = module.weight.data.numpy()
        if 'conv' in str(type(module)):
            default_values = _flatten_conv_filters(default_values)
        else:
            default_values = default_values.T  # Linear Layers have neurons x input_dimensions
        w = np.hstack((w, default_values[:, w.shape[1]:]))

    # Discard extra columns when num_columns > num_desired_dimensions
    if out_size < w.shape[1]:
        logging.warning("Init weight matrix bigger than the number of neurons. "
                        "Expected {} got {}. "
                        "Removing extra columns. ".format(out_size, w.shape[1]))
        w = w[:, :out_size]

    return w

def _flatten_conv_filters(filters):
    """
    Takes the conv filters and flatten them.
    E.g. filters(24x3x5x5) turn into filters(75x24)

    Parameters:
    -----------
    :param filters: matrix 4D
        conv filters 4D: num_filters x in_channels x kernel_size x kernel_size
    :return:
        filters flattened
    """
    assert len(filters.shape) > 2
    return filters.T.reshape(-1, filters.shape[0])


def _reshape_flattened_conv_filters(filters, kernel_filter_size):
    """
    Reshape flattened conv filters back to a conv shape
    E.g. filter(75x24) turn into filters(25x3x5x5) with the kernel_size param as 5

    Parameters:
    -----------
    :param filters: matrix 2D
        flattened conv filters
    :param kernel_filter_size: int
        size of the square filters kernel e.g. 3x3 or 5x5
    :return:
        the filters in a conv shape 4D: num_filters x in_channels x kernel_size x kernel_size
    """
    return filters.T.reshape(filters.shape[1], -1, kernel_filter_size, kernel_filter_size)


def _basic_procedure(w, b, module):
    """
    TODO 
    
    Parameters
    ----------
    w
    b
    module

    Returns
    -------

    """
    # Check size of W
    w = _fit_weights_size(w, module)

    # Set B to be -W*mean(X) such that it centers the data
    b = -np.matmul(w.T, b)
    assert b.shape == module.bias.shape

    # Reshape

    # CONV LAYER
    if type(module) is nn.Conv2d:
        w = _reshape_flattened_conv_filters(w, module.kernel_size[0])

    # LINEAR LAYER
    if type(module) is nn.Linear:
        w = w.T  # Linear Layers have neurons x input_dimensions

    assert (w.shape == module.weight.shape)

    # Set W by adding it to the current random values
    # sn_ratio = 1
    # w = module.weight.data.numpy() / sn_ratio + w

    return w, b


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def pure_lda(layer_index, init_input, init_labels, model, module, **kwargs):
    network_depth = len(list(list(model.children())[0].children()))

    """ Empirical notes 


    Flowers , LDA_simple
    --------------------

        ONLY FIRST LAYER 
        > (scale - none) and (normalize - none) are terrible on all levels
        > All configuration which involve scale have the highest training loss by several orders of magnitude
        > (standard - none) is the best and adding normalise or scale still "kind of" works but its lowering the 
            performance on val/test

        ONLY DISC
        > (none _ Normalize Standardize Scale) is exactly the same as (none _ Standardize Scale)
        > (none _ Normalize) has the highest performance on validation and the lowest loss on train. It is also the one
            that gets destroyed the least by the first/second epoch of training
        > (none _ Normalize Scale) competes with (none _ Normalize) but it has a very slightly higher train loss and 
            it gets slightly more destroyed the least by the first/second epoch of training. However, it achieves a few
            percentages more on both validation and test    

        BOTH 
        > (standard _ Normalize Scale) is achieving similar performances as (none _ Normalize Scale) on validation but 
            performs much better on the training set (20% higher, around 80% acc)
        > (standard _ Normalize Standardize Scale) is making 99.6% accuracy on the train set at START and score a low 
            30% on validation (well below the 50% of (standard _ Normalize Scale) counterpart). Moreover, it stays at 
            such high training performances with a ridiculously small loss on train set 
            
        OTHERS
        > Initial experimentation seems to favor Swish over Softsign
    """
    ###################################################################################################################
    # All layers but the last one
    if layer_index < network_depth:
        logging.info('LDA Transform')
        W, B = lda.transform(X=init_input, y=init_labels)

        # Normalize the weights
        # W, B = _normalize_weights(W, B) commented on purpose, see notes above
        W, B = _standardize_weights(W, B)
        # W, B = _scale_weights(W, B) commented on purpose, see notes above

        W, B = _basic_procedure(W, B, module)

    ###################################################################################################################
    # Last layer
    else:
        logging.info('LDA Discriminants')
        W, B = lda.discriminants(X=init_input, y=init_labels)

        # Normalize the weights
        W, B = _normalize_weights(W, B)
        W, B = _standardize_weights(W, B)
        W, B = _scale_weights(W, B)

    return torch.Tensor(W), torch.Tensor(B)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def mirror_lda(layer_index, init_input, init_labels, model, module, module_type, **kwargs):
    ###################################################################################################################
    # All layers but the last one
    if layer_index != len(list(model.children())) - 1:
        logging.info('LDA Transform')
        W, B = lda.transform(X=init_input, y=init_labels)

        # Compute available columns
        available_columns = module.weight.data.shape[0]

        # Discard dimensions as necessary
        if W.shape[1] * 2 > available_columns:
            W = W[:, 0:int(available_columns / 2)]

        # Mirror W. You don't mirror B because it has to be size of mean(X)
        W = np.hstack((W, -W))

        W, B = _basic_procedure(W, B, module)

    ###################################################################################################################
    # Last layer
    else:
        logging.info('LDA Discriminants')
        W, B = lda.discriminants(X=init_input, y=init_labels)

        # Normalize the weights
        W, B = _normalize_weights(W, B)

    return torch.Tensor(W), torch.Tensor(B)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def highlander_lda(layer_index, init_input, init_labels, model, module, module_type, **kwargs):
    ###################################################################################################################
    # All layers but the last one
    if layer_index != len(list(model.children())) - 1:
        logging.info('LDA Transform')

        # Check if size of model allows (has enough neurons)
        if module.weight.data.shape[0] < len(np.unique(init_labels)) * 2:
            logging.error("Model does not have enough neurons. Expected at least |C|*2 got {}"
                          .format(module.weight.data.shape[0]))
            sys.exit(-1)

        # Compute available columns per class
        available_columns = int(module.weight.data.shape[0] / len(np.unique(init_labels)))

        # Init W and B with the default values
        W = _flatten_conv_filters(module.weight.data.numpy())
        W = np.zeros(W.shape)
        B = module.bias.data.numpy()
        classes = np.unique(init_labels)

        # |C| times
        for i, l in enumerate(classes):
            # Make a new set of labels of 1 vs ALL
            tmp_labels = init_labels.copy()
            tmp_labels[np.where(init_labels == l)] = 0
            tmp_labels[np.where(init_labels != l)] = 1
            w, b = lda.transform(X=init_input, y=tmp_labels)

            start_index = i * available_columns
            end_index = start_index + available_columns
            W[:, start_index:end_index] = w[:, 0:available_columns]
            B = b

        # w, b = lda.transform(X=init_input, y=init_labels)
        # W[classes.size*2:classes.size*2+2, :] = w
        # B[classes.size*2:classes.size*2+2] = b

        W, B = _basic_procedure(W, B, module)

    ###################################################################################################################
    # Last layer
    else:
        logging.info('LDA Discriminants')
        W, B = lda.discriminants(X=init_input, y=init_labels)

        # Normalize the weights
        W, B = _normalize_weights(W, B)

    return torch.Tensor(W), torch.Tensor(B)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def lpca(layer_index, init_input, init_labels, model, module, module_type, **kwargs):
    ###################################################################################################################
    # All layers but the last one
    if layer_index != len(list(model.children())) - 1:
        logging.info('LDA Transform')
        w, B = lda.transform(X=init_input, y=init_labels)
        logging.info('PCA Transform')
        pca = PCA().fit(init_input)
        p = pca.components_.T  # Don't even think about touching this T!
        c = pca.mean_

        # Init W with zeros of same shape of real weight matrix
        W = np.zeros(_flatten_conv_filters(module.weight.data.numpy()).shape)

        # Compute available columns
        half_available_columns = int(module.weight.data.shape[0]/2)

        # Add LDA columns in the first half
        W[:, 0:half_available_columns] = w[:, 0:half_available_columns]

        # Add PCA columns in the second half
        W[:, half_available_columns:2*half_available_columns] = p[:, 0:half_available_columns]

        W, B = _basic_procedure(W, B, module)

    ###################################################################################################################
    # Last layer
    else:
        logging.info('LDA Discriminants')
        W, B = lda.discriminants(X=init_input, y=init_labels)

        # Normalize the weights
        W, B = _normalize_weights(W, B)

    return torch.Tensor(W), torch.Tensor(B)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def reverse_pca(layer_index, init_input, init_labels, model, module, module_type, **kwargs):
    ###################################################################################################################
    # All layers but the last one
    if layer_index != len(list(model.children())) - 1:

        # Check if size of model allows (has enough neurons)
        if module.weight.data.shape[0] < len(np.unique(init_labels)) * 2:
            logging.error("Model does not have enough neurons. Expected at least |C|*2 got {}"
                          .format(module.weight.data.shape[0]))
            sys.exit(-1)

        # Compute available columns per class
        available_columns = int(module.weight.data.shape[0] / len(np.unique(init_labels)))
        bias_available_columns = int(init_input.shape[1] / len(np.unique(init_labels)))

        # Init W and B with zeros
        W = np.zeros(_flatten_conv_filters(module.weight.data.numpy()).shape)
        B = np.zeros(init_input.shape[1])  # Bias should be size of input because its later multiplied by -Wx
        classes = np.unique(init_labels)

        # |C| times
        for i, l in enumerate(classes):
            logging.info('Iteration of class {}'.format(l))
            # Make a new set of labels of 1 vs ALL
            pca = PCA().fit(init_input[np.where(init_labels == l)])
            p = pca.components_.T  # Don't even think about touching this T!
            c = pca.mean_

            start_index = i * available_columns
            end_index = start_index + available_columns
            W[:, start_index:end_index] = p[:, -available_columns:]
            #W[:, start_index:end_index] = p[:, 0:available_columns]

            start_index = i * bias_available_columns
            end_index = start_index + bias_available_columns
            B[start_index:end_index] = c[-bias_available_columns:]
            #B[start_index:end_index] = c[0:bias_available_columns]

        W, B = _basic_procedure(W, B, module)

    ###################################################################################################################
    # Last layer
    else:
        logging.info('LDA Discriminants')
        W, B = lda.discriminants(X=init_input, y=init_labels)

        # Normalize the weights
        W, B = _normalize_weights(W, B)

    return torch.Tensor(W), torch.Tensor(B)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def relda(layer_index, init_input, init_labels, model, module, module_type, **kwargs):
    ###################################################################################################################
    # All layers but the last one
    if layer_index != len(list(model.children())) - 1:

        # Check if size of model allows (has enough neurons)
        if module.weight.data.shape[0] < len(np.unique(init_labels)) * 2:
            logging.error("Model does not have enough neurons. Expected at least |C|*2 got {}"
                          .format(module.weight.data.shape[0]))
            sys.exit(-1)

        # Compute available columns per class
        available_columns = int(module.weight.data.shape[0] / len(np.unique(init_labels)))
        bias_available_columns = int(init_input.shape[1] / len(np.unique(init_labels)))

        # Init W and B with zeros
        W = np.zeros(_flatten_conv_filters(module.weight.data.numpy()).shape)
        B = np.zeros(init_input.shape[1])  # Bias should be size of input because its later multiplied by -Wx
        classes = np.unique(init_labels)

        # |C| times
        logging.info('LDA Transform (sklearn)')
        clf = LinearDiscriminantAnalysis(solver='svd')
        for i, l in enumerate(classes):
            logging.info('Iteration of class {}, n={}'.format(l, len(init_labels)))
            if len(init_input) < 1:
                logging.info('No more wrong samples, exiting loop')
                break

            clf.fit(X=init_input, y=init_labels)
            w = -clf.scalings_
            b = np.mean(init_input, axis=0)

            # Filter the input data with the wrong predictions
            predictions = clf.predict(init_input)
            locs = np.where(predictions != init_labels)
            init_input = init_input[locs]
            init_labels = init_labels[locs]

            # Set main weights
            start_index = i * available_columns
            end_index = start_index + np.min([available_columns, w.shape[1]])
            W[:, start_index:end_index] = w[:, 0:available_columns]

            # Set bias
            start_index = i * bias_available_columns
            end_index = start_index + bias_available_columns
            B[start_index:end_index] = b[0:bias_available_columns]

        W, B = _basic_procedure(W, B, module)

    ###################################################################################################################
    # Last layer
    else:
        logging.info('LDA Discriminants')
        W, B = lda.discriminants(X=init_input, y=init_labels)

        # Normalize the weights
        W, B = _normalize_weights(W, B)

    return torch.Tensor(W), torch.Tensor(B)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def trimlda(layer_index, init_input, init_labels, model, module, module_type, **kwargs):
    ###################################################################################################################
    # All layers but the last one
    if layer_index != len(list(model.children())) - 1:

        init_input, init_labels = _filter_points_trimlda(init_input, init_labels)

        logging.info('LDA Transform')
        w, B = lda.transform(X=init_input, y=init_labels)
        logging.info('PCA Transform')
        pca = PCA().fit(init_input)
        p = pca.components_.T  # Don't even think about touching this T!
        c = pca.mean_

        # Init W with zeros of same shape of real weight matrix
        W = np.zeros(_flatten_conv_filters(module.weight.data.numpy()).shape)

        # Compute available columns
        half_available_columns = int(module.weight.data.shape[0] / 2)

        # Add LDA columns in the first half
        W[:, 0:half_available_columns] = w[:, 0:half_available_columns]

        # Add PCA columns in the second half
        W[:, half_available_columns:2 * half_available_columns] = p[:, 0:half_available_columns]

        W, B = _basic_procedure(W, B, module)

    ###################################################################################################################
    # Last layer
    else:
        logging.info('LDA Discriminants')

        init_input, init_labels = _filter_points_trimlda(init_input, init_labels)

        # Compute the values with the final set of points retained
        W, B = lda.discriminants(X=init_input, y=init_labels)

        # Normalize the weights
        W, B = _normalize_weights(W, B)

    return torch.Tensor(W), torch.Tensor(B)


def _filter_points_trimlda(init_input, init_labels):

    logging.info('Filter points with trim-lda')

    clf = LinearDiscriminantAnalysis(solver='svd')

    # Initialize to full list
    input = init_input
    labels = init_labels

    for i in range(1, 30):
        clf.fit(X=input, y=labels)

        # Filter the input data with the wrong predictions on the full data
        predictions = clf.predict(init_input)
        locs = np.where(predictions == init_labels)
        logging.info("Size of locs{}: {}".format(i, len(locs[0])))
        input = init_input[locs]
        labels = init_labels[locs]

    return input, labels