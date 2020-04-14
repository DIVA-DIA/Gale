import numpy as np


def accuracy_segmentation(label_trues, label_preds, n_class):
    """
    Taken from https://github.com/wkentaro/pytorch-fcn
    Calculates the accuracy measures for the segmentation runner

    Parameters
    ----------
    label_trues: matrix (batch size x H x W)
        contains the true class labels for each pixel
    label_preds: matrix ((batch size x H x W)
        contains the predicted class for each pixel
    n_class: int
        number possible classes
    border_pixel: boolean
        true if border pixel value should be

    Returns
    -------

    overall accuracy, mean accuracy, mean IU, fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        try:
            # the images all have the same size
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        except ValueError:
            # the images have different sizes
            hist += _fast_hist([l.flatten() for l in lt].flatten(), [l.flatten() for l in lp].flatten(), n_class)

    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc * 100, acc_cls * 100, mean_iu * 100, fwavacc * 100


def _fast_hist(label_true, label_pred, n_class):
    """
    Taken from https://github.com/wkentaro/pytorch-fcn
    Parameters
    ----------
    label_true
    label_pred
    n_class

    Returns
    -------

    """
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
