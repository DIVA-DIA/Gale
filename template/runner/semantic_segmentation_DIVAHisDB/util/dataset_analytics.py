import logging
import os

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets as datasets, transforms as transforms

from datasets.util.dataset_analytics import _cms_inmem, _cms_online


def compute_mean_std_segmentation(input_folder, inmem, workers, **kwargs):
    """
    Computes mean and std of a dataset for semantic segmentation. Saves the results as CSV file in the dataset folder.

    Parameters
    ----------
    input_folder : String (path)
        Path to the dataset folder (see above for details)
    inmem : Boolean
        Specifies whether is should be computed i nan online of offline fashion.
    workers : int
        Number of workers to use for the mean/std computation

    Returns
    -------
        None
    """

    # Getting the train dir
    traindir = os.path.join(input_folder, 'train')

    # Load the dataset file names
    train_ds = datasets.ImageFolder(traindir, transform=transforms.Compose([transforms.ToTensor()]))

    # Extract the actual file names and labels as entries
    file_names_all = np.asarray([item[0] for item in train_ds.imgs])
    file_names_gt = np.asarray([f for f in file_names_all if '/gt/' in f])
    file_names_data = np.asarray([f for f in file_names_all if '/data/' in f])

    # Compute mean and std
    mean, std = _cms_inmem(file_names_data) if inmem else _cms_online(file_names_data, workers)

    # Compute class frequencies weights
    class_frequencies_weights, class_ints = _get_class_frequencies_weights_segmentation(file_names_gt)
    # print(class_frequencies_weights)
    # Save results as CSV file in the dataset folder
    df = pd.DataFrame([mean, std, class_frequencies_weights, class_ints])
    df.index = ['mean[RGB]', 'std[RGB]', 'class_frequencies_weights[num_classes]', 'class_encodings']
    df.to_csv(os.path.join(input_folder, 'analytics.csv'), header=False)
    return mean, std


def _get_class_frequencies_weights_segmentation(gt_images, **kwargs):
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    Parameters
    ----------
    gt_images: list of strings
        Path to all ground truth images, which contain the pixel-wise label
    workers: int
        Number of workers to use for the mean/std computation

    Returns
    -------
    ndarray[double] of size (num_classes) and ints the classes are represented as
        The weights vector as a 1D array normalized (sum up to 1)
    """
    logging.info('Begin computing class frequencies weights')

    total_num_pixels = 0
    label_counter = {}

    for path in gt_images:
        img = np.array(Image.open(path))[:, :, 2].flatten()
        total_num_pixels += len(img)
        for i, j in zip(*np.unique(img, return_counts=True)):
            label_counter[i] = label_counter.get(i, 0) + j

    classes = np.array(sorted(label_counter.keys()))
    num_samples_per_class = np.array([label_counter[k] for k in classes])
    class_frequencies = (num_samples_per_class / total_num_pixels)
    logging.info('Finished computing class frequencies weights')
    logging.info('Class frequencies (rounded): {class_frequencies}'
                 .format(class_frequencies=np.around(class_frequencies * 100, decimals=2)))
    # Normalize vector to sum up to 1.0 (in case the Loss function does not do it)
    return (1 / num_samples_per_class) / ((1 / num_samples_per_class).sum()), classes