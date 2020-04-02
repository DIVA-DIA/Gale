# Utils
import logging
import os
import sys

import numpy as np
import pandas as pd
from torchvision import transforms

# Gale
from datasets.image_folder_segmentation import ImageFolderSegmentationDataset, load_class_encodings
from datasets.util.dataset_integrity import verify_dataset_integrity
from template.runner.semantic_segmentation_DIVAHisDB.util.dataset_analytics import compute_mean_std_segmentation, \
    _get_class_frequencies_weights_segmentation
from template.runner.base.base_setup import BaseSetup
from datasets.custom.transforms import TwinRandomCrop
from datasets.custom import transforms as custom_transforms


class SemanticSegmentationSetupHisDB(BaseSetup):

    @classmethod
    def _measure_mean_std(cls, input_folder, **kwargs):
        """Computes mean and std of train images

        Parameters
        ----------
        input_folder : string
            Path string that points to the three folder train/val/test. Example: ~/../../data/svhn

        Returns
        -------
        mean : ndarray[double]
            Mean value (for each channel) of all pixels of the images in the input folder
        std : ndarray[double]
            Standard deviation (for each channel) of all pixels of the images in the input folder
        """
        return compute_mean_std_segmentation(input_folder=os.path.join(input_folder, 'train'), **kwargs)

    @classmethod
    def _measure_weights(cls, input_folder, **kwargs):
        """Computes the class balancing weights (not the frequencies!!)

        Parameters
        ----------
        input_folder : string
            Path string that points to the three folder train/val/test. Example: ~/../../data/svhn

        Returns
        -------
        class_weights : ndarray[double]
            Weight for each class in the train set (one for each class)
        """
        return _get_class_frequencies_weights_segmentation(input_folder=os.path.join(input_folder, 'train'), **kwargs)

    @classmethod
    def get_split(cls, **kwargs):
        classes = load_class_encodings(**kwargs)
        test = 'test' in kwargs['path']
        return ImageFolderSegmentationDataset(class_encodings=classes, is_test=test, **kwargs)

    @classmethod
    def get_train_transform(cls, crop_size, **kwargs):
        mean, std = cls.load_mean_std_from_file(**kwargs)
        return transforms.Compose([TwinRandomCrop(crop_size=crop_size), transforms.Normalize(mean=mean, std=std)])

    @classmethod
    def get_test_transform(cls, **kwargs):
        return cls.get_train_transform(**kwargs)

    @classmethod
    def get_target_transform(cls, **kwargs):
        trains_ds, _, _ = cls._get_datasets(**kwargs)
        return transforms.Compose([
            # transforms the gt image into a one-hot encoded matrix
            custom_transforms.OneHotEncodingDIVAHisDB(class_encodings=trains_ds.class_encodings),
            # transforms the one hot encoding to argmax labels -> for the cross-entropy criterion
            custom_transforms.OneHotToPixelLabelling()])

    @staticmethod
    def output_to_class_encodings(output, class_encodings, perform_argmax=True):
        """
        This function converts the output prediction matrix to an image like it was provided in the ground truth

        Parameters
        -------
        output : np.array of size [#C x H x W]
            output prediction of the network for a full-size image, where #C is the number of classes
        class_encodings : List
            Contains the range of encoded classes
        perform_argmax : bool
            perform argmax on input data
        Returns
        -------
        numpy array of size [C x H x W] (BGR)
        """

        B = np.argmax(output, axis=0) if perform_argmax else output

        class_to_B = {i: j for i, j in enumerate(class_encodings)}

        masks = [B == old for old in class_to_B.keys()]

        for mask, (old, new) in zip(masks, class_to_B.items()):
            B = np.where(mask, new, B)

        rgb = np.dstack((np.zeros(shape=(B.shape[0], B.shape[1], 2), dtype=np.int8), B))

        return rgb

    # Dataloaders handling
    @classmethod
    def set_up_dataloaders(cls, **kwargs):
        """ Set up the dataloaders for the specified datasets.

        Returns
        -------
        train_loader : torch.utils.data.DataLoader
        val_loader : torch.utils.data.DataLoader
        test_loader : torch.utils.data.DataLoader
            Dataloaders for train, val and test.
        int
            Number of classes for the model.
        """
        logging.info('Loading {} from:{}'.format(
            os.path.basename(os.path.normpath(kwargs['input_folder'])),
            kwargs['input_folder'])
        )

        # Load the datasets
        train_ds, val_ds, test_ds = cls._get_datasets(**kwargs)

        # Create the analytics csv
        cls.create_analytics_csv(train_ds=train_ds, **kwargs)

        # Setup transforms
        logging.info('Setting up transforms')
        cls.set_up_transforms(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, **kwargs)

        # Get the dataloaders
        train_loader, val_loader, test_loader = cls._dataloaders_from_datasets(train_ds=train_ds,
                                                                               val_ds=val_ds,
                                                                               test_ds=test_ds,
                                                                               **kwargs)
        logging.info("Dataset loaded successfully")

        verify_dataset_integrity(**kwargs)

        return train_loader, val_loader, test_loader
