# Utils
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from torchvision import transforms

from datasets.custom import transforms as custom_transforms
from datasets.custom.transforms import TwinRandomCrop, OnlyImage, OnlyTarget
# Gale
from datasets.image_folder_segmentation import ImageFolderSegmentationDataset
from datasets.util.dataset_analytics import compute_mean_std
from datasets.util.dataset_integrity import verify_dataset_integrity
from template.runner.base.base_setup import BaseSetup
from template.runner.semantic_segmentation_DIVAHisDB.util.dataset_analytics import \
    _get_class_frequencies_weights_segmentation


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
        return compute_mean_std(input_folder=os.path.join(input_folder, 'train'), **kwargs)

    @classmethod
    def _measure_weights(cls, gt_images, **kwargs):
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
        return _get_class_frequencies_weights_segmentation(gt_images=gt_images, **kwargs)

    @classmethod
    def get_split(cls, **kwargs):
        return ImageFolderSegmentationDataset(**kwargs)

    # TRANSFORMATIONS
    @classmethod
    def set_up_transforms(cls, train_ds, val_ds, test_ds, **kwargs):
        super().set_up_transforms(train_ds, val_ds, test_ds, **kwargs)
        for ds in [train_ds, val_ds, test_ds]:
            if ds is not None:
                ds.twin_transform = cls.get_test_transform(**kwargs)

    @classmethod
    def get_train_transform(cls, crop_size, **kwargs):
        mean, std = cls.load_mean_std_from_file(**kwargs)
        return OnlyImage(transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.ToPILImage()]))

    @classmethod
    def get_test_transform(cls, **kwargs):
        mean, std = cls.load_mean_std_from_file(**kwargs)
        return OnlyImage(transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std),
                                             transforms.ToPILImage()]))

    @classmethod
    def get_target_transform(cls, **kwargs):
        trains_ds, _, _ = cls._get_datasets(**kwargs)
        return OnlyTarget(transforms.Compose([
            # transforms the gt image into a one-hot encoded matrix
            custom_transforms.OneHotEncodingDIVAHisDB(class_encodings=trains_ds.class_encodings),
            # transforms the one hot encoding to argmax labels -> for the cross-entropy criterion
            custom_transforms.OneHotToPixelLabelling()]))

    @classmethod
    def get_twin_transformations(cls, **kwargs):
        return TwinRandomCrop(crop_size=kwargs['crop_size'])

    ###################

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
        # Load the datasets
        train_ds, val_ds, test_ds = cls._get_datasets(**kwargs)

        # Create the analytics csv
        cls.create_analytics_csv(train_ds=train_ds, **kwargs)

        # set class encodings
        class_encodings = cls.load_class_encodings_from_file(**kwargs)
        for ds in [train_ds, val_ds, test_ds]:
            ds.class_encodings = class_encodings
            ds.num_classes = len(class_encodings)

        # Setup transforms
        logging.info('Setting up transforms')
        cls.set_up_transforms(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, class_encodings=class_encodings, **kwargs)

        # Get the dataloaders
        train_loader, val_loader, test_loader = cls._dataloaders_from_datasets(train_ds=train_ds,
                                                                               val_ds=val_ds,
                                                                               test_ds=test_ds,
                                                                               **kwargs)
        logging.info("Dataset loaded successfully")

        verify_dataset_integrity(**kwargs)

        return train_loader, val_loader, test_loader

    @classmethod
    def create_analytics_csv(cls, input_folder, train_ds, **kwargs):
        # If it already exists your job is done
        if (Path(input_folder) / "analytics.csv").is_file():
            return

        logging.warning(f'Missing analytics.csv file for dataset located at {input_folder}')
        logging.warning(f'Attempt creating analytics.csv file')

        # Measure mean and std on train images
        logging.info(f'Measuring mean and std on train images')

        mean, std = cls._measure_mean_std(
            input_folder=input_folder, train_ds=train_ds, **kwargs
        )

        # Measure weights for class balancing
        logging.info(f'Measuring class wrights')
        # create a list with all gt file paths
        file_names_all = np.asarray([item[1] for item in train_ds.img_paths])
        file_names_gt = np.asarray([f for f in file_names_all if '/gt/' in f])
        class_weights, class_ints = cls._measure_weights(
            gt_images=file_names_gt, **kwargs
        )

        # Save results as CSV file in the dataset folder
        logging.info(f'Saving to analytics.csv')
        df = pd.DataFrame([mean, std, class_weights, class_ints])
        df.index = ['mean[RGB]', 'std[RGB]', 'class_frequencies_weights[num_classes]', 'class_encodings']
        df.to_csv(os.path.join(input_folder, 'analytics.csv'), header=False)
        logging.warning(f'Created analytics.csv file for dataset located at {input_folder}')

    @classmethod
    def load_class_encodings_from_file(cls, input_folder, **kwargs):
        """
        This function simply recovers class_encodings from the analytics.csv file

        Parameters
        ----------
        input_folder : string
            Path that points to the three folder train/val/test. Example: ~/../../data/svhn
        inmem : boolean
            Flag: if False, the dataset is loaded in an online fashion i.e. only file names are stored and images are loaded
            on demand. This is slower than storing everything in memory.
        workers : int
            Number of workers to use for the mean/std computation

        Returns
        -------
        ndarray[double]
            Class encodings for the selected dataset, contained in the analytics.csv file.
        """
        # Loads the analytics csv
        if not (Path(input_folder) / "analytics.csv").exists():
            raise SystemError(f"Analytics file not found in '{input_folder}'")
        csv_file = pd.read_csv(Path(input_folder) / "analytics.csv", header=None)
        # Extracts the class encodings
        for row in csv_file.values:
            if 'class_encodings' in str(row[0]).lower():
                class_encodings = np.array([x for x in row[1:] if str(x) != 'nan'], dtype=float)
        if 'class_encodings' not in locals():
            logging.error("Class weights not found in analytics.csv")
            raise EOFError
        return class_encodings
