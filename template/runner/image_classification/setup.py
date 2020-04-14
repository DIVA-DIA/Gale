# Utils
import os
from pathlib import Path

from torchvision.transforms import transforms, RandomResizedCrop, RandomRotation

# Gale
from datasets.custom import OnlyImage
from datasets.generic_image_folder_dataset import ImageFolderDataset
from datasets.util.dataset_analytics import compute_mean_std, get_class_weights
from template.runner.base.base_setup import BaseSetup


class ImageClassificationSetup(BaseSetup):

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
        return get_class_weights(input_folder=os.path.join(input_folder, 'train'), **kwargs)

    @classmethod
    def get_darwin_datasets(cls, input_folder: Path, split_folder: Path, split_type: str, **kwargs):
        """
        Used darwin-py integration to loads the dataset from file system and provide
        the dataset splits for train validation and test

        Parameters
        ----------
        input_folder : Path
            Path string that points to the dataset location
        split_folder : Path
            Path to the folder containing the split txt files
        split_type : str
            Type of the split txt file to choose.

        Returns
        -------
        train_ds : data.Dataset
        val_ds : data.Dataset
        test_ds : data.Dataset
            Train, validation and test splits
        """
        assert input_folder is not None
        input_folder = Path(input_folder)
        assert input_folder.exists()
        # Point to the full path split folder
        assert split_folder is not None
        split_folder = input_folder / "lists" / split_folder
        assert split_folder.exists()

        # Select classification datasets
        from darwin.torch.dataset import ClassificationDataset
        train_ds = ClassificationDataset(
            root=input_folder, split=split_folder / (split_type + "_train.txt")
        )
        val_ds = ClassificationDataset(
            root=input_folder, split=split_folder / (split_type + "_val.txt")
        )
        test_ds = ClassificationDataset(
            root=input_folder, split=split_folder / (split_type + "_test.txt")
        )
        return train_ds, val_ds, test_ds

    @classmethod
    def get_split(cls, **kwargs):
        """ Loads a split from file system and provides the dataset

        Returns
        -------
        torch.utils.data.Dataset
            Split at the chosen path
        """
        return ImageFolderDataset(**kwargs)

    @classmethod
    def get_train_transform(
            cls,
            model_expected_input_size,
            random_resized_crop,
            random_horizontal_flip,
            color_jitter,
            rotation,
            **kwargs):
        """Set up the data transform for image classification
        Parameters
        ----------
        model_expected_input_size : tuple
           Specify the height and width that the model expects.
        random_resized_crop : bool
            Flag for applying the random resized crop
        random_horizontal_flip : bool
            Flag for applying the random horizontal flip
        color_jitter : None or List(float, float, float, float)
            If not None, specifies  the brightness, contrast, saturation and hue of the color-jitter transform
        rotation : None or float
            If not None, specifies the random rotation degrees
        """
        transform_list = []
        # Crop and scale
        if random_resized_crop:
            transform_list.append(RandomResizedCrop(model_expected_input_size))
        else:
            transform_list.append(transforms.Resize(model_expected_input_size))
        # Color Jittering
        if color_jitter is not None:
            assert len(color_jitter) == 4
            transform_list.append(transforms.ColorJitter(*color_jitter))
            # Random grayscale
            transform_list.append(transforms.RandomGrayscale(p=0.01))
        # Random horizontal flip
        if random_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        # Random rotation
        transform_list.append(RandomRotation(degrees=rotation))
        # To tensor
        transform_list.append(transforms.ToTensor())
        # Color Normalization
        mean, std = cls.load_mean_std_from_file(**kwargs)
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        return OnlyImage(transforms.Compose(transform_list))

    @classmethod
    def get_test_transform(cls, **kwargs):
        """Set up the data transform for the test split or inference"""
        return cls.get_train_transform(**kwargs)

    @classmethod
    def get_target_transform(cls, **kwargs):
        """Set up the target transform for all splits"""
        return None
