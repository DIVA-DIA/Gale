import argparse
import inspect
import json
import os
import re
import shutil
import sys
import urllib
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
import torchvision
from PIL import Image

from datasets.util.dataset_splitter import split_dataset
from util.misc import make_folder_if_not_exists


def mnist(output_folder, **kwargs):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the MNIST dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    torchvision.datasets.MNIST(root=output_folder, download=True)

    # Load the data into memory
    train_data, train_labels = torch.load(os.path.join(output_folder,
                                                       'MNIST',
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(os.path.join(output_folder,
                                                     'MNIST',
                                                     'processed',
                                                     'test.pt'))

    # Make output folders
    dataset_root = os.path.join(output_folder, 'MNIST')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(train_folder)
    make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels.detach().numpy())):
            dest = os.path.join(folder, str(label))
            make_folder_if_not_exists(dest)
            Image.fromarray(img.numpy(), mode='L').save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    shutil.rmtree(os.path.join(output_folder, 'MNIST', 'raw'))
    shutil.rmtree(os.path.join(output_folder, 'MNIST', 'processed'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def svhn(output_folder, **kwargs):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the SVHN dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    from scipy.io import loadmat as _loadmat

    # Use torchvision to download the dataset
    torchvision.datasets.SVHN(root=output_folder, split='train', download=True)
    torchvision.datasets.SVHN(root=output_folder, split='test', download=True)

    # Load the data into memory
    train = _loadmat(os.path.join(output_folder,
                                  'train_32x32.mat'))
    train_data, train_labels = train['X'], train['y'].astype(np.int64).squeeze()
    np.place(train_labels, train_labels == 10, 0)
    train_data = np.transpose(train_data, (3, 0, 1, 2))

    test = _loadmat(os.path.join(output_folder,
                                 'test_32x32.mat'))
    test_data, test_labels = test['X'], test['y'].astype(np.int64).squeeze()
    np.place(test_labels, test_labels == 10, 0)
    test_data = np.transpose(test_data, (3, 0, 1, 2))

    # Make output folders
    dataset_root = os.path.join(output_folder, 'SVHN')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(train_folder)
    make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            make_folder_if_not_exists(dest)
            Image.fromarray(img).save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    os.remove(os.path.join(output_folder, 'train_32x32.mat'))
    os.remove(os.path.join(output_folder, 'test_32x32.mat'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def cifar10(output_folder, **kwargs):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the CIFAR dataset to the location specified
    on the file system

    Parameters
    ----------
    output_folder : str
        Path to folder where to put the dataset

    Returns
    -------
        None
    """
    # Make output folders
    dataset_root = os.path.join(output_folder, 'CIFAR10')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    if Path(dataset_root).exists():
        print(f"Path ({dataset_root}) already exists. Nothing done")
        return

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(train_folder)
    make_folder_if_not_exists(test_folder)

    # Use torchvision to download the dataset
    cifar_train = torchvision.datasets.CIFAR10(root=output_folder, train=True, download=True)
    cifar_test = torchvision.datasets.CIFAR10(root=output_folder, train=False, download=True)

    # Load the data into memory
    train_data, train_labels = cifar_train.data, cifar_train.targets

    test_data, test_labels = cifar_test.data, cifar_test.targets

    # Replace numbers with text for class names
    class_names_mapping = {0: 'plane', 1: 'car', 2: 'bird', 3: ' cat', 4: 'deer',
                           5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    train_labels = [class_names_mapping[l] for l in train_labels]
    test_labels = [class_names_mapping[l] for l in test_labels]

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            make_folder_if_not_exists(dest)
            Image.fromarray(img).save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    os.remove(os.path.join(output_folder, 'cifar-10-python.tar.gz'))
    shutil.rmtree(os.path.join(output_folder, 'cifar-10-batches-py'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def COCO(output_folder, **kwargs):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the COCO dataset to the location specified
    on the file system.

    It fetches:

    Parameters
    ----------
    output_folder : str
        Path to folder where to put the dataset

    Returns
    -------
        None
    """
    URLS = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
        "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
    ]

    root = os.path.join(output_folder, "COCO")

    # download files
    for url in URLS:
        print('Downloading {}'.format(url))
        filename = url.rpartition('/')[2]
        file_path = os.path.join(root, filename)
        download_url(url, file_path)
        extract_zip(zip_path=file_path, root=root, remove_finished=True)

    # extract panoptic and stuff annotations
    annotation_path = os.path.join(root, "annotations")
    extract_zip(zip_path=os.path.join(annotation_path, "panoptic_train2017.zip"), root=annotation_path,
                remove_finished=True)
    extract_zip(zip_path=os.path.join(annotation_path, "panoptic_val2017.zip"), root=annotation_path,
                remove_finished=True)
    extract_zip(zip_path=os.path.join(annotation_path, "stuff_train2017_pixelmaps.zip"), root=annotation_path,
                remove_finished=True)
    extract_zip(zip_path=os.path.join(annotation_path, "stuff_val2017_pixelmaps.zip"), root=annotation_path,
                remove_finished=True)


def LVIS(output_folder, **kwargs):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the COCO dataset to the location specified
    on the file system.

    It fetches:

    Parameters
    ----------
    output_folder : str
        Path to folder where to put the dataset

    Returns
    -------
        None
    """
    URLS = [
    ]

    root = os.path.join(output_folder, "LVIS")

    # download files
    for url in URLS:
        print('Downloading {}'.format(url))
        filename = url.rpartition('/')[2]
        file_path = os.path.join(root, filename)
        download_url(url, file_path)
        extract_zip(zip_path=file_path, root=root, remove_finished=True)

    # extract panoptic and stuff annotations
    annotation_path = os.path.join(root, "annotations")
    # extract_zip(zip_path=os.path.join(annotation_path, "panoptic_train2017.zip"), root=annotation_path, remove_finished=True)
    # extract_zip(zip_path=os.path.join(annotation_path, "panoptic_val2017.zip"), root=annotation_path, remove_finished=True)
    # extract_zip(zip_path=os.path.join(annotation_path, "stuff_train2017_pixelmaps.zip"), root=annotation_path, remove_finished=True)
    # extract_zip(zip_path=os.path.join(annotation_path, "stuff_val2017_pixelmaps.zip"), root=annotation_path, remove_finished=True)


def OpenImages(output_folder, **kwargs):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the COCO dataset to the location specified
    on the file system.

    It fetches:

    Parameters
    ----------
    output_folder : str
        Path to folder where to put the dataset

    Returns
    -------
        None
    """
    root = os.path.join(output_folder, "OpenImages")
    root_images = os.path.join(root, "images")
    root_annotations = os.path.join(root, "annotations")
    root_metadata = os.path.join(root, "metadata")
    for p in [root, root_images, root_annotations, root_metadata]:
        if not os.path.exists(p):
            os.makedirs(p)
    for s in ['train', 'validation', 'test']:
        if not os.path.exists(os.path.join(root_annotations, s, "masks")):
            os.makedirs(os.path.join(root_annotations, s, "masks"))

    # IMAGES
    # Train
    # if not os.path.exists(os.path.join(root_images, "train")):
    #     train_url = "https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/train_0{}.zip"
    #     for i in range(9):
    #         url = train_url.format(i)
    #         print('Downloading {}'.format(url))
    #         filename = url.rpartition('/')[2]
    #         file_path = os.path.join(root, filename)
    #         download_url(url, file_path)
    #         extract_zip(zip_path=file_path, root=root_images, remove_finished=True)

    # Val and test sets
    # if not os.path.exists(os.path.join(root_images, "validation")):
    #     URLS = [
    #         "https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/validation.zip",
    #         "https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/test.zip",
    #     ]
    #     for url in URLS:
    #         print('Downloading {}'.format(url))
    #         filename = url.rpartition('/')[2]
    #         file_path = os.path.join(root, filename)
    #         download_url(url, file_path)
    #         extract_zip(zip_path=file_path, root=root_images, remove_finished=True)

    # METADATA
    URLS = [
        "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
        "https://storage.googleapis.com/openimages/v5/classes-segmentation.txt",
    ]
    for url in URLS:
        print('Downloading {}'.format(url))
        filename = url.rpartition('/')[2]
        file_path = os.path.join(root_metadata, filename)
        download_url(url, file_path)

    # ANNOTATIONS
    URLS = [
        "https://storage.googleapis.com/openimages/v5/{0}-annotations-human-imagelabels-boxable.csv",
        "https://storage.googleapis.com/openimages/2018_04/{0}/{0}-images-boxable-with-rotation.csv",
        "https://storage.googleapis.com/openimages/2018_04/{0}/{0}-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/v5/{0}-annotations-object-segmentation.csv"
    ]
    masks = "https://storage.googleapis.com/openimages/v5/{0}-masks/{0}-masks-{1}.zip"

    if not os.listdir(os.path.join(root_annotations, s, "masks")):
        # for s in ['train', 'validation', 'test']:
        for s in ['validation', 'test']:
            for url in URLS:
                url = url.format(s)
                print('Downloading {}'.format(url))
                filename = url.rpartition('/')[2]
                file_path = os.path.join(root_annotations, s, filename)
                download_url(url, file_path)

            fids = [i for i in range(10)] + ['a', 'b', 'c', 'd', 'e', 'f']
            for fid in fids:
                url = masks.format(s, fid)
                print('Downloading {}'.format(url))
                filename = url.rpartition('/')[2]
                file_path = os.path.join(root, filename)
                download_url(url, file_path)
                extract_zip(zip_path=file_path, root=os.path.join(root_annotations, s, "masks"),
                            remove_finished=True)


def download_url(url, file_path, chunk_size=4096):
    r = requests.get(url, allow_redirects=True, stream=True)

    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)


def extract_zip(zip_path, root, remove_finished=False):
    print('Extracting {}'.format(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(root)
    if remove_finished:
        os.unlink(zip_path)


def extract_list_of_classes(files):
    classes = {}
    idx_to_classes = {}
    for i, fname in enumerate(files):
        with open(fname) as f:
            d = json.load(f)
            idx_to_classes[i] = []
            for a in d["annotations"]:
                cls_name = a["name"]
                try:
                    classes[cls_name].add(i)
                except KeyError:
                    classes[cls_name] = set([i])
                if cls_name not in idx_to_classes[i]:
                    idx_to_classes[i].append(cls_name)
    return classes, idx_to_classes


def diva_hisdb(output_folder, **kwargs):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the DIVA HisDB-all dataset for semantic segmentation to the location specified
    on the file system

    See also: https://diuf.unifr.ch/main/hisdoc/diva-hisdb

    Output folder structure: ../HisDB/CB55/train
                             ../HisDB/CB55/val
                             ../HisDB/CB55/test

                             ../HisDB/CB55/test/data -> images
                             ../HisDB/CB55/test/gt   -> pixel-wise annotated ground truth

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # make the root folder
    dataset_root = os.path.join(output_folder, 'HisDB')
    make_folder_if_not_exists(dataset_root)

    # links to HisDB data sets
    link_public = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/all.zip')
    link_test_private = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/private-test/all-privateTest.zip')
    download_path_public = os.path.join(dataset_root, link_public.geturl().rsplit('/', 1)[-1])
    download_path_private = os.path.join(dataset_root, link_test_private.geturl().rsplit('/', 1)[-1])

    # download files
    print('Downloading {}...'.format(link_public.geturl()))
    urllib.request.urlretrieve(link_public.geturl(), download_path_public)

    print('Downloading {}...'.format(link_test_private.geturl()))
    urllib.request.urlretrieve(link_test_private.geturl(), download_path_private)
    print('Download complete. Unpacking files...')

    # unpack relevant folders
    zip_file = zipfile.ZipFile(download_path_public)

    # unpack imgs and gt
    data_gt_zip = {f: re.sub(r'img', 'pixel-level-gt', f) for f in zip_file.namelist() if 'img' in f}
    dataset_folders = [data_file.split('-')[-1][:-4] for data_file in data_gt_zip.keys()]
    for data_file, gt_file in data_gt_zip.items():
        dataset_name = data_file.split('-')[-1][:-4]
        dataset_folder = os.path.join(dataset_root, dataset_name)
        make_folder_if_not_exists(dataset_folder)

        for file in [data_file, gt_file]:
            zip_file.extract(file, dataset_folder)
            with zipfile.ZipFile(os.path.join(dataset_folder, file), "r") as zip_ref:
                zip_ref.extractall(dataset_folder)
                # delete zips
                os.remove(os.path.join(dataset_folder, file))

        # create folder structure
        for partition in ['train', 'val', 'test', 'test-public']:
            for folder in ['data', 'gt']:
                make_folder_if_not_exists(os.path.join(dataset_folder, partition, folder))

    # move the files to the correct place
    for folder in dataset_folders:
        for k1, v1 in {'pixel-level-gt': 'gt', 'img': 'data'}.items():
            for k2, v2 in {'public-test': 'test-public', 'training': 'train', 'validation': 'val'}.items():
                current_path = os.path.join(dataset_root, folder, k1, k2)
                new_path = os.path.join(dataset_root, folder, v2, v1)
                for f in [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]:
                    shutil.move(os.path.join(current_path, f), os.path.join(new_path, f))
            # remove old folders
            shutil.rmtree(os.path.join(dataset_root, folder, k1))

    # fix naming issue
    for old, new in {'CS18': 'CSG18', 'CS863': 'CSG863'}.items():
        os.rename(os.path.join(dataset_root, old), os.path.join(dataset_root, new))

    # unpack private test folders
    zip_file_private = zipfile.ZipFile(download_path_private)

    data_gt_zip_private = {f: re.sub(r'img', 'pixel-level-gt', f) for f in zip_file_private.namelist() if 'img' in f}

    for data_file, gt_file in data_gt_zip_private.items():
        dataset_name = re.search('-(.*)-', data_file).group(1)
        dataset_folder = os.path.join(dataset_root, dataset_name)

        for file in [data_file, gt_file]:
            zip_file_private.extract(file, dataset_folder)
            with zipfile.ZipFile(os.path.join(dataset_folder, file), "r") as zip_ref:
                zip_ref.extractall(os.path.join(dataset_folder, file[:-4]))
            # delete zip
            os.remove(os.path.join(dataset_folder, file))

        # create folder structure
        for folder in ['data', 'gt']:
            make_folder_if_not_exists(os.path.join(dataset_folder, 'test', folder))

        for old, new in {'pixel-level-gt': 'gt', 'img': 'data'}.items():
            current_path = os.path.join(dataset_folder, "{}-{}-privateTest".format(old, dataset_name), dataset_name)
            new_path = os.path.join(dataset_folder, "test", new)
            for f in [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]:
                # the ground truth files in the private test set have an additional ending, which needs to be remove
                if new == "gt":
                    f_new = re.sub('_gt', r'', f)
                else:
                    f_new = f
                shutil.move(os.path.join(current_path, f), os.path.join(new_path, f_new))

            # remove old folders
            shutil.rmtree(os.path.dirname(current_path))

    print('Finished. Data set up at {}.'.format(dataset_root))


if __name__ == "__main__":
    downloadable_datasets = [name[0] for name in inspect.getmembers(sys.modules[__name__],
                                                                    inspect.isfunction) if not name[0].startswith('_')]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download some '
                                                 'datasets and prepare them in a standard format')

    parser.add_argument('--dataset',
                        help='name of the dataset',
                        required=True,
                        type=str,
                        choices=downloadable_datasets)
    parser.add_argument('--output-folder',
                        help='path to where the dataset should be generated.',
                        required=True,
                        type=str,
                        default='./data/')
    args = parser.parse_args()

    getattr(sys.modules[__name__], args.dataset)(**args.__dict__)
