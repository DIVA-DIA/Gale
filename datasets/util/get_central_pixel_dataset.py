import argparse
import os
import random

import numpy as np
from PIL import Image


def get_all_files_in_folders_and_subfolders(root_dir=None):
    """Get all the files in a folder and sub-folders.

    Parameters
    ----------
    root_dir : str
        All files in this directory and it's sub-folders will be returned by this method.

    Returns
    -------
    paths : list of str
        List of paths to all files in this folder and it's subfolders.
    """
    paths = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            paths.append(os.path.join(path, name))
    return paths


def convert_to_single_label(gt):
    # Override text area with background
    locs = np.where(gt[:, :, 0] == 128)
    gt[locs[0], locs[1], 2] = 1

    new_img = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)

    # set deco
    any_deco_locs = np.where(gt[:, :, 2] == 6)
    gt[any_deco_locs[0], any_deco_locs[1], 2] = 4

    any_deco_locs = np.where(gt[:, :, 2] == 12)
    gt[any_deco_locs[0], any_deco_locs[1], 2] = 4

    any_deco_locs = np.where(gt[:, :, 2] == 14)
    gt[any_deco_locs[0], any_deco_locs[1], 2] = 4

    # set comment
    any_comment_locs = np.where(gt[:, :, 2] == 10)
    gt[any_comment_locs[0], any_comment_locs[1], 2] = 2

    txt_locs = np.where(gt[:, :, 2] == 8)
    # new_img[txt_locs[0], txt_locs[1]] = np.array([0,0,255])
    new_img[txt_locs[0], txt_locs[1]] = 1

    deco_locs = np.where(gt[:, :, 2] == 4)
    # new_img[deco_locs[0], deco_locs[1]] = np.array([0,255,0])
    new_img[deco_locs[0], deco_locs[1]] = 2

    comment_locs = np.where(gt[:, :, 2] == 2)
    # new_img[comment_locs[0], comment_locs[1]] = np.array([255,0,0])
    new_img[comment_locs[0], comment_locs[1]] = 3

    return new_img


def get_gt_path(path):
    # if "test" in path:
    #     # This is a dirty fix for DIVA-HisDB.
    #     return os.path.join('/', *path.split('/')[:-2], 'gt', path.split('/')[-1][:-4] + '_gt.png')
    # else:
    #     return os.path.join('/', *path.split('/')[:-2], 'gt', path.split('/')[-1][:-4] + '.png')
    return os.path.join('/', *path.split('/')[:-2], 'gt', path.split('/')[-1][:-4] + '.png')


def get_crop_around_point(images, loc, half_crop):
    x = loc[1]
    y = loc[2]
    return images[loc[0], x:x + 2 * half_crop + 1, y:y + 2 * half_crop + 1, :]


def main(root, output, crop_size, total_size):
    output = output + "_" + str(crop_size)
    os.makedirs(output, exist_ok=True)
    half_crop = int(crop_size / 2)

    splits = zip(['train', 'val', 'test'], [0.7, 0.15, 0.15])

    for split_folder_name, split_size_percentage in splits:
        split_size = int(np.round(total_size * split_size_percentage))
        print(f"Current split is {split_folder_name} of size {split_size} ({split_size_percentage})")

        files = get_all_files_in_folders_and_subfolders(os.path.join(root, split_folder_name, 'data'))
        # Load all data images
        images = np.array([np.array(Image.open(item)) for item in files])
        print("Loaded all data images")
        # Load all gt images
        gts = np.array([convert_to_single_label(np.array(Image.open(get_gt_path(item)))) for item in files])
        print("Loaded all gt images")

        # Get the number of classes
        classes = []
        for gt in gts:
            classes = np.unique(np.hstack([classes, np.unique(gt)]))
        num_classes = len(classes)
        print(f"{num_classes} detected\n")

        for cls in range(num_classes):
            class_size = int(np.round(split_size / num_classes))
            print(f"Current class is {cls}, size ({class_size})")

            locs = np.where(gts[:, half_crop:-half_crop, half_crop:-half_crop] == cls)
            locs = list(zip(locs[0], locs[1], locs[2]))
            # Randomly samples from locs
            random_samples = random.sample(range(0, len(locs) - 1), class_size)
            locs = [locs[i] for i in random_samples]
            # Create the target folder
            target_folder = os.path.join(output, split_folder_name, str(cls))
            os.makedirs(target_folder, exist_ok=True)

            for i, item in enumerate(locs):
                img = get_crop_around_point(images, item, half_crop)
                Image.fromarray(img).save(os.path.join(target_folder, str(i) + '.png'))

    print("ALL DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download some '
                                                 'datasets and prepare them in a standard format')

    parser.add_argument('--root',
                        required=True,
                        type=str,
                        help='Path to a root manuscript folder')
    parser.add_argument('--output',
                        required=True,
                        type=str,
                        help='Path to an output folder')
    parser.add_argument('--crop-size',
                        help='Size to crop around each image',
                        type=int,
                        default=128)
    parser.add_argument('--total-size',
                        help='Total number of images',
                        type=int,
                        default=100000)

    args = parser.parse_args()
    main(**args.__dict__)
