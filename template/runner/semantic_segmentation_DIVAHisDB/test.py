import logging
import os
import time

import matplotlib
import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets.custom.functional import gt_to_one_hot_hisdb
from template.runner.semantic_segmentation_DIVAHisDB.evaluate import get_argmax, SemanticSegmentationHisDBEvaluate
from template.runner.semantic_segmentation_DIVAHisDB.util.accuracy import accuracy_segmentation
from util.TB_writer import TBWriter
from util.metric_logger import MetricLogger, ScalarValue
from .setup import SemanticSegmentationSetupHisDB
from .util.misc import make_colour_legend_image
from .visualization.DIVAHisDB_layout_analysis_tool_visualization import generate_layout_analysis_output


class SemanticSegmentationHisDBTest(SemanticSegmentationHisDBEvaluate):

    @classmethod
    def run(cls, data_loader, epoch, log_interval, logging_label, batch_lr_schedulers, run=None, **kwargs):
        """
                Training routine

                Parameters
                ----------
                data_loader : torch.utils.data.DataLoader
                    The dataloader of the current set.
                epoch : int
                    Number of the epoch (for logging purposes).
                log_interval : int
                    Interval limiting the logging of mini-batches.
                logging_label : string
                    Label for logging purposes. Typically 'train', 'test' or 'valid'.
                    It's prepended to the logging output path and messages.
                run : int
                    Number of run, used in multi-run context to discriminate the different runs
                batch_lr_schedulers : list(torch.optim.lr_scheduler)
                    List of lr schedulers to call step() on after every batch

                Returns
                ----------
                Main metric : float
                    Main metric of the model on the evaluated split
                """
        # 'run' is injected in kwargs at runtime in RunMe.py IFF it is a multi-run event
        multi_run_label = f"_{run}" if run is not None else ""

        # Instantiate the counter
        MetricLogger().reset(postfix=multi_run_label)

        # Custom routine to run at the start of the epoch
        cls.start_of_the_epoch(data_loader=data_loader,
                               epoch=epoch,
                               logging_label=logging_label,
                               multi_run_label=multi_run_label,
                               **kwargs)

        # Iterate over whole training set
        end = time.time()

        canvas = {}
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=130, leave=False)
        for batch_idx, (input_batch, target) in pbar:
            input_batch, top_left_coordinates, test_img_names = input_batch

            # Measure data loading time
            data_time = time.time() - end

            # Moving data to GPU
            input_batch, target = cls.move_to_device(input_batch=input_batch, target=target, **kwargs)

            # Compute output
            output = kwargs['model'](input_batch)

            # Compute and record the loss
            loss = kwargs['criterion'](output, target)
            MetricLogger().update(key='loss', value=loss.item(), n=len(input_batch))

            # Compute and record the accuracy
            _, _, mean_iu, _ = accuracy_segmentation(target.cpu().numpy(), get_argmax(output), kwargs['num_classes'])
            MetricLogger().update(key='meanIU', value=mean_iu, n=len(input_batch))

            # Update the LR according to the scheduler, only during training
            if 'val' not in logging_label and 'test' not in logging_label:
                for lr_scheduler in batch_lr_schedulers:
                    lr_scheduler.step()

            # Add metrics to Tensorboard for the last mini-batch value
            for tag, meter in MetricLogger():
                if isinstance(meter, ScalarValue):
                    TBWriter().add_scalar(tag=logging_label + '/mb_' + tag,
                                          scalar_value=meter.value,
                                          global_step=epoch * len(data_loader) + batch_idx)

            # Measure elapsed time for a mini-batch
            batch_time = time.time() - end
            end = time.time()

            # Output needs to be patched together to form the complete output of the full image
            # patches are returned as a sliding window over the full image, overlapping sections are averaged
            for patch, x, y, img_name in zip(output.data.cpu().numpy(), top_left_coordinates[0].numpy(),
                                             top_left_coordinates[1].numpy(), test_img_names):

                # Is a new image?
                if img_name not in canvas:
                    # Create a new image of the right size filled with NaNs
                    img_size_dict = dict(data_loader.dataset.img_names_sizes)
                    canvas[img_name] = np.empty((kwargs['num_classes'], *img_size_dict[img_name]))
                    canvas[img_name].fill(np.nan)

                # Add the patch to the image
                canvas[img_name] = merge_patches(patch, (x, y), canvas[img_name])

                # Save the image when done
                if not np.isnan(np.sum(canvas[img_name])):
                    # Save the final image
                    mean_iu = process_full_image(img_name, canvas[img_name], **kwargs)
                    # Update the meanIU
                    MetricLogger().update(key='meanIU', value=1)
                    # Remove the entry
                    canvas.pop(img_name)
                    logging.info("\nProcessed image {} with mean IU={}".format(img_name, mean_iu))

            # Log to console
            if batch_idx % log_interval == 0 and len(MetricLogger()) > 0:
                if batch_idx % log_interval == 0 and len(MetricLogger()) > 0:
                    if cls.main_metric() + multi_run_label in MetricLogger():
                        mlogger = MetricLogger()[cls.main_metric()]
                    elif "loss" + multi_run_label in MetricLogger():
                        mlogger = MetricLogger()["loss"]
                    else:
                        raise AttributeError
                pbar.set_description(f'{logging_label} epoch [{epoch}][{batch_idx}/{len(data_loader)}]')
                pbar.set_postfix(Metric=f'{mlogger.global_avg:.3f}',
                                 Time=f'{batch_time:.3f}',
                                 Data=f'{data_time:.3f}')

        # Canvas MUST be empty or something was wrong with coverage of all images
        assert len(canvas) == 0

        # Custom routine to run at the end of the epoch
        cls.end_of_the_epoch(data_loader=data_loader,
                             epoch=epoch,
                             logging_label=logging_label,
                             multi_run_label=multi_run_label,
                             **kwargs)

        # Add metrics to Tensorboard for the full-epoch value
        for tag, meter in MetricLogger():
            if isinstance(meter, ScalarValue):
                TBWriter().add_scalar(tag=logging_label + '/' + tag, scalar_value=meter.global_avg, global_step=epoch)

        if cls.main_metric() + multi_run_label in MetricLogger():
            return MetricLogger()[cls.main_metric()].global_avg
        else:
            return 0

    @classmethod
    def start_of_the_epoch(cls, model, num_classes, **kwargs):
        """See parent method for documentation

        Extra-Parameters
        ----------
        model : torch.nn.module
            The network model being used.
        num_classes : int
            Number of classes in the dataset
        """
        model.eval()

        MetricLogger().add_scalar_meter(tag=cls.main_metric())
        MetricLogger().add_scalar_meter(tag='loss')

    @classmethod
    def main_metric(cls) -> str:
        return super().main_metric()


def merge_patches(patch, coordinates, full_output):
    """
    This function merges the patch into the full output image
    Overlapping values are resolved by taking the max.

    Parameters
    ----------
    patch: numpy matrix of size [batch size x #C x crop_size x crop_size]
        a patch from the larger image
    coordinates: tuple of ints
        top left coordinates of the patch within the larger image for all patches in a batch
    full_output: numpy matrix of size [#C x H x W]
        output image at full size
    Returns
    -------
    full_output: numpy matrix [#C x Htot x Wtot]
    """
    assert len(full_output.shape) == 3
    assert full_output.size != 0

    # Resolve patch coordinates
    x1, y1 = coordinates
    x2, y2 = x1 + patch.shape[1], y1 + patch.shape[2]

    # If this triggers it means that a patch is 'out-of-bounds' of the image and that should never happen!
    assert x2 <= full_output.shape[1]
    assert y2 <= full_output.shape[2]

    mask = np.isnan(full_output[:, x1:x2, y1:y2])
    # if still NaN in full_output just insert value from crop, if there is a value then take max
    full_output[:, x1:x2, y1:y2] = np.where(mask, patch, np.maximum(patch, full_output[:, x1:x2, y1:y2]))

    return full_output


def process_full_image(image_name, output, multi_run, input_folder, class_encoding, post_process, **kwargs):
    """
    Helper function to save the output during testing

    Parameters
    ----------
    meanIU.avg : float
        MeanIU of the model of the evaluated split
    multi_run :

    image_name: str
        name of the image that is saved
    output: numpy matrix of size [#C x H x W]
        output image at full size
    input_folder: str
        path to the dataset folder

    post_process : Boolean
        apply post-processing to the output of the network

    Returns
    -------
    mean_iu : float
        mean iu of this image
    """
    # Load GT
    with open(os.path.join(input_folder, "test", "gt", image_name), 'rb') as f:
        with Image.open(f) as img:
            gt = np.array(img)

    # Get predictions
    if post_process:
        # Load original image
        with open(os.path.join(input_folder, "test", "data", image_name[:-4] + ".JPG"), 'rb') as f:
            with Image.open(f) as img:
                original_image = np.array(img)
        # Apply CRF
        # prediction = crf(original_image, output)
        # output_encoded = output_to_class_encodings(prediction, class_encodings, perform_argmax=False)
    else:
        prediction = np.argmax(output, axis=0)
        output_encoded = SemanticSegmentationSetupHisDB.output_to_class_encodings(output, class_encoding)

    # Get boundary pixels and adjust the gt_image for the border pixel -> set to background (1)
    boundary_mask = gt[:, :, 0].astype(np.uint8) == 128

    # Get the ground truth mapping and filter their values for the boundary pixels
    target = np.argmax(gt_to_one_hot_hisdb(gt, class_encoding).numpy(), axis=0)
    target[np.logical_and.reduce([boundary_mask, prediction != target,
                                  prediction == 0])] = 0  # NOTE: here 0 is 0x1 because it works on the index!

    # Compute and record the meanIU of the whole image
    _, _, mean_iu, _ = accuracy_segmentation(target, prediction, len(class_encoding))

    scalar_label = 'output_{}'.format(image_name) if multi_run is None else 'output_{}_{}'.format(multi_run,
                                                                                                  image_name)
    _save_output_evaluation(class_encoding, output_encoded=output_encoded, gt_image=gt, tag=scalar_label,
                            multi_run=multi_run, **kwargs)

    return mean_iu


def _save_output_evaluation(class_encodings, output_encoded, gt_image, tag, multi_run=None, **kwargs):
    """Utility function to save image in the output folder and also log it to Tensorboard.

    Parameters
    ----------
    class_encodings : List
        Contains the range of encoded classes
    tag : str
        Name of the image.
    output_encoded : ndarray [W x H x C] in RGB
        Image to be saved
    gt_image : ndarray [W x H x C] in RGB
        Image to be saved
    multi_run : int
        Epoch/Mini-batch counter.

    Returns
    -------
    None

    """
    # ##################################################################################################################
    # 1. Create true output

    # Get output folder using the FileHandler from the logger.
    # (Assumes the file handler is the last one)
    output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)
    dest_filename = os.path.join(output_folder, 'images', "output",
                                 tag if multi_run is None else tag + '_{}'.format(multi_run))
    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    # Save the output
    Image.fromarray(output_encoded.astype(np.uint8)).save(dest_filename)

    # ##################################################################################################################
    # 2. Make a more human readable output -> one colour per class
    tag_col = "coloured/" + tag

    dest_filename = os.path.join(output_folder, 'images',
                                 tag_col if multi_run is None else tag_col + '_{}'.format(multi_run))
    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    img = np.copy(output_encoded)
    blue = output_encoded[:, :, 2]  # Extract just blue channel

    # Colours are in RGB
    cmap = matplotlib.cm.get_cmap('Spectral')
    colors = [cmap(i / len(class_encodings), bytes=True)[:3] for i in range(len(class_encodings))]

    # Get the mask for each colour
    masks = {color: (blue == i) > 0 for color, i in zip(colors, class_encodings)}

    # Color the image with relative colors
    for color, mask in masks.items():
        img[mask] = color

    # Make and save the class color encoding
    color_encoding = {str(i): color for color, i in zip(colors, class_encodings)}

    make_colour_legend_image(
        os.path.join(os.path.dirname(dest_filename), "output_visualizations_colour_legend.png"),
        color_encoding)

    # Write image to output folder
    Image.fromarray(img.astype(np.uint8)).save(dest_filename)

    # ##################################################################################################################
    # 3. Layout analysis evaluation
    # Output image as described in https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator

    tag_la = "layout_analysis_evaluation/" + tag

    dest_filename = os.path.join(output_folder, 'images',
                                 tag_la if multi_run is None else tag_la + '_{}'.format(multi_run))
    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    generate_layout_analysis_output(os.path.join(output_folder, 'images'), gt_image, output_encoded, dest_filename,
                                    legend=True)
