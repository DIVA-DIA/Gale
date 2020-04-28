# Utils
import numpy as np

# DeepDIVA
from template.runner.base.base_routine import BaseRoutine
from template.runner.semantic_segmentation_DIVAHisDB.util.accuracy import accuracy_segmentation
from util.metric_logger import MetricLogger

from torchvision import transforms
# Torch related stuff


class SemanticSegmentationHisDBTrain(BaseRoutine):

    @classmethod
    def start_of_the_epoch(cls, model, **kwargs):
        """See parent method for documentation

        Extra-Parameters
        ----------
        model : torch.nn.module
            The network model being used.
        """
        model.train()

        MetricLogger().add_scalar_meter(tag=cls.main_metric())
        MetricLogger().add_scalar_meter(tag='loss')

    @classmethod
    def run_one_mini_batch(cls, model, criterion, input_batch, target, multi_run_label, **kwargs):
        # Compute output
        output = model(input_batch)

        # Compute and record the loss
        loss = criterion(output, target)
        try:
            MetricLogger().update(key='loss', value=loss.item(), n=len(input_batch))
        except AttributeError:
            MetricLogger().update(key='loss', value=loss.data[0], n=len(input_batch))

        # Compute and record the accuracy
        _, _, mean_iu, _ = accuracy_segmentation(target.data.cpu().numpy(), get_argmax(output), kwargs['num_classes'])
        MetricLogger().update(key='meanIU', value=mean_iu, n=len(input_batch))

        optimizer = kwargs['optimizer']
        # Reset gradient
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Perform a step by updating the weights
        optimizer.step()

    @classmethod
    def main_metric(cls) -> str:
        return "meanIU"


def get_argmax(output):
    """ Gets the argmax values for each sample in the minibatch"""
    return np.array([np.argmax(o, axis=0) for o in output.data.cpu().numpy()])