# Utils
import numpy as np

from template.runner.semantic_segmentation_DIVAHisDB.util.accuracy import accuracy_segmentation
# DeepDIVA
from template.runner.base.base_routine import BaseRoutine
from util.metric_logger import MetricLogger


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
    def run_one_mini_batch(cls, model, criterion, input, target, multi_run_label, **kwargs):
        # Compute output
        output = model(input)

        # Compute and record the loss
        loss = criterion(output, target)
        try:
            MetricLogger().update(key='loss', value=loss.item(), n=len(input))
        except AttributeError:
            MetricLogger().update(key='loss', value=loss.data[0], n=len(input))

        output_argmax = np.array([np.argmax(o, axis=0) for o in output.data.cpu().numpy()])
        target_argmax = target.data.cpu().numpy()

        # Compute and record the accuracy
        _, _, mean_iu, _ = accuracy_segmentation(target_argmax, output_argmax, kwargs['num_classes'])
        MetricLogger().update(key='meanIU', value=mean_iu, n=len(input))

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
