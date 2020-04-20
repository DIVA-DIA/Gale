import numpy as np

# Gale
from util.metric_logger import MetricLogger
from .train import SemanticSegmentationHisDBTrain
from .util.accuracy import accuracy_segmentation


class SemanticSegmentationHisDBEvaluate(SemanticSegmentationHisDBTrain):

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
    def run_one_mini_batch(cls, model, criterion, input_batch, target, **kwargs):
        """See parent method for documentation"""
        # Compute output
        output = model(input_batch)

        # Compute and record the loss
        loss = criterion(output, target)
        MetricLogger().update(key='loss', value=loss.item(), n=len(input_batch))

        # Compute and record the accuracy
        _, _, mean_iu, _ = accuracy_segmentation(target.cpu().numpy(), get_argmax(output), kwargs['num_classes'])
        MetricLogger().update(key='meanIU', value=mean_iu, n=len(input_batch))


def get_argmax(output):
    """ Gets the argmax values for each sample in the minibatch"""
    return np.array([np.argmax(o, axis=0) for o in output.data.cpu().numpy()])
