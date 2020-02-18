# Gale
from init.initializer import init_model
from template.runner.base import BaseRunner


# Delegated
from .setup import ImageClassificationSetup
from .train import ImageClassificationTrain
from .evaluate import ImageClassificationEvaluate

class ImageClassification(BaseRunner):

    def __init__(self):
        """
        Attributes
        ----------
        setup = BaseSetup
            (strategy design pattern) Object responsible for setup operations
        """
        super().__init__()
        self.setup =  ImageClassificationSetup()

    ####################################################################################################################
    def prepare(self, model_name, resume, batch_lrscheduler_name, epoch_lrscheduler_name, init=False, **kwargs) -> dict:
        """See parent method for documentation

        Extra-Parameters
        ----------
        init : bool
            Flag for use advanced init methods
        """
        d = super().prepare(model_name, resume, batch_lrscheduler_name, epoch_lrscheduler_name, **kwargs)

        # Init the model
        if init:
            init_model(model=d['model'], data_loader=d['train_loader'], **kwargs)

        return d

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package. 
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """

    def _train(self, train_loader, **kwargs):
        return ImageClassificationTrain.run(data_loader=train_loader, logging_label='train', **kwargs)

    def _validate(self, val_loader, **kwargs):
        return ImageClassificationEvaluate.run(data_loader=val_loader, logging_label='val', **kwargs)

    def _test(self, test_loader, **kwargs):
        return ImageClassificationEvaluate.run(data_loader=test_loader, logging_label='test', **kwargs)
