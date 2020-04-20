# Utils

# Delegated
from .evaluate import SemanticSegmentationHisDBEvaluate
from .setup import SemanticSegmentationSetupHisDB
from .train import SemanticSegmentationHisDBTrain
from ..base import BaseRunner


class SemanticSegmentationHisDB(BaseRunner):
    class_encoding = None
    img_names_sizes_dict = None

    def __init__(self):
        super().__init__()
        self.setup = SemanticSegmentationSetupHisDB()

    def prepare(self, model_name, resume, batch_lrscheduler_name, epoch_lrscheduler_name, **kwargs) -> dict:
        """
        See parent class for documentation
        """
        # Setting up the dataloaders
        train_loader, val_loader, test_loader = self.setup.set_up_dataloaders(**kwargs)
        self.class_encoding = train_loader.dataset.classes
        self.img_names_sizes_dict = dict(test_loader.dataset.img_names_sizes)  # (gt_img_name, img_size (H, W))
        num_classes = len(self.class_encoding)

        # return model, len(cls.class_encoding), best_value, train_loader, val_loader, test_loader, optimizer, criterion

        # Setting up model, optimizer, criterion
        model = self.setup.setup_model(model_name=model_name, num_classes=num_classes, train_loader=train_loader,
                                       **kwargs)
        optimizer = self.setup.get_optimizer(model=model, **kwargs)
        criterion = self.setup.get_criterion(**kwargs)

        # Setup the lr schedulers for epochs and batch updates
        batch_lr_schedulers = [self.setup.get_lrscheduler(optimizer=optimizer, lrscheduler_name=name, **kwargs)
                               for name in batch_lrscheduler_name]
        # Append by default a warm-up learning rate scheduler setup on 1 epoch period
        batch_lr_schedulers.append(self.setup.warmup_lr_scheduler(optimizer=optimizer,
                                                                  warmup_factor=1. / 1000000,
                                                                  warmup_iters=len(train_loader) - 1))
        epoch_lr_schedulers = [
            self.setup.get_lrscheduler(optimizer=optimizer, lrscheduler_name=name, verbose=True, patience=20, **kwargs)
            for name in epoch_lrscheduler_name
        ]

        # Resume from checkpoint if necessary
        if resume:
            best_value = self.setup.resume_checkpoint(model=model,
                                                      optimizer=optimizer,
                                                      resume=resume,
                                                      batch_lr_schedulers=batch_lr_schedulers,
                                                      epoch_lr_schedulers=epoch_lr_schedulers,
                                                      **kwargs)
        else:
            best_value = 0.0

        return {
            "model": model,
            "num_classes": num_classes,
            "best_value": best_value,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "optimizer": optimizer,
            "criterion": criterion,
            "batch_lr_schedulers": batch_lr_schedulers,
            "epoch_lr_schedulers": epoch_lr_schedulers,
        }

    ####################################################################################################################
    @classmethod
    def _train(cls, train_loader, **kwargs):
        return SemanticSegmentationHisDBTrain.run(data_loader=train_loader,
                                                  logging_label='train',
                                                  class_encodings=cls.class_encoding,
                                                  **kwargs)

    @classmethod
    def _validate(cls, val_loader, **kwargs):
        return SemanticSegmentationHisDBEvaluate.run(data_loader=val_loader,
                                                     logging_label='train',
                                                     class_encodings=cls.class_encoding,
                                                     **kwargs)

    @classmethod
    def _test(cls, test_loader, **kwargs):
        return SemanticSegmentationHisDBEvaluate.run(data_loader=test_loader,
                                                     logging_label='train',
                                                     class_encodings=cls.class_encoding,
                                                     **kwargs)
