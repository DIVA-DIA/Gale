from template.runner.base import BaseCLArguments


class CLArguments(BaseCLArguments):

    def __init__(self):
        super().__init__()

        # Add all options
        self._semantic_segmentation_options()

    def _semantic_segmentation_options(self):
        """
        Triplet options

        These parameters are used by the runner class template.runner.semantic_segmentation
        """
        semantic_segmentation = self.parser.add_argument_group('Semantic', 'Semantic Segmentation')

        semantic_segmentation.add_argument('--input-patch-size',
                                           type=int,
                                           default=128, metavar='N',
                                           help='size of the square input patch e.g. with 32 the input will be re-sized to 32x32')

        # parameters for HisDB
        semantic_segmentation.add_argument('--imgs-in-memory',
                                           type=int,
                                           default=4, metavar='N',
                                           help='number of pages that are loaded into RAM and learned on')
        semantic_segmentation.add_argument('--crop-size',
                                           type=int,
                                           default=128, metavar='N',
                                           help='size of each crop taken (default 32x32)')
        semantic_segmentation.add_argument('--crops-per-image',
                                           type=int,
                                           default=50, metavar='N',
                                           help='number of crops per iterations per page')
        semantic_segmentation.add_argument('--post-process',
                                           action='store_true',
                                           default=False,
                                           help='apply post processing to the image')

        # parameter for DeepLabV3 if you want to use cityscapes pre-trained model
        semantic_segmentation.add_argument('--cityscapes',
                                           default=False,
                                           action='store_true',
                                           help='set if you want to use the cityscapes pre-trained model for DeepLabV3')

        # parameters for COCO
        semantic_segmentation.add_argument('--resize-coco',
                                           type=int,
                                           default=None, metavar='size',
                                           help='size you want coco input to be resized to')
