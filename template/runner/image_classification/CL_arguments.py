import argparse

from init import advanced_init
from template.runner.base import BaseCLArguments

class CLArguments(BaseCLArguments):

    def __init__(self):
        super().__init__()

        # Add all options
        self._init_options()
        self._transform_options()

    def parse_arguments(self, args=None):
        args, self.parser = super().parse_arguments(args)

        # Set default value for --split-type in _darwin_options()
        if args.split_type is None:
            args.split_type = "stratified_tag"

        return args, self.parser

    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def _transform_options(self):
        """ Chinko options """

        parser_transform = self.parser.add_argument_group('TRANSFORMS', 'Transforms Options')
        parser_transform.add_argument("--random-resized-crop",
                                      default=False,
                                      action='store_true',
                                      help='Flag for using transforms.RandomResizedCrop()')
        parser_transform.add_argument("--random-horizontal-flip",
                                      default=False,
                                      action='store_true',
                                      help='Flag for using transforms.RandomHorizontalFlip()')
        parser_transform.add_argument("--color-jitter",
                                      default=None,
                                      type=float,
                                      nargs='+',
                                      help="Specify the brightness, contrast, saturation and hue of the color-jitter transform. "
                                           "Use as --color-jitter float float float float", )
        parser_transform.add_argument("--rotation",
                                      default=0,
                                      type=float,
                                      help="If present, specifies the random rotation degrees", )


    def _init_options(self):
        """
        Options specific for network initialization methods
        """
        init_options = [name for name in advanced_init.__dict__ if callable(advanced_init.__dict__[name])]

        parser_init = self.parser.add_argument_group('INIT', 'Init Options')

        parser_init.add_argument('--init',
                                 action='store_true',
                                 default=False,
                                 help='use advanced init methods such as LDA')
        parser_init.add_argument('--init-function',
                                 choices=init_options,
                                 help='which initialization function should be used.')
        parser_init.add_argument('--num-samples',
                                 type=int,
                                 help='number of samples to use to perform data-driven initialization')
        parser_init.add_argument('--patches-cap',
                                 type=float,
                                 default=10000,
                                 help='ratio of patch to extract from each sample for conv layers')
        parser_init.add_argument('--solver',
                                 type=str,
                                 choices=['svd', 'eigen'],
                                 default='svd',
                                 help='Which solver is going to be used for LDA operations')
        parser_init.add_argument('--activation-function',
                                 type=str,
                                 choices=['swish', 'softsign'],
                                 default='swish',
                                 help='Which activation function to use in the model for non-linearity')

        # Flags for normalizations
        parser_init.add_argument("--conv-normalize",
                                 type=int,
                                 default=0,
                                 help="Flag for normalizing conv weights")
        parser_init.add_argument("--conv-standardize",
                                 type=int,
                                 default=0,
                                 help="Flag for standardizing conv weights")
        parser_init.add_argument("--conv-scale",
                                 type=int,
                                 default=0,
                                 help="Flag for scaling conv weights")

        parser_init.add_argument("--lin-normalize",
                                 type=int,
                                 default=1,  # Don't even think about touching this!
                                 help="Flag for normalizing linear weights")
        parser_init.add_argument("--lin-standardize",
                                 type=int,
                                 default=0,
                                 help="Flag for standardizing linear weights")
        parser_init.add_argument("--lin-scale",
                                 type=int,
                                 default=0,
                                 help="Flag for normalizing linear weights")

        parser_init.add_argument("--trim-lda",
                                 type=self.str2bool,
                                 default="False",
                                 help="Flag for trimming lda samples on last layer")
        parser_init.add_argument('--sn-ratio',
                                 type=int,
                                 default=0,
                                 help='ratio of noise to be added on the conv weights')

