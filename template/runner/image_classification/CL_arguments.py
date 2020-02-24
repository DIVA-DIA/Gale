import argparse

from init import advanced_init
from template.runner.base import BaseCLArguments

class CLArguments(BaseCLArguments):

    def __init__(self):
        super().__init__()

        # Add all options
        self._init_options()

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
        parser_init.add_argument('--num-samples',
                                 type=int,
                                 default=50000,
                                 help='number of samples to use to perform data-driven initialization')
        parser_init.add_argument('--init-function',
                                 choices=init_options,
                                 help='which initialization function should be used.')
        parser_init.add_argument('--max-patches',
                                 type=float,
                                 default=0.99,
                                 help='ratio of patch to extract from each sample for conv layers')

        parser_init.add_argument("--conv-normalize",
                                 type=self.str2bool,
                                 help="Flag for normalizing conv weights")
        parser_init.add_argument("--conv-standardize",
                                 type=self.str2bool,
                                 default="True",
                                 help="Flag for standardizing conv weights")
        parser_init.add_argument("--conv-scale",
                                 type=self.str2bool,
                                 help="Flag for scaling conv weights")

        parser_init.add_argument("--lin-normalize",
                                 type=self.str2bool,
                                 default="True",
                                 help="Flag for normalizing linear weights")
        parser_init.add_argument("--lin-standardize",
                                 type=self.str2bool,
                                 default="True",
                                 help="Flag for standardizing linear weights")
        parser_init.add_argument("--lin-scale",
                                 type=self.str2bool,
                                 default="True",
                                 help="Flag for normalizing linear weights")

        parser_init.add_argument('--trim-lda-iterations',
                                 type=int,
                                 default=0,
                                 help='number of iterations for trimming the points of the lda')
