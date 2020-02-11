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
                                 default="pure_lda",
                                 help='which initialization function should be used.')
        parser_init.add_argument('--max-patches',
                                 type=float,
                                 default=0.9,
                                 help='ratio of patch to extract from each sample for conv layers')