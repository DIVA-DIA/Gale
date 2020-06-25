import models
import warnings
import argparse
import re
import traceback


def list_all_model_parameters():
    for model_name in models.__dict__:
        m = models.__dict__[model_name]
        _print_model_parameters(m)


def list_model_parameters_by_pattern(pattern):
    for model_name in models.__dict__:
        if re.search(pattern, model_name):
            m = models.__dict__[model_name]
            _print_model_parameters(m)


def _print_model_parameters(model):
    if callable(model):
        try:
            model = model(num_classes=10, output_channels=10, pretrained=False)
            print(model.__class__.__name__)
            print('Number of trainable model parameters: %d'
                  % sum(p.numel() for p in model.parameters() if p.requires_grad))
            print('Number of total model parameters: %d\n' % sum(
                p.numel() for p in model.parameters()))
        except:
            warnings.warn("Couldn't access parameters of callable %s due to the following error:" %
                          model.__class__.__name__)
            traceback.print_exc()
            print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Lists the number of model parameters')

    parser.add_argument(
        "--all",
        action="store_true",
        help="List all possible parameters"
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="List models that follow a defined pattern"
    )

    args = parser.parse_args()
    if args.all:
        list_all_model_parameters()
    elif args.pattern is not None:
        list_model_parameters_by_pattern(args.pattern)
    else:
        print("Invalid CL arguments")
        parser.print_help()

