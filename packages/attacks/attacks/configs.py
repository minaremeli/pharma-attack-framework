from json import JSONEncoder
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ModelConfig:
    def __init__(self, parser):
        parser.add_argument("--rounds", help="How many rounds should the model train for.", type=int, default=1000)
        parser.add_argument("--input_size", help="Size of input.", type=int, default=32000)
        parser.add_argument("--hidden_sizes", nargs="+", help="Hidden sizes", default=[6000], type=int)
        parser.add_argument("--first_dropout", help="Dropout after first layer.", type=float, default=0.0)
        parser.add_argument("--middle_dropout", help="Dropout for layers before the last", type=float, default=0.2)
        parser.add_argument("--last_dropout", help="Last dropout", type=float, default=0.2)
        parser.add_argument("--weight_decay", help="Weight decay", type=float, default=1e-6)
        parser.add_argument("--last_non_linearity", help="Last layer non-linearity", type=str, default="relu",
                            choices=["relu", "tanh"])
        parser.add_argument("--middle_non_linearity", help="MIddle layer non-linearity", type=str, default="relu", 
                            choices=["relu", "tanh"])
        parser.add_argument("--non_linearity", help="Before last layer non-linearity", type=str, default="relu",
                            choices=["relu", "tanh"])
        parser.add_argument("--input_transform", help="Transformation to apply to inputs", type=str, default="binarize",
                            choices=["binarize", "none", "tanh"])
        parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
        parser.add_argument("--lr_alpha", help="Learning rate decay multiplier", type=float, default=0.3)
        parser.add_argument("--lr_steps", nargs="+", help="Learning rate decay steps", type=int, default=[10])
        parser.add_argument("--input_size_freq", help="Number of high importance features", type=int, default=None)
        parser.add_argument("--uncertainty_weights", help="Whether or not to use uncertainty weighting. Possible "
                                                          "values: ['yes', 'true', 't', 'y', '1'] and ['no', 'false',"
                                                          " 'f', 'n', '0']", type=str2bool, default="no")
        parser.add_argument("--compression", help="Whether to apply compression on trunk gradients before "
                                                  "aggregation. Possible values: ['threshold', 'top-k', "
                                                  "'random subset', 'quantization']",
                            choices=["threshold", "top-k", "random subset", "quantization"], default=None,
                            type=str)
        parser.add_argument("--compression_parameter", help="A parameter with which to run the desired compression "
                                                            "technique. Eg. 0.5 for 'random subset' would mean that "
                                                            "50% of the gradients will be kept at random.",
                            type=float, default=None)
        parser.parse_known_args(namespace=self)


class ModelConfigEncoder(JSONEncoder):
    """
    A specialised JSONEncoder that encodes ModelConfig.
    """

    def default(self, obj):
        if isinstance(obj, ModelConfig):
            return obj.__dict__
        else:
            return JSONEncoder.default(self, obj)



class TrunkActivationAttackConfig:
    def __init__(self, parser):
        parser.add_argument("--num_samples", help="Number of member and non-member samples that the attacker collects "
                                                  "for training and evaluating her attack.", type=int, default=500)
        parser.add_argument("--n_estimators",
                            help="Number of estimators used to train the RandomForestModel. Default is 100.", type=int,
                            default=100)
        parser.parse_known_args(namespace=self)


class NGMAttackConfig:
    def __init__(self, parser):
        parser.add_argument("--num_samples", help="Number of member and non-member samples that the attacker collects "
                                                  "for evaluating her attack.", type=int, default=50)
        parser.add_argument("--voting_threshold", help="The fraction of gradients connected to a non-zero input that "
                                                       "need to be non-zero. Affects attack precision. Usually a "
                                                       "higher threshold means higher precision (higher confidence in "
                                                       "positive predictions).", type=float, default=0.5)
        parser.parse_known_args(namespace=self)

class LeavingAttackConfig:
    def __init__(self, parser):
        parser.add_argument("--num_epochs", help="Number of epochs that the attacker tests his hypothesis on. One "
                                                 "epoch is equal to 50 rounds.", type=int, default=30)
        parser.add_argument("--voting_threshold", help="The fraction of gradients connected to a non-zero input that "
                                                       "need to be non-zero. Affects attack precision. Usually a "
                                                       "higher threshold means higher precision (higher confidence in "
                                                       "positive predictions).", type=float, default=0.2)
        parser.parse_known_args(namespace=self)
