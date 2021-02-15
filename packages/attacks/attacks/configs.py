import argparse


class ModelConfig:
    def __init__(self, parser):
        parser.add_argument("--rounds", help="How many rounds should the model train for.", type=int, default=1000)
        parser.add_argument("--input_size", help="Size of input.", type=int, default=32000)
        parser.add_argument("--hidden_sizes", nargs="+", help="Hidden sizes", default=[6000], type=int)
        parser.add_argument("--middle_dropout", help="Dropout for layers before the last", type=float, default=0.0)
        parser.add_argument("--last_dropout", help="Last dropout", type=float, default=0.2)
        parser.add_argument("--weight_decay", help="Weight decay", type=float, default=1e-6)
        parser.add_argument("--last_non_linearity", help="Last layer non-linearity", type=str, default="relu",
                            choices=["relu", "tanh"])
        parser.add_argument("--non_linearity", help="Before last layer non-linearity", type=str, default="relu",
                            choices=["relu", "tanh"])
        parser.add_argument("--input_transform", help="Transformation to apply to inputs", type=str, default="binarize",
                            choices=["binarize", "none", "tanh"])
        parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
        parser.add_argument("--lr_alpha", help="Learning rate decay multiplier", type=float, default=0.3)
        parser.add_argument("--lr_steps", nargs="+", help="Learning rate decay steps", type=int, default=[10])
        parser.add_argument("--input_size_freq", help="Number of high importance features", type=int, default=None)
        parser.parse_known_args(namespace=self)


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
        parser.parse_known_args(namespace=self)
