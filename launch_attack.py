import argparse
from attacks.configs import ModelConfig, TrunkActivationAttackConfig, NGMAttackConfig
from attacks import NGMAttack, TrunkActivationAttack, BaseAttack
import torch
import random
import numpy as np

class AttackLauncher():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Launching an attack against a federated run.")
        parser.add_argument("--attack_name", help="The name of the attack you want to launch.", required=True, type=str,
                            choices=["NGMA", "TrunkActivation", "NOATTACK"])
        parser.add_argument("--results_file", help="Name of the .csv file you want to save your results in. New results "
                                                "will be appended. Example: results.csv", required=True, type=str)
        parser.add_argument("--seed", help="Set a seed for running the attack and training the model.", required=True, type=int)

        parser.parse_known_args(namespace=self)

        # set seed for everything
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        subparsers = parser.add_subparsers(title="config_parsers")
        model_config_parser = subparsers.add_parser('model_config', help="Training a multi-task model.")
        attack_config_parser = subparsers.add_parser('attack_parser', help="Attack arguments")

        run_config = ModelConfig(model_config_parser)
        if self.attack_name == "NOATTACK":
            attack_config = TrunkActivationAttackConfig(attack_config_parser)
            attack_config.__dict__ = {}
            attack = BaseAttack(attack_config, run_config, results_path=self.results_file)
            trunk, server, clients = attack._initialize_and_train_targeted_model()
            attack._evaluate_model(clients)
            attack.log_results()
        else:
            if self.attack_name == "TrunkActivation":
                attack_config = TrunkActivationAttackConfig(attack_config_parser)
                attack = TrunkActivationAttack(attack_config, run_config, results_path=self.results_file)
            else:
                attack_config = NGMAttackConfig(attack_config_parser)
                attack = NGMAttack(attack_config, run_config, results_path=self.results_file)
            attack.run_attack()
            attack.log_results()


if __name__ == '__main__':
    al = AttackLauncher()

