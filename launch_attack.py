import argparse
from attacks.configs import ModelConfig, TrunkActivationAttackConfig, NGMAttackConfig, LeavingAttackConfig, str2bool
from attacks import NGMAttack, TrunkActivationAttack, BaseAttack, LeavingAttack, ActiveTrunkActivationAttack
import torch
import random
import numpy as np
import os

class AttackLauncher():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Launching an attack against a federated run.")
        parser.add_argument("--attack_name", help="The name of the attack you want to launch.", required=True, type=str,
                            choices=["NGMA", "TrunkActivation", "NOATTACK", "Leaving", "ActiveTrunkActivation"])
        parser.add_argument("--results_file", help="Name of the .csv file you want to save your results in. New results "
                                                "will be appended. Example: results.csv", required=True, type=str)
        parser.add_argument("--seed", help="Set a seed for running the attack and training the model.", required=True, type=int)
        parser.add_argument("--model_save", help="Indicates whether the trained model should be saved. Positive "
                                                 "values: ['yes', 'true', 't', 'y', '1']. Negative values: ['no', "
                                                 "'false', 'f', 'n', '0']",
                            required=True, type=str2bool)
        parser.add_argument("--model_save_path", help="If --model_save is set to true, it specifies the path where "
                                                      "the models should be saved. If --model_save is set to false, "
                                                      "it specifies from where it should load the models. Default is "
                                                      "None, when it does not save the models at all.", type=str,
                            default=None)

        parser.parse_known_args(namespace=self)

        # set seed for everything
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)


        if self.model_save_path and not self.model_save:
            self.load_saved_model = True
        else:
            self.load_saved_model = False

        subparsers = parser.add_subparsers(title="config_parsers")
        model_config_parser = subparsers.add_parser('model_config', help="Training a multi-task model.")
        attack_config_parser = subparsers.add_parser('attack_parser', help="Attack arguments")

        run_config = ModelConfig(model_config_parser)
        run_config.seed = self.seed
        if self.attack_name == "NOATTACK":
            attack_config = TrunkActivationAttackConfig(attack_config_parser)
            attack_config.__dict__ = {}
            attack = BaseAttack(attack_config, run_config, results_path=self.results_file, load_saved_model=self.load_saved_model, model_save=self.model_save, save_path=self.model_save_path)
            trunk, server, clients = attack._initialize_and_train_targeted_model()
            attack._evaluate_model(clients)
            attack.log_results()
        else:
            if self.attack_name == "TrunkActivation":
                attack_config = TrunkActivationAttackConfig(attack_config_parser)
                attack = TrunkActivationAttack(attack_config, run_config, results_path=self.results_file, load_saved_model=self.load_saved_model, model_save=self.model_save, save_path=self.model_save_path)
            elif self.attack_name == "ActiveTrunkActivation":
                attack_config = TrunkActivationAttackConfig(attack_config_parser)
                attack = ActiveTrunkActivationAttack(attack_config, run_config, results_path=self.results_file, load_saved_model=self.load_saved_model, model_save=self.model_save, save_path=self.model_save_path)
            elif self.attack_name == "NGMA":
                attack_config = NGMAttackConfig(attack_config_parser)
                attack = NGMAttack(attack_config, run_config, results_path=self.results_file, load_saved_model=self.load_saved_model, model_save=self.model_save, save_path=self.model_save_path)
            else:
                attack_config = LeavingAttackConfig(attack_config_parser)
                attack = LeavingAttack(attack_config, run_config, results_path=self.results_file, load_saved_model=self.load_saved_model, model_save=self.model_save, save_path=self.model_save_path)
            attack.run_attack()
            attack.log_results()


if __name__ == '__main__':
    al = AttackLauncher()

