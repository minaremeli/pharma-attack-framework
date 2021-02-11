import argparse
from attacks.configs import ModelConfig, TrunkActivationAttackConfig, NGMAttackConfig
from attacks import NGMAttack, TrunkActivationAttack

class AttackLauncher():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Launching an attack against a federated run.")
        parser.add_argument("--attack_name", help="The name of the attack you want to launch.", required=True, type=str,
                            choices=["NGMA", "TrunkActivation"])

        parser.parse_known_args(namespace=self)

        subparsers = parser.add_subparsers(title="config_parsers")
        model_config_parser = subparsers.add_parser('model_config', help="Training a multi-task model.")
        attack_config_parser = subparsers.add_parser('attack_parser', help="Attack arguments")

        run_config = ModelConfig(model_config_parser)
        if self.attack_name == "TrunkActivation":
            attack_config = TrunkActivationAttackConfig(attack_config_parser)
            attack = TrunkActivationAttack(attack_config, run_config, results_path="test_results.csv")
        else:
            attack_config = NGMAttackConfig(attack_config_parser)
            attack = NGMAttack(attack_config, run_config, results_path="test_results2.csv")
        attack.run_attack()
        attack.log_results()


if __name__ == '__main__':
    al = AttackLauncher()

