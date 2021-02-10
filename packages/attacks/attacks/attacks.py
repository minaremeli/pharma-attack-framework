import csv
import os
import sparsechem as sc
import utils.data_utils as du
from collaborative.model import TrunkAndHead
from collaborative.participant import Client
from collaborative.participant import Server
from collaborative.model import Trunk
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
import numpy as np
import torch


def sparse_to_tensor(mtx):
    return torch.FloatTensor(mtx.toarray())


class BaseAttack:
    def __init__(self, attack_config, model_config, results_path):
        self.attack_config = attack_config
        self.model_config = model_config
        print(attack_config)

        self.results_path = results_path
        self.results_dict = {}

    def run_attack(self):
        raise NotImplementedError()
import csv
import os
import sparsechem as sc
import utils.data_utils as du
from collaborative.model import TrunkAndHead
from collaborative.participant import Client
from collaborative.participant import Server
from collaborative.model import Trunk
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
import numpy as np
import torch


def sparse_to_tensor(mtx):
    return torch.FloatTensor(mtx.toarray())


class BaseAttack:
    def __init__(self, attack_config, model_config, results_path):
        self.attack_config = attack_config
        self.model_config = model_config
        print(attack_config.__dict__)

        self.results_path = results_path
        self.results_dict = {}

    def run_attack(self):
        raise NotImplementedError()

    def log_results(self):
        attack_model_results_dict = {**self.attack_config.__dict__, **self.model_config.__dict__, **self.results_dict}

        with open(self.results_path, "a+") as f:
            writer = csv.DictWriter(f, fieldnames=list(attack_model_results_dict.keys()))
            if os.path.getsize(self.results_path) == 0:
                ## if file is empty
                writer.writeheader()
            writer.writerow(attack_model_results_dict)

    def _initialize_and_train_targeted_model(self):
        print("Train targeted model...")
        rounds = self.model_config.rounds

        trunk = Trunk(self.model_config)
        server = Server(trunk, conf=self.model_config)
        clients = self._initialize_clients(trunk)
        self._train_targeted_model(clients=clients, server=server, rounds=rounds)

        return trunk, server, clients

    def _initialize_clients(self, trunk):
        partner_data = du.load_partner_data("data", range(10))
        clients = []
        for i, (x, y) in enumerate(partner_data):
            self.model_config.output_size = y.shape[1]
            self.model_config.batch_size = int(x.shape[0] * 0.02)
            model = TrunkAndHead(trunk, self.model_config)
            print("Model of %d. partner" % i)
            print(model)
            dataset = sc.SparseDataset(x, y)
            client = Client(model, conf=self.model_config, dataset=dataset)
            clients.append(client)
        return clients

    def _train_targeted_model(self, clients, server, rounds):
        for r in tqdm(range(rounds)):
            self._train_clients(clients)
            # server updates his trunk
            server.update_weights()
            # server zeroes his trunk
            server.zero_grad()

    def _train_clients(self, clients):
        for i, client in enumerate(clients):
            # client gets his next batch
            batch = client.get_next_batch()
            # client calculates updates for trunk+head on that batch
            client.train(batch)
            # client updates his head
            client.update_weights()
            # client zeroes his head
            client.zero_grad()


class TrunkActivationAttack(BaseAttack):
    def __init__(self, attack_config, model_config, results_path):
        print("Trunk Activation Attack")
        super(TrunkActivationAttack, self).__init__(attack_config, model_config, results_path)

    def run_attack(self):
        trunk, server, clients = self._initialize_and_train_targeted_model()
        x_member, _, x_non_member, _ = du.load_member_non_member_data("data", range(10))
        activation_shape = self.model_config.hidden_sizes[0]
        num_samples = self.attack_config.num_samples

        activations = self._load_activations(trunk=trunk, member_samples=x_member,
                                             non_member_samples=x_non_member,
                                             activation_shape=activation_shape, num_samples=num_samples)

        X = np.vstack([activations["member"][:len(activations["non-member"])], activations["non-member"]])
        y = np.array([1] * activations["non-member"].shape[0] + [0] * activations["non-member"].shape[0])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        n_estimators = self.attack_config.n_estimators

        atk = RandomForestClassifier(n_estimators)
        atk.fit(X_train, y_train)

        y_pred = atk.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        self.results_dict["TN"], self.results_dict["FP"], self.results_dict["FN"], self.results_dict[
            "TP"] = tn, fp, fn, tp
        self.results_dict["accuracy"] = accuracy_score(y_pred, y_test)
        self.results_dict["precision"] = precision_score(y_pred, y_test)
        self.results_dict["recall"] = recall_score(y_pred, y_test)

        print("Attack results:")
        print(self.results_dict)

    def _load_activations(self, trunk, member_samples, non_member_samples, activation_shape, num_samples=None):
        activations = {}

        if num_samples is not None and type(num_samples) == int:
            activations["member"] = np.empty((num_samples, activation_shape))
            activations["non-member"] = np.empty((num_samples, activation_shape))

            for i, s in tqdm(enumerate(member_samples[:num_samples])):
                activations["member"][i] = trunk(sparse_to_tensor(s)).detach().numpy()

            for i, s in tqdm(enumerate(non_member_samples[:num_samples])):
                activations["non-member"][i] = trunk(sparse_to_tensor(s)).detach().numpy()
        else:
            activations["member"] = np.empty((member_samples.shape[0], activation_shape))
            activations["non-member"] = np.empty((non_member_samples.shape[0], activation_shape))

            for i, s in tqdm(enumerate(member_samples)):
                activations["member"][i] = trunk(sparse_to_tensor(s)).detach().numpy()

            for i, s in tqdm(enumerate(non_member_samples)):
                activations["non-member"][i] = trunk(sparse_to_tensor(s)).detach().numpy()

        return activations


class NGMAttack(BaseAttack):
    def __init__(self, attack_config, model_config, results_path):
        super(NGMAttack, self).__init__(self, attack_config, model_config, results_path)

    def run_attack(self):
        print("Naive Gradient Attack")
        trunk, server, clients = self._initialize_and_train_targeted_model()

        num_attacks = self.attack_config.num_attacks
        for i in tqdm(range(num_attacks)):
            self._train_clients(clients)

    def log_results(self):
        attack_model_results_dict = {**self.attack_config.__dict__, **self.model_config.__dict__, **self.results_dict}

        with open(self.results_path, "a+") as f:
            writer = csv.DictWriter(f, fieldnames=list(attack_model_results_dict.keys()))
            if os.path.getsize(self.results_path) == 0:
                ## if file is empty
                writer.writeheader()
            writer.writerow(attack_model_results_dict)

    def _initialize_and_train_targeted_model(self):
        print("Train targeted model...")
        rounds = self.model_config.rounds

        trunk = Trunk(self.model_config)
        server = Server(trunk, conf=self.model_config)
        clients = self._initialize_clients(trunk)
        self._train_targeted_model(clients=clients, server=server, rounds=rounds)

        return trunk, server, clients

    def _initialize_clients(self, trunk):
        partner_data = du.load_partner_data("data", range(10))
        clients = []
        for i, (x, y) in enumerate(partner_data):
            self.model_config.output_size = y.shape[1]
            self.model_config.batch_size = int(x.shape[0] * 0.02)
            model = TrunkAndHead(trunk, self.model_config)
            print("Model of %d. partner" % i)
            print(model)
            dataset = sc.SparseDataset(x, y)
            client = Client(model, conf=self.model_config, dataset=dataset)
            clients.append(client)
        return clients

    def _train_targeted_model(self, clients, server, rounds):
        for r in tqdm(range(rounds)):
            self._train_clients(clients)
            # server updates his trunk
            server.update_weights()
            # server zeroes his trunk
            server.zero_grad()

    def _train_clients(self, clients):
        for i, client in enumerate(clients):
            # client gets his next batch
            batch = client.get_next_batch()
            # client calculates updates for trunk+head on that batch
            client.train(batch)
            # client updates his head
            client.update_weights()
            # client zeroes his head
            client.zero_grad()


class TrunkActivationAttack(BaseAttack):
    def __init__(self, attack_config, model_config, results_path):
        print("Trunk Activation Attack")
        super(TrunkActivationAttack, self).__init__(attack_config, model_config, results_path)

    def run_attack(self):
        trunk, server, clients = self._initialize_and_train_targeted_model()
        x_member, _, x_non_member, _ = du.load_member_non_member_data("data", range(10))
        activation_shape = self.model_config.hidden_sizes[0]
        num_samples = self.attack_config.num_samples

        activations = self._load_activations(trunk=trunk, member_samples=x_member,
                                             non_member_samples=x_non_member,
                                             activation_shape=activation_shape, num_samples=num_samples)

        X = np.vstack([activations["member"][:len(activations["non-member"])], activations["non-member"]])
        y = np.array([1] * activations["non-member"].shape[0] + [0] * activations["non-member"].shape[0])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        n_estimators = self.attack_config.n_estimators

        atk = RandomForestClassifier(n_estimators)
        atk.fit(X_train, y_train)

        y_pred = atk.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        self.results_dict["TN"], self.results_dict["FP"], self.results_dict["FN"], self.results_dict[
            "TP"] = tn, fp, fn, tp
        self.results_dict["accuracy"] = accuracy_score(y_pred, y_test)
        self.results_dict["precision"] = precision_score(y_pred, y_test)
        self.results_dict["recall"] = recall_score(y_pred, y_test)

        print("Attack results:")
        print(self.results_dict)

    def _load_activations(self, trunk, member_samples, non_member_samples, activation_shape, num_samples=None):
        activations = {}

        if num_samples is not None and type(num_samples) == int:
            activations["member"] = np.empty((num_samples, activation_shape))
            activations["non-member"] = np.empty((num_samples, activation_shape))

            for i, s in tqdm(enumerate(member_samples[:num_samples])):
                activations["member"][i] = trunk(sparse_to_tensor(s)).detach().numpy()

            for i, s in tqdm(enumerate(non_member_samples[:num_samples])):
                activations["non-member"][i] = trunk(sparse_to_tensor(s)).detach().numpy()
        else:
            activations["member"] = np.empty((member_samples.shape[0], activation_shape))
            activations["non-member"] = np.empty((non_member_samples.shape[0], activation_shape))

            for i, s in tqdm(enumerate(member_samples)):
                activations["member"][i] = trunk(sparse_to_tensor(s)).detach().numpy()

            for i, s in tqdm(enumerate(non_member_samples)):
                activations["non-member"][i] = trunk(sparse_to_tensor(s)).detach().numpy()

        return activations


class NGMAttack(BaseAttack):
    def __init__(self, attack_config, model_config, results_path):
        super(NGMAttack, self).__init__(self, attack_config, model_config, results_path)

    def run_attack(self):
        print("Naive Gradient Attack")
        trunk, server, clients = self._initialize_and_train_targeted_model()

        num_attacks = self.attack_config.num_attacks
        for i in tqdm(range(num_attacks)):
            self._train_clients(clients)
