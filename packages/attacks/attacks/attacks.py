import csv
import os
import sparsechem as sc
import utils.data_utils as du
from collaborative.model import TrunkAndHead
from collaborative.participant import Client
from collaborative.participant import Server
from collaborative.model import Trunk
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scipy.sparse as sparse
import scipy.stats as stat
from pyintersect import intersect, match
from os import path
import json
from .configs import ModelConfigEncoder
from compression import GradientCompressor, CompressionMethods
from .adversaries import IsolatingServer


def sparse_to_tensor(mtx):
    return torch.FloatTensor(mtx.toarray())


class BaseAttack:
    def __init__(self, attack_config, model_config, results_path, load_saved_model, model_save, save_path):
        self.attack_config = attack_config
        self.model_config = model_config
        print(attack_config.__dict__)

        self.results_path = results_path
        self.load_saved_model = load_saved_model
        self.model_save = model_save
        self.save_path = save_path
        self.results_dict = {}

        if self.model_config.compression:
            self.compress = True
        else:
            self.compress = False

    def run_attack(self):
        raise NotImplementedError()

    def log_results(self):
        del self.model_config.batch_size
        del self.model_config.output_size
        attack_model_results_dict = {**self.attack_config.__dict__, **self.model_config.__dict__, **self.results_dict}

        with open(self.results_path, "a+") as f:
            writer = csv.DictWriter(f, fieldnames=list(attack_model_results_dict.keys()))
            if os.path.getsize(self.results_path) == 0:
                ## if file is empty
                writer.writeheader()
            writer.writerow(attack_model_results_dict)

    def _get_compression_method(self):
        compression_method = self.model_config.compression
        if compression_method == "threshold":
            return CompressionMethods.THRESHOLD
        elif compression_method == "top-k":
            return CompressionMethods.TOP_K
        elif compression_method == "random subset":
            return CompressionMethods.RAND_SUB
        elif compression_method == "quantization":
            return CompressionMethods.QUANTIZATION

    def _initialize_and_train_targeted_model(self, save_intermediate=False):
        trunk, server, clients = self._initialize_targeted_model()

        if self.compress:
            self._register_compression(trunk.parameters(), clients)

        if not self.load_saved_model:
            print("Train targeted model...")
            rounds = self.model_config.rounds
            self._train_targeted_model(clients=clients, server=server, rounds=rounds, save_intermediate=save_intermediate)
            if self.model_save:
                self._save_models(server, clients)

        return trunk, server, clients

    def _initialize_targeted_model(self, filename="model.pkl"):
        if self.load_saved_model:
            self._load_model_config()
        trunk, server = self._initialize_server(filename)
        clients = self._initialize_clients(trunk, filename)

        return trunk, server, clients

    def _save_models(self, server, clients, filename="model.pkl"):
        print("Saving trained models...")
        server.save_model(path.join(self.save_path, "server"), filename)
        for i, c in enumerate(clients):
            c.save_model(path.join(self.save_path, str(i)), filename)
        print("Save model config...")
        model_dict = {**self.model_config.__dict__}
        # we don't want to overwrite the seed value after re-loading the model
        del model_dict["seed"]
        with open(path.join(self.save_path, "model_config.json"), "w") as f:
            json.dump(model_dict, f, cls=ModelConfigEncoder)

    def _register_compression(self, parameters, clients):
        self.compressor = GradientCompressor(parameters=[p for p in parameters() if p.requires_grad],
                                             kind=self._get_compression_method(),
                                             compression_parameter=self.model_config.compression_parameter)
        for client in clients:
            client.register_train(subscriber=self.compressor)

    def _initialize_server(self, filename="model.pkl"):
        trunk = Trunk(self.model_config)
        server = Server(trunk, conf=self.model_config)
        if self.load_saved_model:
            server.load_model(path.join(self.save_path, "server"), filename)
        return trunk, server

    def _load_model_config(self):
        with open(path.join(self.save_path, "model_config.json"), "r") as f:
            attr_dict = json.load(f)
            self.model_config.__dict__.update(attr_dict)

    def _initialize_clients(self, trunk, filename="model.pkl"):
        partner_data = du.load_partner_data("data", range(10))
        clients = []
        for i, (x, y) in enumerate(partner_data):
            self.model_config.output_size = y.shape[1]
            self.model_config.batch_size = int(x.shape[0] * self.model_config.batch_ratio)
            model = TrunkAndHead(trunk, self.model_config)
            dataset = sc.SparseDataset(x, y)
            client = Client(model, conf=self.model_config, dataset=dataset)
            if self.load_saved_model:
                client.load_model(path.join(self.save_path, str(i)), filename)
            clients.append(client)
        return clients

    def _train_targeted_model(self, clients, server, rounds, save_intermediate=False):
        for r in tqdm(range(1, rounds+1)):
            self._train_clients(clients)
            if self.compress:
                self.compressor.compress_and_set()
            # server updates his trunk
            server.update_weights()
            # server zeroes his trunk
            server.zero_grad()

            rounds_in_epoch = int(1 / self.model_config.batch_ratio)
            if save_intermediate and r % rounds_in_epoch == 0:  # save after every epoch
                self._save_models(server, clients, filename="model_%d.pkl" % (r//rounds_in_epoch))


    def _train_clients(self, clients, compress=False):
        for i, client in enumerate(clients):
            # client gets his next batch
            batch = client.get_next_batch()
            # client calculates updates for trunk+head on that batch
            client.train(batch)
            if compress:
                self.compressor.compress_and_set()
            # client updates his head
            client.update_weights()
            # client zeroes his head
            client.zero_grad()

    def _evaluate_attack(self, predictions, true_labels):
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        self.results_dict["TN"], self.results_dict["FP"], self.results_dict["FN"], self.results_dict[
            "TP"] = tn, fp, fn, tp
        self.results_dict["accuracy"] = accuracy_score(true_labels, predictions)
        self.results_dict["precision"] = precision_score(true_labels, predictions)
        self.results_dict["recall"] = recall_score(true_labels, predictions)

    def _evaluate_model(self, clients):
        auc_prs = []
        loglosses = []
        for client in clients:
            loss, auc_pr = client.eval(on_train=True)
            auc_prs.append(auc_pr)
            loglosses.append(loss)
        self.results_dict["model_avg_auc_pr"] = np.mean(auc_prs)
        self.results_dict["model_avg_logloss"] = np.mean(loglosses)


class TrunkActivationAttack(BaseAttack):
    def __init__(self, attack_config, model_config, results_path, load_saved_model, model_save, save_path):
        super().__init__(attack_config, model_config, results_path, load_saved_model, model_save, save_path)

    def run_attack(self):
        print("Trunk Activation Attack")
        trunk, server, clients = self._initialize_and_train_targeted_model()
        x_member, _, x_non_member, _ = du.load_member_non_member_data("data", range(10))

        y_pred, y_test = self._attack_on_trunk(trunk, x_member, x_non_member)

        self._evaluate_model(clients)
        self._evaluate_attack(y_pred, y_test)

        print("Attack results:")
        print(self.results_dict)

    def _attack_on_trunk(self, trunk, x_member, x_non_member):
        activation_shape = self.model_config.hidden_sizes[0]
        num_samples = min(self.attack_config.num_samples, x_member.shape[0])

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

        return y_pred, y_test


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


class ActiveTrunkActivationAttack(TrunkActivationAttack):
    def __init__(self, attack_config, model_config, results_path, load_saved_model, model_save, save_path):
        super().__init__(attack_config, model_config, results_path, load_saved_model, model_save, save_path)

    def run_attack(self):
        print("Active Trunk Activation Attack")
        isolated_trunks, adversarial_server, clients = self._initialize_and_train_isolated_models()
        y_preds, y_tests = [], []
        # performing Trunk Activation Attack on isolated trunks
        for i, trunk in enumerate(isolated_trunks):
            # training and non-training data of specific partner i
            x_member, _, x_non_member, _ = du.load_member_non_member_data("data", [i])

            y_individual_pred, y_individual_test = self._attack_on_trunk(trunk, x_member, x_non_member)
            y_preds.append(y_individual_pred)
            y_tests.append(y_individual_test)

        y_pred = np.concatenate(y_preds)
        y_test = np.concatenate(y_tests)

        self._evaluate_model(clients)
        self._evaluate_attack(y_pred, y_test)

        print("Attack results:")
        print(self.results_dict)

    def _initialize_and_train_isolated_models(self):
        print("Initialize and train isolated models...")
        trunks, server = self._initialize_server()
        clients = self._initialize_clients(trunks)
        if self.compress:
            for trunk in trunks:
                self._register_compression(trunk.parameters(), clients)

        if not self.load_saved_model:
            print("Train targeted model...")
            rounds = self.model_config.rounds
            self._train_targeted_model(clients=clients, server=server, rounds=rounds)
            if self.model_save:
                self._save_models(server, clients)

        return trunks, server, clients

    def _initialize_server(self):
        trunk = Trunk(self.model_config)
        server = IsolatingServer(num_clients=10, model=trunk, conf=self.model_config)
        if self.load_saved_model:
            server.load_model(path.join(self.save_path, "server"))
        return server.get_isolated_models(), server

    def _initialize_clients(self, individual_trunks):
        partner_data = du.load_partner_data("data", range(10))
        clients = []
        for i, (x, y) in enumerate(partner_data):
            self.model_config.output_size = y.shape[1]
            self.model_config.batch_size = int(x.shape[0] * self.model_config.batch_ratio)
            model = TrunkAndHead(individual_trunks[i], self.model_config)
            print("Model of %d. partner" % i)
            print(model)
            dataset = sc.SparseDataset(x, y)
            client = Client(model, conf=self.model_config, dataset=dataset)
            if self.load_saved_model:
                client.load_model(path.join(self.save_path, str(i)))
            clients.append(client)
        return clients


class NGMAttack(BaseAttack):
    def __init__(self, attack_config, model_config, results_path, load_saved_model, model_save, save_path):
        super().__init__(attack_config, model_config, results_path, load_saved_model, model_save, save_path)

    def run_attack(self):
        print("Naive Gradient Attack")
        trunk, server, clients = self._initialize_and_train_targeted_model()

        num_samples = self.attack_config.num_samples

        # collect predictions on "positive" samples
        predictions_on_positive_samples = []
        for _ in tqdm(range(num_samples)):
            x_member = self._train_clients_collect_batches(clients, compress=self.compress)

            sample = du.get_random_sample(x_member)
            sample = sparse.csr_matrix(sample.to_dense().numpy())
            trunk_weight_gradient = sparse.csr_matrix(server.get_gradients("trunk.net_freq")[0].flatten())
            trunk_hidden_size = self.model_config.hidden_sizes[0]
            prediction = self._naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            # prediction = self._fast_naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            # prediction = self._mod_naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            predictions_on_positive_samples.append(prediction)

            # delete trunk gradient
            server.zero_grad()

        _, _, x_non_member, _ = du.load_member_non_member_data("data", range(10))
        # collect predictions on "negative" samples
        predictions_on_negative_samples = []
        for _ in tqdm(range(num_samples)):
            self._train_clients(clients, compress=self.compress)

            sample = du.get_random_sample(x_non_member)
            trunk_weight_gradient = sparse.csr_matrix(server.get_gradients("trunk.net_freq")[0].flatten())
            trunk_hidden_size = self.model_config.hidden_sizes[0]
            prediction = self._naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            # prediction = self._mod_naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            predictions_on_negative_samples.append(prediction)

            # delete trunk gradient
            server.zero_grad()

        predictions = np.array(predictions_on_positive_samples + predictions_on_negative_samples)
        true_labels = np.array(num_samples * [1] + num_samples * [0])

        self._evaluate_model(clients)
        self._evaluate_attack(predictions, true_labels)

        print("Attack results:")
        print(self.results_dict)

    def _naive_majority_vote_attack(self, sample, grad, hidden_size):
        voting_threshold = self.attack_config.voting_threshold
        nnz_indices = sample.indices
        target_is_in = True
        grad_nnz = grad.indices.astype(np.uint32)
        for nnz_index in nnz_indices:
            # nnz = np.array([int(nnz_index * hidden_size + i) for i in range(hidden_size)], dtype=np.uint32)
            nnz = np.arange(nnz_index * hidden_size, (nnz_index + 1) * hidden_size, dtype=np.uint32)
            # second parameter of intersect needs to be the larger array!
            num_matches = len(intersect(nnz, grad_nnz))
            if num_matches < int(hidden_size * voting_threshold):
                target_is_in = False
                break
        return int(target_is_in)

    def _fast_naive_majority_vote_attack(self, sample, grad, hidden_size):
        voting_threshold = self.attack_config.voting_threshold
        nnz_indices = sample.indices

        return match(nnz_indices, grad.indices, hidden_size, voting_threshold)

    def _mod_naive_majority_vote_attack(self, sample, grad, hidden_size):
        voting_threshold = self.attack_config.voting_threshold
        nnz_indices = sample.indices

        return voting_threshold <= len(grad.indices) / (len(nnz_indices) * hidden_size)

    def _train_clients_collect_batches(self, clients, compress=False):
        batches = []
        for i, client in enumerate(clients):
            # client gets his next batch
            batch = client.get_next_batch()
            x = torch.sparse_coo_tensor(
                batch["x_ind"],
                batch["x_data"],
                size=[batch["batch_size"], self.model_config.input_size])
            batches.append(x)
            # client calculates updates for trunk+head on that batch
            client.train(batch)
            if compress:
                self.compressor.compress_and_set()
            # client updates his head
            client.update_weights()
            # client zeroes his head
            client.zero_grad()

        return torch.cat(batches, dim=0)


class LeavingAttack(NGMAttack):
    def __init__(self, attack_config, model_config, results_path, load_saved_model, model_save, save_path):
        super().__init__(attack_config, model_config, results_path, load_saved_model, model_save, save_path)

    def run_attack(self):
        print("Leaving / N-1 Attack")
        trunk, server, clients = self._initialize_and_train_targeted_model()

        num_epochs = self.attack_config.num_epochs

        leaving_partner_idx = 6
        x_member, _, _, _ = du.load_member_non_member_data("data", range(10))
        sample = x_member[26822]

        # rounds where all clients train together
        together = []
        for _ in tqdm(range(num_epochs * 50)):
            self._train_clients(clients, compress=True)

            trunk_weight_gradient = sparse.csr_matrix(server.get_gradients("trunk.net_freq")[0][2945])
            trunk_hidden_size = self.model_config.hidden_sizes[0]
            prediction = self._naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            together.append(prediction)

            # delete trunk gradient
            server.zero_grad()
        together = np.array(together)

        # rounds where all BUT the first client train together
        alone = []
        for _ in tqdm(range(num_epochs * 50)):
            self._train_clients(clients[:leaving_partner_idx] + clients[leaving_partner_idx + 1:])

            trunk_weight_gradient = sparse.csr_matrix(server.get_gradients("trunk.net_freq")[0][2945])
            trunk_hidden_size = self.model_config.hidden_sizes[0]
            prediction = self._naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            alone.append(prediction)

            # delete trunk gradient
            server.zero_grad()
        alone = np.array(alone)

        pos_epochs_alone = 0
        pos_epochs_together = 0

        for k in range(num_epochs):
            if len(alone[k * 50:(k + 1) * 50].nonzero()[0]) >= 7:
                pos_epochs_alone += 1
            if len(together[k * 50:(k + 1) * 50].nonzero()[0]) >= 7:
                pos_epochs_together += 1

        print("Total number of epochs in both cases: ", num_epochs)
        print("Number of positive epochs in together: ", pos_epochs_together)
        print("Number of positive epochs in alone: ", pos_epochs_alone)
        freq_pos_epochs_together = pos_epochs_together / num_epochs
        print("Frequency of positive epochs when together: ", freq_pos_epochs_together)

        result = stat.binom_test(pos_epochs_alone, num_epochs, p=freq_pos_epochs_together, alternative="two-sided")

        print("Probability of the two samples coming from the same distribution: %.8f" % result)

        self._evaluate_model(clients)
        self.results_dict["pos_epochs_together"] = pos_epochs_together
        self.results_dict["pos_epochs_left"] = pos_epochs_alone
        self.results_dict["p"] = result

    def _naive_majority_vote_attack(self, sample, grad, hidden_size):
        voting_threshold = self.attack_config.voting_threshold
        grad_nnz = grad.indices
        match_percent = len(grad_nnz) / hidden_size
        if match_percent < voting_threshold:
            return 0
        else:
            return 1


class MultiModelTrunkActivationAttack(TrunkActivationAttack):
    def __init__(self, attack_config, model_config, results_path, load_saved_model, model_save, save_path):
        super().__init__(attack_config, model_config, results_path, load_saved_model, model_save, save_path)
        self.model_save = True  # save the trained CP models by default

    def run_attack(self):
        print("Multi-Model Trunk Activation Attack")
        base_save_path = self.save_path
        attacked_trunks = []
        # train CP models
        for r in range(1, self.attack_config.num_models+1):
            self.model_config.lr = np.round_(np.random.random() / 10, 5)  # select random learning rate
            self.save_path = path.join(base_save_path, "CP_%d" % r)  # change save path
            trunk, server, clients = self._initialize_and_train_targeted_model(save_intermediate=True)
            attacked_trunks.append(trunk)

        if self.attack_config.best_models:
            num_epochs = self.model_config.rounds // int(1 / self.model_config.batch_ratio)
            random_client = np.random.choice(10)
            print("Selection of best model based on %d. client's auc_pr." % random_client)
            best_CP = 1
            best_epoch = 1
            best_auc_pr = 0.0
            # find best CP, best epoch
            for cp in range(1, self.attack_config.num_models+1):
                self.save_path = path.join(base_save_path, "CP_%d" % cp)  # change save path
                for e in range(1, num_epochs+1):
                    filename = "model_%d.pkl" % e
                    trunk, server, clients = self._initialize_targeted_model(filename)
                    client = clients[random_client]
                    _, auc_pr = client.eval(on_train=True)
                    if auc_pr > best_auc_pr:
                        best_CP = cp
                        best_epoch = e
            print("Best model is in CP_%d, epoch %d." % (best_CP, best_epoch))
            self.save_path = path.join(base_save_path, "CP_%d" % best_CP)  # change save path
            filename = "model_%d.pkl" % best_epoch
            trunk, server, clients = self._initialize_targeted_model(filename)
            attacked_trunks.append(trunk)  # add trunk to attacked trunks

            if self.attack_config.intermediate_models:
                # load trunks from every 2 epochs
                for e in range(1, num_epochs+1, 2):
                    filename = "model_%d.pkl" % e
                    trunk, server, clients = self._initialize_targeted_model(filename)
                    attacked_trunks.append(trunk)  # add them to attacked trunks

        # collect trunk outputs on member and non-member data
        x_member, _, x_non_member, _ = du.load_member_non_member_data("data", range(10))
        trunk_outputs = []
        activation_shape = self.model_config.hidden_sizes[0]
        num_samples = min(self.attack_config.num_samples, x_member.shape[0])
        for trunk in attacked_trunks:
            activations = self._load_activations(trunk=trunk, member_samples=x_member,
                                                 non_member_samples=x_non_member,
                                                 activation_shape=activation_shape, num_samples=num_samples)

            X = np.vstack([activations["member"][:len(activations["non-member"])], activations["non-member"]])
            membership = np.array([1] * activations["non-member"].shape[0] + [0] * activations["non-member"].shape[0])
            trunk_outputs.append(X)

        del self.model_config.lr  # delete so lr won't be logged (only latest lr would be logged)

        # concatenate
        trunk_outputs = np.concatenate(trunk_outputs, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(trunk_outputs, membership, test_size=0.33, random_state=42)

        attack_type = self.attack_config.attack_type
        if attack_type == "nn":
            y_pred, y_test = self._neural_network_attack(X_train, X_test, y_train, y_test)
        elif attack_type == "rf":
            atk = RandomForestClassifier()
            atk.fit(X_train, y_train)
            y_pred = atk.predict(X_test)
        elif attack_type == "gb":
            atk = GradientBoostingClassifier()
            atk.fit(X_train, y_train)
            y_pred = atk.predict(X_test)

        # self._evaluate_model(clients)
        self._evaluate_attack(y_pred, y_test)

        print("Attack results:")
        print(self.results_dict)

    def _neural_network_attack(self, X_train, X_test, y_train, y_test):
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                      torch.unsqueeze(torch.from_numpy(y_train).float(), dim=1))
        test_dataset = TensorDataset(torch.from_numpy(X_test).float(),
                                     torch.unsqueeze(torch.from_numpy(y_test).float(), dim=1))
        train_data_loader = DataLoader(train_dataset, batch_size=32)
        test_data_loader = DataLoader(test_dataset, batch_size=32)

        # ... and run through attack
        model = nn.Sequential(
            nn.Linear(in_features=X_train.shape[1], out_features=1024),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )
        optimizer = torch.optim.Adam(params=model.parameters())
        loss_func = nn.BCEWithLogitsLoss()

        print("Train attacker model...")
        print(model)
        model.train()
        for _ in tqdm(range(self.attack_config.num_epochs)):
            for (x, y) in train_data_loader:
                optimizer.zero_grad()
                out = model(x)
                loss = loss_func(out, y)
                loss.backward()
                optimizer.step()

        model.eval()
        y_pred, y_test = [], []
        for (x, y) in tqdm(test_data_loader):
            pred = model(x)
            pred = (pred > 0.5).int().squeeze().tolist()
            y = y.int().squeeze().tolist()
            y_pred.extend(pred)
            y_test.extend(y)

        return y_pred, y_test
