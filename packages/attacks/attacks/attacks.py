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
import scipy.sparse as sparse
import scipy.stats as stat
from pyintersect import intersect, match
from os import path
import json
from .configs import ModelConfigEncoder

from sklearn.decomposition import PCA
from umap import UMAP
#from umap.parametric_umap import ParametricUMAP

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
        if self.load_saved_model:
            print("Load model config")
            with open(path.join(self.save_path, "model_config.json"), "r") as f:
                attr_dict = json.load(f)
                self.model_config.__dict__.update(attr_dict)

            print("Load saved models from %s" % self.save_path)
            trunk = Trunk(self.model_config)
            server = Server(trunk, conf=self.model_config)
            server.load_model(path.join(self.save_path, "server"))
            clients = self._initialize_clients(trunk)
        else:
            print("Train targeted model...")
            rounds = self.model_config.rounds

            trunk = Trunk(self.model_config)
            server = Server(trunk, conf=self.model_config)
            clients = self._initialize_clients(trunk)
            self._train_targeted_model(clients=clients, server=server, rounds=rounds)
            if self.model_save:
                print("Saving trained models...")
                server.save_model(path.join(self.save_path, "server"))
                for i, c in enumerate(clients):
                    c.save_model(path.join(self.save_path, str(i)))
                print("Save model config...")
                model_dict = {**self.model_config.__dict__}
                # we don't want to overwrite the seed value after re-loading the model
                del model_dict["seed"]
                with open(path.join(self.save_path, "model_config.json"), "w") as f:
                    json.dump(model_dict, f, cls=ModelConfigEncoder)

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
            if self.load_saved_model:
                client.load_model(path.join(self.save_path, str(i)))
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
        print("Trunk Activation Attack")
        super().__init__(attack_config, model_config, results_path, load_saved_model, model_save, save_path)

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
        
        print ("UMAP is performed.")
        #dim_reducer = PCA(n_components=40)
        #dim_reducer = ParametricUMAP()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        #X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
        
        dim_reducer = UMAP(n_components=50, n_neighbors=15).fit(X_train, y_train)

        X_train = dim_reducer.transform(X_train)
        X_test = dim_reducer.transform(X_test)

        n_estimators = self.attack_config.n_estimators

        atk = RandomForestClassifier(n_estimators)
        atk.fit(X_train, y_train)

        y_pred = atk.predict(X_test)

        self._evaluate_model(clients)
        self._evaluate_attack(y_pred, y_test)

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
    def __init__(self, attack_config, model_config, results_path, load_saved_model, model_save, save_path):
        super().__init__(attack_config, model_config, results_path, load_saved_model, model_save, save_path)

    def run_attack(self):
        print("Naive Gradient Attack")
        trunk, server, clients = self._initialize_and_train_targeted_model()

        num_samples = self.attack_config.num_samples

        # collect predictions on "positive" samples
        predictions_on_positive_samples = []
        for _ in tqdm(range(num_samples)):
            x_member = self._train_clients_collect_batches(clients)

            sample = du.get_random_sample(x_member)
            sample = sparse.csr_matrix(sample.to_dense().numpy())
            trunk_weight_gradient = sparse.csr_matrix(server.get_gradients("trunk.net_freq")[0].flatten())
            trunk_hidden_size = self.model_config.hidden_sizes[0]
            #prediction = self._naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            #prediction = self._fast_naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            prediction = self._mod_naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            predictions_on_positive_samples.append(prediction)

            # delete trunk gradient
            server.zero_grad()

        _, _, x_non_member, _ = du.load_member_non_member_data("data", range(10))
        # collect predictions on "negative" samples
        predictions_on_negative_samples = []
        for _ in tqdm(range(num_samples)):
            self._train_clients(clients)

            sample = du.get_random_sample(x_non_member)
            trunk_weight_gradient = sparse.csr_matrix(server.get_gradients("trunk.net_freq")[0].flatten())
            trunk_hidden_size = self.model_config.hidden_sizes[0]
            #prediction = self._naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            prediction = self._mod_naive_majority_vote_attack(sample, trunk_weight_gradient, trunk_hidden_size)
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
            #nnz = np.array([int(nnz_index * hidden_size + i) for i in range(hidden_size)], dtype=np.uint32)
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

    def _train_clients_collect_batches(self, clients):
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
            # client updates his head
            client.update_weights()
            # client zeroes his head
            client.zero_grad()

        return torch.cat(batches, dim=0)


class LeavingAttack(BaseAttack):
    def __init__(self, attack_config, model_config, results_path, load_saved_model, model_save, save_path):
        super().__init__(attack_config, model_config, results_path, load_saved_model, model_save, save_path)

    def run_attack(self):
        print("Leaving / N-1 Attack")
        trunk, server, clients = self._initialize_and_train_targeted_model()

        num_epochs = self.attack_config.num_epochs

        leaving_partner_idx = 0
        x_leaving_partner, _ = du.load_partner_data("data", [leaving_partner_idx])[0]
        print(x_leaving_partner.shape)
        sample = du.get_random_sample(x_leaving_partner)

        # rounds where all clients train together
        together = []
        for _ in tqdm(range(num_epochs*50)):
            self._train_clients(clients)

            trunk_weight_gradient = sparse.csr_matrix(server.get_gradients("trunk.net_freq")[0].flatten())
            trunk_hidden_size = self.model_config.hidden_sizes[0]
            prediction = self._naive_gradient_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            together.append(prediction)

            # delete trunk gradient
            server.zero_grad()
        together = np.array(together)
        print(together)

        # rounds where all BUT the first client train together
        alone = []
        for _ in tqdm(range(num_epochs*50)):
            self._train_clients(clients[1:])

            trunk_weight_gradient = sparse.csr_matrix(server.get_gradients("trunk.net_freq")[0].flatten())
            trunk_hidden_size = self.model_config.hidden_sizes[0]
            prediction = self._naive_gradient_attack(sample, trunk_weight_gradient, trunk_hidden_size)
            alone.append(prediction)

            # delete trunk gradient
            server.zero_grad()
        alone = np.array(alone)
        print(alone)

        pos_epochs_alone = 0
        pos_epochs_together = 0

        for k in range(num_epochs):
            if len(alone[k * 50:(k + 1) * 50].nonzero()[0]) > 0:
                pos_epochs_alone += 1
            if len(together[k * 50:(k + 1) * 50].nonzero()[0]) > 0:
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

    def _naive_gradient_attack(self, sample, grad, hidden_size):
        nnz_indices = sample.indices
        target_is_in = True
        for nnz_index in nnz_indices:
            nnz_set = set({nnz_index * hidden_size + i for i in range(hidden_size)})
            num_matches = len(nnz_set.intersection(grad.indices))
            if num_matches != hidden_size:
                target_is_in = False
                break
        return int(target_is_in)






