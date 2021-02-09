import torch
import sparsechem as sc
from torch.utils.data import DataLoader
import itertools as it
import pickle


class Participant:
    def __init__(self,
                 model,
                 conf,
                 dataset,
                 dataset_va=None,
                 sampler=None,
                 loss=None,
                 optimizer=None,
                 dev="cpu"):
        self.model = model.to(dev)
        self.conf = conf
        self.loss = loss
        self.optimizer = optimizer
        # create loader
        if sampler is not None:
            ## data loader with custom sampler
            self.data_loader = DataLoader(dataset, sampler=sampler, batch_size=conf.batch_size, num_workers=0,
                                          collate_fn=sc.sparse_collate, drop_last=False)
            if dataset_va:
                self.data_loader_va = DataLoader(dataset_va, sampler=sampler, batch_size=conf.batch_size, num_workers=0,
                                                 collate_fn=sc.sparse_collate, drop_last=False)
        else:
            ## data loader without custom sampler
            self.data_loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=0,
                                          collate_fn=sc.sparse_collate, shuffle=True, drop_last=False)
            if dataset_va:
                self.data_loader_va = DataLoader(dataset_va, batch_size=conf.batch_size, num_workers=0,
                                                 collate_fn=sc.sparse_collate, drop_last=False)
        self.cyclic_loader = it.cycle(iter(self.data_loader))
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none") if loss is None else loss
        self.dev = dev

    def get_next_batch(self):
        return next(self.cyclic_loader)

    def train(self, b):
        self.model.train()

        X = torch.sparse_coo_tensor(
            b["x_ind"],
            b["x_data"],
            size=[b["batch_size"], self.conf.input_size],
            device=self.dev)
        y_ind = b["y_ind"].to(self.dev)
        y_data = b["y_data"].to(self.dev)
        y_data = (y_data + 1) / 2.0

        ## [batch_size x 2808] matrix with predictions - yhat_all
        yhat_all = self.model(X)
        ## yhat (1D) - 1 where the prediction matches the label, 0 when FN
        yhat = yhat_all[y_ind[0], y_ind[1]]

        ## average loss of data
        output = self.loss(yhat, y_data).sum()
        ## average loss on one data point
        output_n = output / b["batch_size"]

        ## computes gradients
        output_n.backward()

    def eval(self, on_train=True):
        self.model.eval()
        if not self.loss:
            raise RuntimeError("No loss function was given. Cannot evaluate without loss function.")
        if on_train:
            results = sc.evaluate_binary(self.model, self.data_loader, self.loss, self.dev)
        else:
            if not self.data_loader_va:
                raise RuntimeWarning("There is no validation dataset to evaluate on. Skipping evaluation.")
            else:
                results = sc.evaluate_binary(self.model, self.data_loader_va, self.loss, self.dev)
        aucs = results["aucs"].mean()
        print(
            f"\tloss={results['logloss']:.5f}\taucs={aucs:.5f}")

        return results['logloss'], aucs

    def update_weights(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_module(self, module_name):
        for name, m in self.model.named_modules():
            if name == module_name:
                return m

    def get_gradients(self, module_name):
        grads = []
        module = self.get_module(module_name)
        for p in module.parameters():
            if p.requires_grad:
                grad = p.grad.numpy()
                grads.append(grad)
        return grads

    def get_weights(self, module_name):
        weights = []
        module = self.get_module(module_name)
        for p in module.parameters():
            if p.requires_grad:
                weight = p.data.numpy()
                weights.append(weight)
        return weights


class Server(Participant):
    def __init__(self, model, conf, dataset, sampler=None, loss=None):
        # set trunk optimizer
        optimizer = torch.optim.Adam(model.trunk.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        super().__init__(model=model, conf=conf, dataset=dataset, sampler=sampler, loss=loss, optimizer=optimizer)


class Client(Participant):
    def __init__(self, model, conf, dataset, dataset_va=None, sampler=None, loss=None):
        # set head optimizer
        optimizer = torch.optim.Adam(model.head.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        super().__init__(model=model, conf=conf, dataset=dataset, dataset_va=dataset_va, sampler=sampler, loss=loss,
                         optimizer=optimizer)

    def save_client_model(self, path, filename="model.pkl"):
        with open(path + filename, "wb") as f:
            pickle.dump(self.model, f)

    def init_parameters(self):
        # this initializes the head parameters anew
        self.model.init_weights(self.model.head)
