import torch
import sparsechem as sc
from torch.utils.data import DataLoader
import itertools as it
import os
from os import path


class Participant:
    def __init__(self,
                 model,
                 conf,
                 dataset,
                 dataset_va=None,
                 sampler=None,
                 loss=None,
                 optimizer=None,
                 scheduler=None,
                 dev="cpu"):
        self.model = model.to(dev)
        self.conf = conf
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_subscribers = set()
        # create loader
        if sampler is not None:
            if dataset:
                ## data loader with custom sampler
                self.data_loader = DataLoader(dataset, sampler=sampler, batch_size=conf.batch_size, num_workers=0,
                                              collate_fn=sc.sparse_collate, drop_last=False)
            if dataset_va:
                self.data_loader_va = DataLoader(dataset_va, sampler=sampler, batch_size=conf.batch_size, num_workers=0,
                                                 collate_fn=sc.sparse_collate, drop_last=False)
        else:
            if dataset:
                ## data loader without custom sampler
                self.data_loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=0,
                                              collate_fn=sc.sparse_collate, shuffle=True, drop_last=False)
            if dataset_va:
                self.data_loader_va = DataLoader(dataset_va, batch_size=conf.batch_size, num_workers=0,
                                                 collate_fn=sc.sparse_collate, drop_last=False)
        if dataset:
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
        y_ind = b["y_ind"].to(self.dev, non_blocking=True)
        y_data = b["y_data"].to(self.dev, non_blocking=True)
        y_data = (y_data + 1) / 2.0

        if self.conf.uncertainty_weights:
            yhat_all, logvars = self.model(X)
            logvars = logvars.expand(yhat_all.shape[0], logvars.shape[0]).to(self.dev, non_blocking=True)
            logvars_sub = logvars[y_ind[0], y_ind[1]].to(self.dev, non_blocking=True)
        else:
            yhat_all = self.model(X)
        yhat = yhat_all[y_ind[0], y_ind[1]]


        ## average loss of data
        if self.conf.uncertainty_weights:
            output = (self.loss(yhat, y_data) * torch.exp(-logvars_sub) + logvars_sub/2).sum() / 5000
        else:
            output = self.loss(yhat, y_data).sum() / 5000
        ## average loss on one data point
        output_n = output / b["batch_size"]

        ## computes gradients
        output_n.backward()

        ## notifies subscribers that train() has been called
        self._update_subscribers()

    def _update_subscribers(self):
        for subscriber in self.train_subscribers:
            subscriber.update()

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
        aucs = results["metrics"]["auc_pr"].mean()
        print(
            f"\tloss={results['logloss']:.5f}\tauc_pr={aucs:.5f}")

        return results['logloss'], aucs

    def update_weights(self):
        self.optimizer.step()
        self.scheduler.step()

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

    def save_model(self, save_path, filename="model.pkl"):
        if not path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(), path.join(save_path, filename))

    def load_model(self, save_path, filename="model.pkl"):
        state_dict = torch.load(path.join(save_path, filename))
        self.model.load_state_dict(state_dict)

    def register_train(self, subscriber):
        self.train_subscribers.add(subscriber)

    def unregister_train(self, unsubscriber):
        self.train_subscribers.discard(unsubscriber)



class Server(Participant):
    def __init__(self, model, conf, dataset=None, sampler=None, loss=None):
        lr_steps = [s * 50 for s in conf.lr_steps]
        # set trunk optimizer
        optimizer = torch.optim.Adam(model.trunk.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=conf.lr_alpha)
        super().__init__(model=model, conf=conf, dataset=dataset, sampler=sampler, loss=loss, optimizer=optimizer, scheduler=scheduler)


class Client(Participant):
    def __init__(self, model, conf, dataset, dataset_va=None, sampler=None, loss=None):
        lr_steps = [s * 50 for s in conf.lr_steps]
        # set head optimizer
        optimizer = torch.optim.Adam(model.head.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=conf.lr_alpha)
        super().__init__(model=model, conf=conf, dataset=dataset, dataset_va=dataset_va, sampler=sampler, loss=loss,
                         optimizer=optimizer, scheduler=scheduler)

    def init_parameters(self):
        # this initializes the head parameters anew
        self.model.init_weights(self.model.head)
