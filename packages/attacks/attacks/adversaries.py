import copy
from collaborative import Server
import torch
import os
from os import path


class IsolatingServer(Server):
    def __init__(self, num_clients, model, conf, dataset=None, sampler=None, loss=None):
        super().__init__(model, conf, dataset, sampler, loss)
        self.isolated_models = [copy.deepcopy(model) for _ in range(num_clients)]
        self.isolated_model_optimizers = []
        self.isolated_model_schedulers = []

        lr_steps = [s * 50 for s in conf.lr_steps]
        for model in self.isolated_models:
            optimizer = torch.optim.Adam(model.trunk.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
            self.isolated_model_optimizers.append(optimizer)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=conf.lr_alpha)
            self.isolated_model_schedulers.append(scheduler)

    def get_isolated_models(self):
        return self.isolated_models

    def update_weights(self):
        super().update_weights()
        for optimizer, scheduler in zip(self.isolated_model_optimizers, self.isolated_model_schedulers):
            optimizer.step()
            scheduler.step()

    def save_model(self, save_path, filename="model.pkl"):
        super().save_model(save_path, filename)
        for i, model in enumerate(self.isolated_models):
            save_path = path.join(save_path, str(i))
            if not path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), path.join(save_path, filename))

    def load_model(self, save_path, filename="model.pkl"):
        super().load_model(save_path, filename)
        for i, model in enumerate(self.isolated_models):
            save_path = path.join(save_path, str(i))
            state_dict = torch.load(path.join(save_path, filename))
            model.load_state_dict(state_dict)
