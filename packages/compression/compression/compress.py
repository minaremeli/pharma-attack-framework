import torch
from torch.nn.parameter import Parameter
from enum import Enum
from .compression_methods import TopK, Threshold, RandomSubset, Quantize_f16
from typing import List
import random


class CompressionMethods(Enum):
    THRESHOLD = "threshold"
    TOP_K = "top-k"
    RAND_SUB = "random subset"
    QUANTIZATION = "quantization"


compression_class = {CompressionMethods.THRESHOLD: Threshold,
                     CompressionMethods.TOP_K: TopK,
                     CompressionMethods.RAND_SUB: RandomSubset,
                     CompressionMethods.QUANTIZATION: Quantize_f16}


class GradientCompressor:
    def __init__(self, parameters: List[Parameter], kind: CompressionMethods, compression_parameter: float = None, device="cpu"):
        self.dev = device
        self.kind = kind
        if kind in CompressionMethods:
            if compression_parameter is not None:
                self.compression_method = compression_class[kind](device, compression_parameter)
            else:
                self.compression_method = compression_class[kind](device)
        else:
            raise ValueError("No such compression method exists. The following are available: ",
                             CompressionMethods.__members__)

        self.model_parameters = parameters
        self.prev_aggr_grads = None
        self.individual_gradients = []
        self.individual_errors = []

    def update(self):
        if not self.prev_aggr_grads:
            self.prev_aggr_grads = [p.grad.clone().detach().to(self.dev) for p in self.model_parameters]
            self.individual_gradients.append(self.prev_aggr_grads)
        else:
            current_aggr_grads = [p.grad for p in self.model_parameters]
            self.individual_gradients.append(list(map(torch.sub, current_aggr_grads, self.prev_aggr_grads)))
            self.individual_errors.append([torch.zeros(grad.shape).to(self.dev) for grad in current_aggr_grads])
            self.prev_aggr_grads = [grad.clone().detach().to(self.dev) for grad in current_aggr_grads]

    def _aggregate_individual_gradients(self):
        aggr_grads = [torch.zeros(grad.shape).to(self.dev) for grads in self.individual_gradients for grad in grads]
        for grads in self.individual_gradients:
            aggr_grads = list(map(lambda g1, g2: g1 + g2, aggr_grads, grads))
        return aggr_grads

    def _compress(self, error_propagation):
        for i, (grads, errors) in enumerate(zip(self.individual_gradients, self.individual_errors)):
            if error_propagation:
                it = map(lambda g, e: self.compression_method.compress_error_prop(g, e), grads, errors)
            else:
                it = map(lambda g: self.compression_method.compress(g), grads)
            client_grads_errors = list(map(list, zip(*it)))
            self.individual_gradients[i] = client_grads_errors[0]
            self.individual_errors[i] = client_grads_errors[1]

        if self.kind == CompressionMethods.RAND_SUB:
            self.compression_method.seed = random.randint(0, 10000)

    def _set_gradients_to_compressed(self):
        if self.individual_gradients:
            new_aggr_grads = self._aggregate_individual_gradients()
            for param, new_grad in zip(self.model_parameters, new_aggr_grads):
                param.grad = new_grad

            # reset everything
            self.individual_gradients = []
            self.individual_errors = []
        else:
            raise Warning("No individual gradients were collected yet.")

    def compress_and_set(self, error_propagation=True):
        self._compress(error_propagation)
        self._set_gradients_to_compressed()

