import torch
from torch.distributions.bernoulli import Bernoulli

class BaseCompression:
    def __init__(self, dev):
        self.dev = dev

    def compress(self, grad):
        raise NotImplementedError()

    def compress_error_prop(self, grad, error):
        raise NotImplementedError()


class TopK(BaseCompression):
    def __init__(self, dev, fraction=None):
        super().__init__(dev)
        self.fraction = fraction

    def compress_error_prop(self, grad, error):
        tmp = torch.abs(torch.flatten(grad.add(error)))
        th = torch.min(torch.topk(tmp.to(self.dev), round(self.fraction * len(tmp))).values).to(self.dev)
        comp = torch.where(torch.abs(grad.add(error)).to(self.dev) >= th,
                           grad.add(error).to(self.dev),
                           torch.zeros(1).to(self.dev))
        rem = torch.add(grad, error).add(torch.neg(comp)).to(self.dev)
        return comp, rem

    def compress(self, grad):
        return self.compress_error_prop(grad, torch.zeros(grad.shape).to(self.dev))[0]


class Threshold(BaseCompression):
    def __init__(self, dev, threshold=None):
        super().__init__(dev)
        self.threshold = threshold

    def _lower_clip(self, grad):
        return grad.clamp_min(self.threshold)

    def _upper_clip(self, grad):
        return grad.clamp_max(self.threshold)

    def compress_error_prop(self, grad, error):
        comp = self._upper_clip(self._lower_clip(torch.add(grad, error))).to(self.dev)
        return comp, (torch.add(grad, error) - comp).to(self.dev)

    def compress(self, grad):
        return self.compress_error_prop(grad, torch.zeros(grad.shape).to(self.dev))[0]


class RandomSubset(BaseCompression):
    def __init__(self, dev, ratio):
        super().__init__(dev)
        self.ratio = ratio
        assert self.ratio is not None and 0 < self.ratio < 1
        self._seed = 0

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    def compress_error_prop(self, grad, error):
        bernoulli_generator = Bernoulli(torch.zeros(grad.shape).fill_(self.ratio))
        torch.manual_seed(self._seed)  # set the seed so we can always generate the same mask
        mask = bernoulli_generator.sample().to(self.dev)
        return torch.add(grad, error) * mask, torch.add(grad, error).add(torch.neg(torch.add(grad, error) * mask))

    def compress(self, grad):
        return self.compress_error_prop(grad, torch.zeros(grad.shape).to(self.dev))[0]


class Quantize_f16(BaseCompression):

    def __init__(self, dev):
        super().__init__(dev)

    def compress_error_prop(self, grad, error):
        comp = torch.add(grad, error).type(torch.HalfTensor).to(self.dev)
        rem = torch.add(grad, error).add(torch.neg(comp)).to(self.dev)
        return comp, rem

    def compress(self, grad):
        return self.compress_error_prop(grad, torch.zeros(grad.shape).to(self.dev))[0]
