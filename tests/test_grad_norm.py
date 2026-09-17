"""The gradient-norm log must not synchronise per parameter.

Summing `.item()` over each tensor costs a GPU->CPU sync for every one of ~470
parameters, and each stalls the pipeline. The Lightning profiler measured 1.09s per
optimizer step -- 2.5% of training time -- to log one scalar.
"""

import unittest

import torch

from modules.utils_train import LogGradientNorm


class _Module:
    def __init__(self, parameters):
        self._parameters = parameters
        self.logged = {}

    def parameters(self):
        return iter(self._parameters)

    def log(self, name, value):
        self.logged[name] = value


class _Trainer:
    def __init__(self, module):
        self.lightning_module = module


class GradientNormTests(unittest.TestCase):

    def _module(self, grads):
        parameters = []
        for grad in grads:
            parameter = torch.nn.Parameter(torch.zeros_like(grad))
            parameter.grad = grad
            parameters.append(parameter)
        return _Module(parameters)

    def test_matches_the_global_l2_norm(self):
        grads = [torch.tensor([3.0, 4.0]), torch.tensor([[12.0]])]
        module = self._module(grads)
        LogGradientNorm().on_before_optimizer_step(_Trainer(module))
        # sqrt(9 + 16 + 144) = 13
        self.assertAlmostEqual(float(module.logged["train/grad_norm"]), 13.0, places=5)

    def test_logs_a_tensor_so_nothing_synchronises(self):
        module = self._module([torch.tensor([1.0, 2.0])])
        LogGradientNorm().on_before_optimizer_step(_Trainer(module))
        self.assertIsInstance(module.logged["train/grad_norm"], torch.Tensor)

    def test_no_gradients_logs_nothing(self):
        """Before the first backward there is nothing to reduce over."""
        module = _Module([torch.nn.Parameter(torch.zeros(2))])
        LogGradientNorm().on_before_optimizer_step(_Trainer(module))
        self.assertEqual({}, module.logged)


if __name__ == "__main__":
    unittest.main()
