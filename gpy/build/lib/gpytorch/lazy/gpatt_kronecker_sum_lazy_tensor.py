#!/usr/bin/env python3


from gpytorch.lazy import SumLazyTensor, DiagLazyTensor


class GPattKroneckerSumLazyTensor(SumLazyTensor):
    """
    Class to wrap a sum of Kronecker products, but ensure we stay inside the GPatt family of LazyTensors
    Simple extension of SumLazyTensor with custom __add__ routine, ensuring that a pass through the likelihood
    yields a GPattKroneckerSumAddedDiagLazyTensor
    """

    def __add__(self, other):
        if isinstance(other, DiagLazyTensor):
            from .gpatt_kronecker_sum_added_diag_lazy_tensor import GPattKroneckerSumAddedDiagLazyTensor
            return GPattKroneckerSumAddedDiagLazyTensor(self, other)
        else:
            raise RuntimeError("Invalid addition")
