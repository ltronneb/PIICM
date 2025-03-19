#!/usr/bin/env python3

from .kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from .diag_lazy_tensor import DiagLazyTensor


class GPattKroneckerProductLazyTensor(KroneckerProductLazyTensor):
    """
    Simple class to wrap a Kroneckerproduct such that adding a diagonal ensures it becomes a
    GPattAddedDiagLazyTensor -- which have custom log-determinant calculation and preconditioner
    """
    def __init__(self, input):
        if not isinstance(input, KroneckerProductLazyTensor):
            raise RuntimeError("The GPattKroneckerProductLazyTensor can only wrap a KroneckerProductLazyTensor")
        super().__init__(input)

    def __add__(self, other):
        if isinstance(other, DiagLazyTensor):
            from .gpatt_added_diag_lazy_tensor import GPattAddedDiagLazyTensor
            return GPattAddedDiagLazyTensor(self, other)
        elif isinstance(other, GPattKroneckerProductLazyTensor):
            from .gpatt_kronecker_sum_lazy_tensor import GPattKroneckerSumLazyTensor
            return GPattKroneckerSumLazyTensor(self, other)
        else:
            raise RuntimeError("Invalid addition")