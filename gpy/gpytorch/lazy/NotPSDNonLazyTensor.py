#!/usr/bin/env python3
from typing import Tuple, Optional

import torch
from torch import Tensor

from .lazy_tensor import LazyTensor
from .. import settings


class NotPSDNonLazyTensor(LazyTensor):
    def _check_args(self, tsr):
        if not torch.is_tensor(tsr):
            return "NotPSDNonLazyTensor must take a torch.Tensor; got {}".format(tsr.__class__.__name__)
        if tsr.dim() < 2:
            return "NotPSDNonLazyTensor expects a matrix (or batches of matrices) - got a Tensor of size {}.".format(
                tsr.shape
            )

    def __init__(self, tsr):
        """
        Not a lazy tensor

        Args:
        - tsr (Tensor: matrix) a Tensor
        """
        super(NotPSDNonLazyTensor, self).__init__(tsr)
        self.tensor = tsr
        
    def _cholesky_solve(self, rhs, upper=False):
        return torch.cholesky_solve(rhs, self.evaluate(), upper=upper)

    def _expand_batch(self, batch_shape):
        return self.__class__(self.tensor.expand(*batch_shape, *self.matrix_shape))

    def _get_indices(self, row_index, col_index, *batch_indices):
        # Perform the __getitem__
        res = self.tensor[(*batch_indices, row_index, col_index)]
        return res

    def _getitem(self, row_index, col_index, *batch_indices):
        # Perform the __getitem__
        res = self.tensor[(*batch_indices, row_index, col_index)]
        return self.__class__(res)

    def _matmul(self, rhs):
        return torch.matmul(self.tensor, rhs)

    def _prod_batch(self, dim):
        return self.__class__(self.tensor.prod(dim))

    def _quad_form_derivative(self, left_vecs, right_vecs):
        res = left_vecs.matmul(right_vecs.transpose(-1, -2))
        return (res,)

    def _size(self):
        return self.tensor.size()

    def _sum_batch(self, dim):
        return self.__class__(self.tensor.sum(dim))

    def _transpose_nonbatch(self):
        return NotPSDNonLazyTensor(self.tensor.transpose(-1, -2))

    def _t_matmul(self, rhs):
        return torch.matmul(self.tensor.transpose(-1, -2), rhs)

    def diag(self):
        if self.tensor.ndimension() < 3:
            return self.tensor.diag()
        else:
            row_col_iter = torch.arange(0, self.matrix_shape[-1], dtype=torch.long, device=self.device)
            return self.tensor[..., row_col_iter, row_col_iter].view(*self.batch_shape, -1)

    def evaluate(self):
        return self.tensor

    def __add__(self, other):
        if isinstance(other, NotPSDNonLazyTensor):
            return NotPSDNonLazyTensor(self.tensor + other.tensor)
        elif isinstance(other, torch.Tensor):
            return NotPSDNonLazyTensor(self.tensor + other)
        else:
            return super(NotPSDNonLazyTensor, self).__add__(other)

    def mul(self, other):
        if isinstance(other, NotPSDNonLazyTensor):
            return NotPSDNonLazyTensor(self.tensor * other.tensor)
        else:
            return super(NotPSDNonLazyTensor, self).mul(other)

    def _symeig(
            self, eigenvectors: bool = False, return_evals_as_lazy: bool = False
    ) -> Tuple[Tensor, Optional[LazyTensor]]:
        """
        Method that allows implementing special-cased symeig computation. Should not be called directly
        Copy of lazy_tensor._symeig but does not clamp eigenvalues to zero
        """
        from gpytorch.lazy.non_lazy_tensor import NonLazyTensor

        if settings.verbose_linalg.on():
            settings.verbose_linalg.logger.debug(f"Running symeig on a NotPSDNonLazyTensor of size {self.shape}. "
                                                 f"Assumed Hermitian. Not clamping eigenvalues to zero")
        # potentially perform decomposition in double precision for numerical stability
        dtype = self.dtype
        if settings.use_eigvalsh.on():
            evals = torch.linalg.eigvalsh(self.evaluate().to(dtype=settings._linalg_dtype_symeig.value()))
            evecs = None
        else:
            evals, evecs = torch.linalg.eigh(self.evaluate().to(dtype=settings._linalg_dtype_symeig.value()))
            if eigenvectors:
                evecs = NonLazyTensor(evecs.to(dtype=dtype))
            else:
                evecs = None
        return evals, evecs


def notpsdlazify(obj):
    """
    A function which ensures that `obj` is a NotPSDLazyTensor.

    If `obj` is a LazyTensor, this function does nothing.
    If `obj` is a (normal) Tensor, this function wraps it with a `NonLazyTensor`.
    """

    if torch.is_tensor(obj):
        return NotPSDNonLazyTensor(obj)
    elif isinstance(obj, NotPSDNonLazyTensor):
        return obj
    else:
        raise TypeError("object of class {} cannot be made into a NotPSDNonLazyTensor".format(obj.__class__.__name__))


__all__ = ["NotPSDNonLazyTensor", "notpsdlazify"]
