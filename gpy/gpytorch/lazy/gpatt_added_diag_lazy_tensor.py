#!/usr/bin/env python3
import torch

from .diag_lazy_tensor import DiagLazyTensor
from .sum_lazy_tensor import SumLazyTensor
from ..utils import broadcasting


class GPattAddedDiagLazyTensor(SumLazyTensor):
    """
    This encodes the GPatt Structure for a Tensor, it ensures that solves are taken using a preconditioned
    CG that ignores missing observations, and a custom logdet calculation

    The GPattLazyTensor is constructed in two parts,
        1) a base tensor that is the kernel itself
        2) a diagonal tensor that encodes the missing observation noise and preconditioner

    The GPattAddedDiagLazyTensor is the analog of KroneckerProductAddedDiagLazyTensor for a GPattKroneckerLazyTensor
    """

    def __init__(self, *lazy_tensors):
        lazy_tensors = list(lazy_tensors)
        super(GPattAddedDiagLazyTensor, self).__init__(*lazy_tensors)
        if len(lazy_tensors) > 2:
            raise RuntimeError("A GPattAddedDiagLazyTensor must have exactly two components")

        broadcasting._mul_broadcast_shape(lazy_tensors[0].shape, lazy_tensors[1].shape)

        if isinstance(lazy_tensors[0], DiagLazyTensor) and isinstance(lazy_tensors[1], DiagLazyTensor):
            raise RuntimeError("Trying to lazily add two DiagLazyTensors. Create a single DiagLazyTensor instead.")
        elif isinstance(lazy_tensors[0], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[0]
            self._lazy_tensor = lazy_tensors[1]
        elif isinstance(lazy_tensors[1], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[1]
            self._lazy_tensor = lazy_tensors[0]
        else:
            raise RuntimeError("One of the LazyTensors input to AddedDiagLazyTensor must be a DiagLazyTensor!")
        self.missing_idx = (self._diag_tensor.diag() >= 500.).clone().detach()  # Hardcoded which is not optimal
        self.n_missing = self.missing_idx.sum()
        self.n_total = self.missing_idx.numel()
        self.n_obs = self.n_total - self.n_missing

    def add_diag(self, added_diag):
        raise RuntimeError("Tried to add diag to GPattAddedDiagLazyTensor")

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        if inv_quad_rhs is not None:
            inv_quad_term, _ = super().inv_quad_logdet(
                inv_quad_rhs=inv_quad_rhs, logdet=False, reduce_inv_quad=reduce_inv_quad
            )
        else:
            inv_quad_term = None
        logdet_term = self._logdet() if logdet else None
        return inv_quad_term, logdet_term

    def _logdet(self):
        """
        Log-determinant computed uses an approximation via Weyl's inequality
        """
        # Compute eigenvectors for gradients
        evals_unsorted, _ = self._lazy_tensor.symeig(eigenvectors=False)
        evals = evals_unsorted.sort(descending=True)[0]
        # Pull out the constant diagonal
        noise_unsorted = self._diag_tensor.diag()
        noise_unsorted = noise_unsorted.masked_fill(self.missing_idx, 0)  # Mask large variances
        noise = noise_unsorted.sort(descending=True)[0]
        # Apply Weyl's inequality
        weyl = torch.zeros(evals.shape, device=self.device)
        weyl[0::2] = evals[0::2] + noise[0::2]
        weyl[1::2] = evals[1::2] + noise[0::2]
        top_evals = (self.n_obs / self.n_total) * weyl[:self.n_obs]
        logdet = torch.log(top_evals).sum(dim=-1)
        return logdet

    def _preconditioner(self):
        def GPatt_precond(tensor):
            return tensor / self._diag_tensor.diag().sqrt().unsqueeze(-1)

        return GPatt_precond, None, None

    def _solve(self, rhs, preconditioner, num_tridiag=0):
        # CG for solves
        return super()._solve(rhs, preconditioner=preconditioner, num_tridiag=num_tridiag)

    def _matmul(self, rhs):
        return torch.addcmul(self._lazy_tensor._matmul(rhs), self._diag_tensor._diag.unsqueeze(-1), rhs)

    def __add__(self, other):
        from .diag_lazy_tensor import DiagLazyTensor

        if isinstance(other, DiagLazyTensor):
            return self.__class__(self._lazy_tensor, self._diag_tensor + other)
        else:
            raise RuntimeError("Only DiagLazyTensors can be added to a GPattAddedDiagLazyTensor!")
