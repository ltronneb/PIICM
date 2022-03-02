#!/usr/bin/env python3
from typing import Tuple, Optional

import torch
from torch import Tensor

from .non_lazy_tensor import lazify
from .lazy_tensor import LazyTensor
from . import InterpolatedLazyTensor
from ..utils import sparse


class PermutationLazyTensor(InterpolatedLazyTensor):
    def __init__(
        self,
        base_lazy_tensor,
        left_interp_indices=None,
        left_interp_values=None,
        right_interp_indices=None,
        right_interp_values=None,
    ):
        base_lazy_tensor = lazify(base_lazy_tensor)
        #print("base_lazy_tensor.device:", base_lazy_tensor.device)
        #print("left_interp_indices.device:", left_interp_indices.device)
        #print("left_interp_values.device:", left_interp_values.device)
        #print("right_interp_indices.device:", right_interp_indices.device)
        #print("right_interp_values.device:", right_interp_values.device)
        

        if left_interp_indices is None:
            num_rows = base_lazy_tensor.size(-2)
            left_interp_indices = torch.arange(0, num_rows, dtype=torch.long, device=base_lazy_tensor.device)
            left_interp_indices.unsqueeze_(-1)
            left_interp_indices = left_interp_indices.expand(*base_lazy_tensor.batch_shape, num_rows, 1)

        if left_interp_values is None:
            left_interp_values = torch.ones(
                left_interp_indices.size(), dtype=base_lazy_tensor.dtype, device=base_lazy_tensor.device
            )

        if right_interp_indices is None:
            num_cols = base_lazy_tensor.size(-1)
            right_interp_indices = torch.arange(0, num_cols, dtype=torch.long, device=base_lazy_tensor.device)
            right_interp_indices.unsqueeze_(-1)
            right_interp_indices = right_interp_indices.expand(*base_lazy_tensor.batch_shape, num_cols, 1)

        if right_interp_values is None:
            right_interp_values = torch.ones(
                right_interp_indices.size(), dtype=base_lazy_tensor.dtype, device=base_lazy_tensor.device
            )

        if left_interp_indices.shape[:-2] != base_lazy_tensor.batch_shape:
            try:
                base_lazy_tensor = base_lazy_tensor._expand_batch(left_interp_indices.shape[:-2])
            except RuntimeError:
                raise RuntimeError(
                    "interp size ({}) is incompatible with base_lazy_tensor size ({}). ".format(
                        right_interp_indices.size(), base_lazy_tensor.size()
                    )
                )

        super(PermutationLazyTensor, self).__init__(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        self.base_lazy_tensor = base_lazy_tensor
        self.left_interp_indices = left_interp_indices
        self.left_interp_values = left_interp_values
        self.right_interp_indices = right_interp_indices
        self.right_interp_values = right_interp_values

    def __add__(self, other):
        if isinstance(other, PermutationLazyTensor):
            from .sum_permutation_lazy_tensor import SumPermutationLazyTensor
            return SumPermutationLazyTensor(self, other)
        return super().__add__(other)

    def _matmul(self, rhs):
        # Overwrite the Interpolatedlazytensor's matmul, we don't need interpolation
        # but the rest of the machinery is fine
        # Get sparse tensor representations of left/right interp matrices
        left_interp_t = self._sparse_left_interp_t(self.left_interp_indices, self.left_interp_values)
        right_interp_t = self._sparse_right_interp_t(self.right_interp_indices, self.right_interp_values)

        if rhs.ndimension() == 1:
            is_vector = True
            rhs = rhs.unsqueeze(-1)
        else:
            is_vector = False

        # right_interp^T * rhs
        right_interp_res = sparse.bdsmm(right_interp_t, rhs)

        # base_lazy_tensor * right_interp^T * rhs
        base_res = self.base_lazy_tensor._matmul(right_interp_res)

        # left_interp * base_lazy_tensor * right_interp^T * rhs
        left_interp_mat = left_interp_t.transpose(-1, -2)
        res = sparse.bdsmm(left_interp_mat, base_res)

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    #def evaluate(self):
       # # Get sparse tensor representations of left/right interp matrices
       # left_interp_t = self._sparse_left_interp_t(self.left_interp_indices, self.left_interp_values)
       # right_interp_t = self._sparse_right_interp_t(self.right_interp_indices, self.right_interp_values)
       # rhs = torch.eye(self.base_lazy_tensor.size(-1))
       # # right_interp^T * rhs
       # right_interp_res = sparse.bdsmm(right_interp_t, rhs)
       # # base_lazy_tensor * right_interp^T * rhs
       # base_res = self.base_lazy_tensor._matmul(right_interp_res)
       # # left_interp * base_lazy_tensor * right_interp^T * rhs
       # left_interp_mat = left_interp_t.transpose(-1, -2)
       # res = sparse.bdsmm(left_interp_mat, base_res)
       # return res

    def _symeig(self, eigenvectors: bool = False) -> Tuple[Tensor, Optional["LazyTensor"]]:
        raise NotImplementedError("PermutationLazyTensor does not allow for symeig to be called. "
                                  "Permuted matrix might not be hermitian!")





