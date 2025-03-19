from copy import deepcopy

import torch

from .kernel import Kernel
from .index_kernel import IndexKernel
from ..lazy import KroneckerProductLazyTensor, GPattKroneckerProductLazyTensor, lazify
from ..lazy.NotPSDNonLazyTensor import notpsdlazify
from ..lazy.permutation_lazy_tensor import PermutationLazyTensor


class DrugResponseKernel(Kernel):
    """
    Implements the intrinsic coregionalization model (ICM) with or without encoded invariances
    """

    def __init__(self, data_covar_module, num_combinations, num_cell_lines, symmetric=True, drug_rank=1,
                 cell_linerank=1,
                 task_covar_prior=None,
                 **kwargs):
        super(DrugResponseKernel, self).__init__(**kwargs)
        # Check for CUDA
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        if symmetric:
            self.expanded_num_combinations = 2 * num_combinations
        else:
            self.expanded_num_combinations = num_combinations
        self.num_cell_lines = num_cell_lines
        self.drugcombo_covar_module = IndexKernel(
            num_tasks=self.expanded_num_combinations, batch_shape=self.batch_shape, rank=drug_rank,
            prior=task_covar_prior
        )
        self.cellline_covar_module = IndexKernel(
            num_tasks=self.num_cell_lines, batch_shape=self.batch_shape, rank=cell_linerank,
            prior=task_covar_prior
        )
        # If symmetric, set up permutation matrix
        if symmetric:
            interp_indices = torch.zeros((self.expanded_num_combinations, self.expanded_num_combinations),
                                         dtype=torch.long, device=dev)
            interp_values = torch.zeros((self.expanded_num_combinations, self.expanded_num_combinations),device=dev)
            colcounter = 0
            for i in range(num_combinations):
                interp_indices[colcounter, colcounter] = i + num_combinations
                interp_values[colcounter, colcounter] = 1
                colcounter += 1
            for i in range(num_combinations):
                interp_indices[colcounter, colcounter] = i
                interp_values[colcounter, colcounter] = 1
                colcounter += 1
            self.symmetric_indices = interp_indices
            self.symmetric_values = interp_values
            self.symmetric = symmetric
            # And reflection matrix
            self.reflection = torch.tensor([[0.0, 1.0], [1.0, 0.0]],device=dev)
        else:
            self.symmetric = False
        self.data_covar_module = data_covar_module
        self.num_combinations = num_combinations

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        covar_drugcombo = self.drugcombo_covar_module.covar_matrix
        covar_cellline = self.cellline_covar_module.covar_matrix
        data_covar_matrix = self.data_covar_module.forward(x1, x2, **params)
        covar_x = lazify(data_covar_matrix)
        if self.symmetric:
            # Ensure things are on correct device
            device = x1.device
            self.symmetric_indices = self.symmetric_indices.to(device)
            self.symmetric_values = self.symmetric_values.to(device)
            self.reflection = self.reflection.to(device)
            # Make some copies
            covar_drugcombo_t = covar_drugcombo.clone()
            covar_drugcombo_tt = covar_drugcombo.clone()
            covar_drugcombo_sym_row = PermutationLazyTensor(covar_drugcombo_t,
                                                            left_interp_indices=self.symmetric_indices,
                                                            left_interp_values=self.symmetric_values,
                                                            right_interp_indices=None,
                                                            right_interp_values=None)
            covar_drugcombo_sym_total = PermutationLazyTensor(covar_drugcombo_tt,
                                                              left_interp_indices=self.symmetric_indices,
                                                              left_interp_values=self.symmetric_values,
                                                              right_interp_indices=self.symmetric_indices,
                                                              right_interp_values=self.symmetric_values)
            if x1.shape[1] > 1:  # For the 2-d case we flip the axis
                data_covar_module_reflected = deepcopy(self.data_covar_module)
                data_covar_matrix_reflected = data_covar_module_reflected.forward(x1.matmul(self.reflection),
                                                                                  x2, **params)
                covar_x_reflected = notpsdlazify(data_covar_matrix_reflected)
                kron_lt1 = KroneckerProductLazyTensor(covar_x, covar_cellline,
                                                      0.25 * covar_drugcombo + 0.25 * covar_drugcombo_sym_total)
                kron_lt2 = KroneckerProductLazyTensor(covar_x_reflected, covar_cellline,
                                                      0.25 * covar_drugcombo_sym_row +
                                                      0.25 * covar_drugcombo_sym_row.t())
                res = GPattKroneckerProductLazyTensor(kron_lt1) + GPattKroneckerProductLazyTensor(kron_lt2)

            else:
                covar_k = 0.25 * covar_drugcombo + 0.25 * covar_drugcombo_sym_row + \
                          0.25 * covar_drugcombo_sym_row.t() + 0.25 * covar_drugcombo_sym_total
                kron_lt = KroneckerProductLazyTensor(covar_x, covar_k, covar_cellline)
                res = GPattKroneckerProductLazyTensor(kron_lt)
        else:
            kron_lt = KroneckerProductLazyTensor(covar_x, covar_cellline, covar_drugcombo)
            res = GPattKroneckerProductLazyTensor(kron_lt)

        return res.diag() if diag else res

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return (self.expanded_num_combinations * self.num_cell_lines)
