

from .kernel import Kernel
from .index_kernel import IndexKernel
from ..lazy import KroneckerProductLazyTensor, lazify
from ..lazy.gpatt_kronecker_lazy_tensor import GPattKroneckerProductLazyTensor


class MissingMultitaskKernel(Kernel):
    """
    Complete copy of regular Multitask kernel, but returns a GPattLazyTensor instead of a Kronecker one
    """

    def __init__(self, data_covar_module, num_tasks, rank=1, task_covar_prior=None, **kwargs):
        """"""
        super(MissingMultitaskKernel, self).__init__(**kwargs)
        self.task_covar_module = IndexKernel(
            num_tasks=num_tasks, batch_shape=self.batch_shape, rank=rank, prior=task_covar_prior
        )
        self.data_covar_module = data_covar_module
        self.num_tasks = num_tasks

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        covar_i = self.task_covar_module.covar_matrix
        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        covar_x = lazify(self.data_covar_module.forward(x1, x2, **params))
        lt_kron_prod = KroneckerProductLazyTensor(covar_x, covar_i)
        res = GPattKroneckerProductLazyTensor(lt_kron_prod)
        return res.diag() if diag else res

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks
