from typing import Tuple, Optional

import torch
from torch import Tensor

from gpytorch import settings
from .sum_lazy_tensor import SumLazyTensor
from .lazy_tensor import LazyTensor


class SumPermutationLazyTensor(SumLazyTensor):

    def _symeig(
            self, eigenvectors: bool = False, return_evals_as_lazy: bool = False
    ) -> Tuple[Tensor, Optional[LazyTensor]]:
        """
        Method that allows implementing special-cased symeig computation. Should not be called directly
        Copy of lazy_tensor._symeig but does not clamp eigenvalues to zero
        """
        from gpytorch.lazy.non_lazy_tensor import NonLazyTensor

        if settings.verbose_linalg.on():
            settings.verbose_linalg.logger.debug(f"Running symeig on a SumPermutationLazyTensor of size {self.shape}. "
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
