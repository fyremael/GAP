"""Linear constraint operators for protected-mode experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LinearConstraint:
    """Finite-dimensional linear constraint ``A x = 0``.

    The kernel projector is the dense orthogonal projector
    ``P_K = I - A^dagger A``. This implementation is intentionally small and
    explicit so early diagnostics can audit exactly which operator is used.
    """

    A: torch.Tensor
    rtol: float | None = 1e-6
    atol: float | None = None
    _kernel_projector: torch.Tensor | None = field(default=None, init=False, repr=False)
    _complement_projector: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.A, torch.Tensor):
            self.A = torch.as_tensor(self.A)
        if self.A.ndim != 2:
            raise ValueError("A must be a 2D tensor with shape (codim, dim).")
        if not torch.is_floating_point(self.A):
            self.A = self.A.to(torch.get_default_dtype())

    @property
    def dim(self) -> int:
        """Dimension of the state space acted on by ``A``."""

        return int(self.A.shape[1])

    @property
    def codim(self) -> int:
        """Dimension of the constraint residual ``A x``."""

        return int(self.A.shape[0])

    def to(self, *args, **kwargs) -> "LinearConstraint":
        """Return a copy of the constraint on another dtype or device."""

        return LinearConstraint(self.A.to(*args, **kwargs), rtol=self.rtol, atol=self.atol)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``A`` to row-vector states with shape ``(..., dim)``."""

        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {x.shape[-1]}.")
        A = self.A.to(device=x.device, dtype=x.dtype)
        return x @ A.T

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the Euclidean adjoint ``A^*`` to row-vector residuals."""

        if y.shape[-1] != self.codim:
            raise ValueError(f"Expected last dimension {self.codim}, got {y.shape[-1]}.")
        A = self.A.to(device=y.device, dtype=y.dtype)
        return y @ A

    def normal_matrix(self) -> torch.Tensor:
        """Return the dense normal operator ``A^* A``."""

        return self.A.T @ self.A

    def kernel_projector(self) -> torch.Tensor:
        """Return the cached dense projector onto ``ker A``."""

        self._ensure_projectors()
        assert self._kernel_projector is not None
        return self._kernel_projector

    def complement_projector(self) -> torch.Tensor:
        """Return the cached dense projector onto the row-space complement."""

        self._ensure_projectors()
        assert self._complement_projector is not None
        return self._complement_projector

    def project_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Project row-vector states onto the protected kernel ``ker A``."""

        P = self.kernel_projector().to(device=x.device, dtype=x.dtype)
        return x @ P.T

    def project_complement(self, x: torch.Tensor) -> torch.Tensor:
        """Project row-vector states onto the orthogonal complement of ``ker A``."""

        P = self.complement_projector().to(device=x.device, dtype=x.dtype)
        return x @ P.T

    def violation(
        self, x: torch.Tensor, *, relative: bool = False, eps: float = 1e-12
    ) -> torch.Tensor:
        """Return ``||A x||`` for each state, optionally normalized by ``||x||``."""

        residual = torch.linalg.norm(self.apply(x), dim=-1)
        if not relative:
            return residual
        return residual / (torch.linalg.norm(x, dim=-1) + eps)

    def _ensure_projectors(self) -> None:
        if self._kernel_projector is not None and self._complement_projector is not None:
            return
        kwargs: dict[str, float] = {}
        if self.rtol is not None:
            kwargs["rtol"] = self.rtol
        if self.atol is not None:
            kwargs["atol"] = self.atol
        A_pinv = torch.linalg.pinv(self.A, **kwargs)
        dim = self.A.shape[1]
        identity = torch.eye(dim, dtype=self.A.dtype, device=self.A.device)
        P_perp = A_pinv @ self.A
        P_kernel = identity - P_perp
        self._kernel_projector = 0.5 * (P_kernel + P_kernel.T)
        self._complement_projector = 0.5 * (P_perp + P_perp.T)


@dataclass
class ImplicitLinearConstraint:
    """Constraint projection through an implicit residual-space solve.

    This class supports dense or sparse COO tensors for ``A``. It avoids
    forming ``A^dagger`` and instead projects by solving
    ``(A A^* + damping I) lambda = A x`` with batched conjugate gradients, then
    returning ``x - A^* lambda``. It is intended as the first sparse-compatible
    scaffold, not as a production Hodge solver.
    """

    A: torch.Tensor
    damping: float = 0.0
    max_iter: int = 128
    tol: float = 1e-10
    _kernel_projector: torch.Tensor | None = field(default=None, init=False, repr=False)
    _complement_projector: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.A, torch.Tensor):
            self.A = torch.as_tensor(self.A)
        if self.A.ndim != 2:
            raise ValueError("A must be a 2D tensor with shape (codim, dim).")
        if not torch.is_floating_point(self.A):
            self.A = self.A.to(torch.get_default_dtype())
        if self.A.layout == torch.sparse_coo:
            self.A = self.A.coalesce()

    @property
    def dim(self) -> int:
        """Dimension of the state space acted on by ``A``."""

        return int(self.A.shape[1])

    @property
    def codim(self) -> int:
        """Dimension of the constraint residual ``A x``."""

        return int(self.A.shape[0])

    def to(self, *args, **kwargs) -> "ImplicitLinearConstraint":
        """Return a copy of the implicit constraint on another dtype or device."""

        return ImplicitLinearConstraint(
            self.A.to(*args, **kwargs),
            damping=self.damping,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``A`` to row-vector states with shape ``(..., dim)``."""

        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {x.shape[-1]}.")
        return _apply_matrix(self.A, x, transpose=True)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the Euclidean adjoint ``A^*`` to row-vector residuals."""

        if y.shape[-1] != self.codim:
            raise ValueError(f"Expected last dimension {self.codim}, got {y.shape[-1]}.")
        return _apply_matrix(self.A, y, transpose=False)

    def normal_matrix(self) -> torch.Tensor:
        """Return a dense ``A^* A`` matrix for low-dimensional diagnostics."""

        A = self.A.to_dense() if _is_sparse_matrix(self.A) else self.A
        return A.T @ A

    def kernel_projector(self) -> torch.Tensor:
        """Return an approximate dense projector by probing basis vectors."""

        self._ensure_projectors()
        assert self._kernel_projector is not None
        return self._kernel_projector

    def complement_projector(self) -> torch.Tensor:
        """Return an approximate dense complement projector by probing basis vectors."""

        self._ensure_projectors()
        assert self._complement_projector is not None
        return self._complement_projector

    def project_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Project row-vector states onto ``ker A`` through an implicit solve."""

        return self._project_kernel_iterative(x)

    def project_complement(self, x: torch.Tensor) -> torch.Tensor:
        """Project row-vector states onto the row-space complement."""

        residual = self.apply(x)
        lagrange = self._solve_residual_system(residual)
        return self.adjoint(lagrange)

    def violation(
        self, x: torch.Tensor, *, relative: bool = False, eps: float = 1e-12
    ) -> torch.Tensor:
        """Return ``||A x||`` for each state, optionally normalized by ``||x||``."""

        residual = torch.linalg.norm(self.apply(x), dim=-1)
        if not relative:
            return residual
        return residual / (torch.linalg.norm(x, dim=-1) + eps)

    def _project_kernel_iterative(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.apply(x)
        lagrange = self._solve_residual_system(residual)
        return x - self.adjoint(lagrange)

    def _solve_residual_system(self, b: torch.Tensor) -> torch.Tensor:
        original_shape = b.shape
        flat_b = b.reshape(-1, self.codim)
        if flat_b.numel() == 0:
            return torch.zeros_like(b)
        x = torch.zeros_like(flat_b)
        r = flat_b.clone()
        p = r.clone()
        rs_old = torch.sum(r * r, dim=-1, keepdim=True)
        if torch.sqrt(rs_old.max()) <= self.tol:
            return x.reshape(original_shape)

        for _ in range(self.max_iter):
            Ap = self._gram_apply(p)
            denom = torch.sum(p * Ap, dim=-1, keepdim=True).clamp_min(
                torch.finfo(flat_b.dtype).eps
            )
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r * r, dim=-1, keepdim=True)
            if torch.sqrt(rs_new.max()) <= self.tol:
                break
            beta = rs_new / rs_old.clamp_min(torch.finfo(flat_b.dtype).eps)
            p = r + beta * p
            rs_old = rs_new
        return x.reshape(original_shape)

    def _gram_apply(self, y: torch.Tensor) -> torch.Tensor:
        out = self.apply(self.adjoint(y))
        if self.damping:
            out = out + self.damping * y
        return out

    def _ensure_projectors(self) -> None:
        if self._kernel_projector is not None and self._complement_projector is not None:
            return
        identity = torch.eye(
            self.dim,
            dtype=self.A.dtype,
            device=self.A.device,
        )
        P_kernel = self._project_kernel_iterative(identity)
        P_complement = identity - P_kernel
        self._kernel_projector = 0.5 * (P_kernel + P_kernel.T)
        self._complement_projector = 0.5 * (P_complement + P_complement.T)


def _apply_matrix(A: torch.Tensor, x: torch.Tensor, *, transpose: bool) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    matrix = A.to(device=x.device, dtype=x.dtype)
    if _is_sparse_matrix(matrix):
        sparse_matrix = matrix.transpose(0, 1) if not transpose else matrix
        out = torch.sparse.mm(sparse_matrix, flat.T).T
    else:
        out = flat @ (matrix.T if transpose else matrix)
    return out.reshape(*x.shape[:-1], out.shape[-1])


def _is_sparse_matrix(matrix: torch.Tensor) -> bool:
    return matrix.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }
