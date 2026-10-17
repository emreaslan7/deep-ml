import torch

def svd_2x2_singular_values(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 torch tensor
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
    """
    # Your code here

    B = A.T @ A                                                  # Gram matrix
    theta = 0.5 * torch.atan2(2 * B[0, 1], B[0, 0] - B[1, 1])    # Jacobi angle
    c, s = torch.cos(theta), torch.sin(theta)

    V = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])  # rotation
    AV = A @ V                                                   # orthogonal cols
    S = AV.norm(dim=0)                                           # = U * S (desc.)
    U = AV / torch.where(S > 0, S, torch.ones_like(S))           # guard S == 0
    return U, S, V.T
    pass