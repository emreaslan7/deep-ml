import torch

def reshape_matrix(a, new_shape) -> torch.Tensor:
    if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
        return torch.tensor([])

    a_t = torch.as_tensor(a, dtype=torch.float)

    # Your implementation here

    return torch.reshape(a_t, new_shape)

    pass
