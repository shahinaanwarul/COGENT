import numpy as np
import torch

from cogent.graph import normalized_adjacency_and_laplacian


def test_normalized_laplacian_is_symmetric_psd():
    A = torch.tensor([[0.0, 1.0, 0.2], [1.0, 0.0, 0.3], [0.2, 0.3, 0.0]])
    _, L = normalized_adjacency_and_laplacian(A)
    assert torch.allclose(L, L.T, atol=1e-6)
    eig = torch.linalg.eigvalsh(L)
    assert float(eig.min()) > -1e-5
