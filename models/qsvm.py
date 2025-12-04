import numpy as np
import pennylane as qml
from pennylane.templates import (
    AmplitudeEmbedding,
)


def get_kernel_matrix_func(n_qubits: int, seed: int = 42):
    np.random.seed(seed)

    dev_kernel = qml.device("default.qubit", wires=n_qubits)

    projector = np.zeros((2**n_qubits, 2**n_qubits))
    projector[0, 0] = 1

    @qml.qnode(dev_kernel)
    def kernel(x1, x2):
        """The quantum kernel."""
        AmplitudeEmbedding(x1, wires=range(n_qubits), pad_with=0)
        qml.adjoint(AmplitudeEmbedding)(x2, wires=range(n_qubits), pad_with=0)
        return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))  # type: ignore

    def get_kernel_matrix(A, B):
        """Compute the matrix whose entries are the kernel
        evaluated on pairwise data from sets A and B."""
        return np.array([[kernel(a, b) for b in B] for a in A])

    return get_kernel_matrix
