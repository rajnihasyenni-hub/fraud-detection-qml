"""
ibm_backend.py
Drop this next to quantum_kernel_svm.py and import from it to swap the
local AerSimulator for a real (or cloud-simulated) IBM Quantum backend.

One-time setup (run once, saves credentials to disk so you never
hardcode your token in scripts):

    pip install qiskit-ibm-runtime --break-system-packages

    python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel='ibm_quantum_platform',
    token='YOUR_IBM_QUANTUM_TOKEN',   # from quantum.ibm.com -> your account
    instance='YOUR_CRN_OR_INSTANCE',  # e.g. the CRN shown on your dashboard
    overwrite=True,
)
"

After that, QiskitRuntimeService() below picks up saved credentials
automatically — no token in your repo, safe to commit.
"""

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def get_service():
    """Loads saved IBM Quantum credentials. Run save_account() once first."""
    return QiskitRuntimeService()


def pick_backend(service, min_qubits=8, simulator_ok=False, use_least_busy=True):
    """
    Picks a backend for the fraud-detection feature map.
    min_qubits should match however many features your ZZFeatureMap encodes.
    """
    if use_least_busy:
        return service.least_busy(
            operational=True,
            simulator=simulator_ok,
            min_num_qubits=min_qubits,
        )
    # or pick a specific one you already know from the QGSS labs:
    return service.backend("ibm_fez")  # or "ibm_marrakesh"


def build_kernel(feature_map, backend, shots=2048):
    """
    Builds a FidelityQuantumKernel that runs on real IBM hardware via
    Qiskit Runtime's SamplerV2, instead of the local statevector simulator.

    NOTE: fidelity kernel computation needs O(n^2) circuit evaluations,
    so keep your quantum training subset small (this is exactly why your
    dashboard already labels it a "balanced subset" comparison, not a
    full retrain).
    """
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots

    # ComputeUncompute builds the fidelity ("overlap") circuits internally;
    # pass_manager transpiles those circuits to the backend's ISA before
    # they're submitted through the Runtime sampler.
    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)

    kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    return kernel, pm, sampler


if __name__ == "__main__":
    # Quick connectivity check you can run standalone:
    #   python ibm_backend.py
    service = get_service()
    backend = pick_backend(service, min_qubits=4)
    print(f"Connected. Picked backend: {backend.name}")
    print(f"Queue length: {backend.status().pending_jobs}")
