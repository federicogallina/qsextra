from qsextra import ExcitonicSystem, ChromophoreSystem
from qsextra.qcomo import qevolve
from qsextra.spectroscopy.spectroscopy import Spectroscopy, FeynmanDiagram
from qsextra.tools import if_scalar_to_list
from qsextra.tools.oracle import oracle_word
from qiskit import (QuantumCircuit,
                    QuantumRegister,
                    AncillaRegister,
                    ClassicalRegister,
                    )
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.result.result import Result
from qiskit_aer import AerSimulator, AerError
import numpy as np
import warnings
from qsextra.tools.oracle import ending_sentence

def qspectroscopy(system: ChromophoreSystem | ExcitonicSystem,
                  spectroscopy: Spectroscopy,
                  shots: int | None = None,
                  evolution_method: str = 'qcomo',
                  qevolve_dict: dict = {},
                  verbose: bool = True,
                  ) -> np.ndarray:
    '''
    Parameters
    ----------
    system: ChromophoreSystem | ExcitonicSystem
        The system to simulate.

    spectroscopy: Spectroscopy
        The `Spectroscopy` object to simulate.

    shots: int | None
        If `None` or `0`, does not add measurements to the quantum circuits. If positive, add measurement operations and perform that number of measurements on the circuits.

    evolution_method: str
        The method used to perform the evolution. Accepted:
        - `'qcomo'`: Quantum collision model.

    qevolve_kwds: dict
        A dictionary with keywords for the `qevolve` method.

    verbose: bool
        If `True`, print during the execution of the code.

    Returns
    -------

    '''
    signal = np.zeros(spectroscopy.delay_time.shape, dtype = complex)
    
    ending_sentence(verbose)
    return signal