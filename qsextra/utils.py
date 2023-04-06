from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from typing import Literal

def cnot_staircase_circuit(qubits, dt, Pauli_dict):
    '''A simple routine to generate a CNOT staircase circuit implementing some Pauli rotations.
    Input:
    - qubits [int]: number of qubits of the system
    - dt [float, double]: time step of evolution
    - Pauli_dict [dictionary]: dictionary of Pauli strings and relative coefficients'''
    
    q_reg = QuantumRegister(qubits)
    qc = QuantumCircuit(q_reg)
    for Pauli_string, Pauli_coef in Pauli_dict.items():
        last_qubit = None
        qc_rotations = QuantumCircuit(q_reg)
        for npauli, pauli in enumerate(Pauli_string):
            if pauli == 'X':
                qc_rotations.h(npauli)
            elif pauli == 'Y':
                qc_rotations.rx(np.pi/2, npauli)
        qc_stairs = QuantumCircuit(q_reg)
        for npauli, pauli in enumerate(Pauli_string):
            if pauli != 'I':
                last_qubit = npauli
                finder = False
                index = npauli + 1
                while finder == False and index < qubits:
                    if Pauli_string[index] == 'I':
                        index = index + 1
                    else:
                        finder = True
                        qc_stairs.cx(npauli, index)
        if last_qubit != None:
            qc.compose(qc_rotations,inplace=True)
            qc.compose(qc_stairs,inplace=True)
            qc.rz(Pauli_coef * dt, last_qubit)
            qc.compose(qc_stairs.inverse(),inplace=True)
            qc.compose(qc_rotations.inverse(),inplace=True)
    return qc

class Options():
    def __init__(self,
                 sampling_steps: int = 1,
                 qubits_per_pseudomode: int = 1,
                 job_chunks: int = 200):
        '''
        Extra options for the computation of the dynamics.

        Input:
        - sampling_steps
            Number of dt evolution steps before sampling the circuit
        - qubits_per_pseudomode: int
            Number of qubits to use in the collision model to implement a pseudomode.
        - job_chunks: int
            Number of chunks into which the computation of the dynamics is divided.
        '''
        self.options_dict = {}
        self.options_dict['sampling_steps'] = sampling_steps
        self.options_dict['qubits_per_pseudomode'] = qubits_per_pseudomode
        self.options_dict['job_chunks'] = job_chunks

    def set(self,
            property: Literal['qubits_per_pseudomode', 'job_chunks', 'sampling_steps'],
            value: int
            ) -> None:
        self.options_dict[property] = value

    def get(self) -> dict:
        return self.options_dict
