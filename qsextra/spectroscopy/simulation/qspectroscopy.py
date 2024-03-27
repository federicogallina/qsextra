from qsextra import ExcitonicSystem, ChromophoreSystem
from qsextra.qcomo import qevolve
from qsextra.spectroscopy.spectroscopy import Spectroscopy
from qsextra.tools import create_checkpoint_folder, destroy_checkpoint_folder
from qiskit import (QuantumCircuit,
                    QuantumRegister,
                    AncillaRegister,
                    ClassicalRegister,
                    )
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator
import numpy as np
from itertools import product
import os
import warnings
from qsextra.tools.oracle import ending_sentence

def __create_circuit(system: ChromophoreSystem | ExcitonicSystem,
                     bool_ancillae: bool,
                     ):
    register_list = []
    qr_kb = AncillaRegister(1, 'ketbra')
    register_list.append(qr_kb)
    # Exciton system qubits
    qr_e = QuantumRegister(system.system_size, 'sys_e')
    register_list.append(qr_e)
    # Pseudomode qubits
    if type(system) is ChromophoreSystem:
        qr_p_list = []
        for i in range(system.system_size):
            for k in range(len(system.mode_dict['omega_mode'])):
                # Checking the number of qubits of the pseudomode
                d_k = system.mode_dict['lvl_mode'][k]
                n_qubits = np.log2(d_k)
                tollerance = 1e-10
                if np.abs(np.round(n_qubits) - n_qubits) > tollerance:
                    # This instruction occurs when d is not a power of 2
                    n_qubits = np.ceil(n_qubits).astype(int)
                else:
                    n_qubits = np.round(n_qubits).astype(int)
                qr_p_list.append(QuantumRegister(n_qubits, f'mode({i},{k})'))
        register_list += qr_p_list
    # Ancilla qubit for collision model
    if bool_ancillae:
        qr_a = AncillaRegister(1, 'a')
        register_list.append(qr_a)
    cr = ClassicalRegister(1)
    register_list.append(cr)
    qc = QuantumCircuit(*register_list)
    return qc, qr_kb, qr_e

def __run(system: ChromophoreSystem | ExcitonicSystem,
          spectroscopy: Spectroscopy,
          shots: int,
          checkpoint: bool = False,
          restart_from_checkpoint: str | None = None,
          qevolve_kwds: dict = {},
          ) -> np.ndarray:
    
    def combinations_of_indices(sizes: list[int],
                                ):
        indices = [range(size) for size in sizes]
        return list(product(*indices))

    def qc_dipole_application(qc: QuantumCircuit,
                              qr_kb: QuantumRegister,
                              qr_e: QuantumRegister,
                              site: int,
                              side: str,
                              op: str,
                              ):
        if side == 'b':
            qc.x(qr_kb[0])
        if op == 'X':
            qc.cx(qr_kb[0], qr_e[site])
        elif op == 'Y':
            qc.cy(qr_kb[0], qr_e[site])
        if side == 'b':
            qc.x(qr_kb[0])
        return qc

    N = system.system_size

    # Defining the quantum circuit and the list of registers
    bool_ancillae = (qevolve_kwds.get('coll_rates') is not None)
    qc_og, qr_kb, qr_e = __create_circuit(system, bool_ancillae)
    qc_og.h(qr_kb[0])

    # Defining the expectation operators (on the ketbra ancilla)
    n_qubits = len(qc_og.qubits)
    e_op = [SparsePauliOp('I'*(n_qubits - 1) + 'X'), SparsePauliOp('I'*(n_qubits - 1) + 'Y')]

    # Defining the output variable (signal)
    size_signal = [len(T) for T in spectroscopy.delay_time]
    signal = np.zeros(size_signal, dtype = complex)
    # Defining a list with all the indices of signal
    time_indices_list = combinations_of_indices(size_signal)
    last_time_indices_number = -1

    # Doing a similar thing to define the site pathways
    site_pathways_indices_list = combinations_of_indices([N] * (len(spectroscopy.delay_time) + 1))

    # Checkpoint folder
    if checkpoint and restart_from_checkpoint is None:
        checkpoint_folder = create_checkpoint_folder()
        print(f'Checkpoint folder created ({checkpoint_folder})')
    elif restart_from_checkpoint is not None:
        if os.path.exists(restart_from_checkpoint):
            checkpoint_folder = restart_from_checkpoint
            last_time_indices_number = int(np.load(os.path.join(checkpoint_folder, 'last_time_indices_number.npy')))
            signal = np.load(os.path.join(checkpoint_folder, 'signal.npy'))

    # Defining the estimator
    estimator = Estimator(run_options = {'shots':shots})

    # Running over all the possibile combinations of times
    for time_indices_number, time_indices in enumerate(time_indices_list[last_time_indices_number + 1 :], last_time_indices_number + 1):
        qc_mega_list = []
        coeff_mega_list = []
        # Running over the site pathways
        for site_pathways_indices in site_pathways_indices_list:
            first_k = True
            first_b = True
            last_op = False
            qc_list = [qc_og.copy()]
            coeff_list = [1.]
            # Running over the light-matter interaction events + last dipole (signal emission)
            for n_interaction in range(len(site_pathways_indices)):
                if n_interaction == len(site_pathways_indices) - 1:    # Last interaction is not an interaction but signal emission
                    last_op = True
                    qc_evo = QuantumCircuit(n_qubits - 1)    # Empty circuit: we do not need to evolve anymore
                else:
                    qc_evo = qevolve(system,
                                     spectroscopy.delay_time[n_interaction][time_indices[n_interaction]],
                                     shots = None,
                                     initialize_circuit = False,
                                     verbose = False,
                                     **qevolve_kwds,
                                     )[0]
                qc_new_list = []
                coeff_new_list = []
                for n_qc, qc in enumerate(qc_list):
                    if first_b or first_k or last_op:
                        qc_new = qc_dipole_application(qc.copy(), qr_kb, qr_e, site_pathways_indices[n_interaction], spectroscopy.side_seq[n_interaction], 'X')
                        qc_new.compose(qc_evo, qubits = [qubit for qubit in range(1, n_qubits)], inplace = True)
                        qc_new_list.append(qc_new)
                        coeff_new_list.append(coeff_list[n_qc] * system.dipole_moments[site_pathways_indices[n_interaction]])
                    else:
                        for op in ['X', 'Y']:
                            qc_new = qc_dipole_application(qc.copy(), qr_kb, qr_e, site_pathways_indices[n_interaction], spectroscopy.side_seq[n_interaction], op)
                            qc_new.compose(qc_evo, qubits = [qubit for qubit in range(1, n_qubits)], inplace = True)
                            qc_new_list.append(qc_new)
                            if op == 'X':
                                coeff_new_list.append(coeff_list[n_qc] * system.dipole_moments[site_pathways_indices[n_interaction]] / 2)
                            elif (spectroscopy.side_seq[n_interaction] == 'k') ^ (spectroscopy.direction_seq[n_interaction] == 'i'):
                                coeff_new_list.append(coeff_list[n_qc] * system.dipole_moments[site_pathways_indices[n_interaction]] * (1.j / 2))   # That is I have applied mu^- operator
                            else:
                                coeff_new_list.append(coeff_list[n_qc] * system.dipole_moments[site_pathways_indices[n_interaction]] * (-1.j / 2))    # That is I applied mu^+ operator
                qc_list = qc_new_list
                coeff_list = coeff_new_list
                if spectroscopy.side_seq[n_interaction] == 'b':
                    first_b = False
                elif spectroscopy.side_seq[n_interaction] == 'k':
                    first_k = False
            qc_mega_list += qc_list
            coeff_mega_list += coeff_list
        # Executing the circuits and getting expectation values
        results_X = estimator.run(qc_mega_list, [e_op[0]]*len(qc_mega_list)).result().values
        results_Y = estimator.run(qc_mega_list, [e_op[1]]*len(qc_mega_list)).result().values
        results = results_X + 1.j * results_Y
        signal[*time_indices] += np.sum(results * np.array(coeff_mega_list))    # Sum of the expectation values of the dipole operators
        # Savaing checkpoint if requested
        if checkpoint:
            np.save(os.path.join(checkpoint_folder, 'last_time_indices_number.npy'), time_indices_number)
            np.save(os.path.join(checkpoint_folder, 'signal.npy'), signal)
    # Deleting checkpoint folder
    if checkpoint or restart_from_checkpoint is not None:
        destroy_checkpoint_folder(checkpoint_folder)
    return signal

def qspectroscopy(system: ChromophoreSystem | ExcitonicSystem,
                  spectroscopy: Spectroscopy,
                  shots: int = 1000,
                  verbose: bool = True,
                  checkpoint: bool = False,
                  restart_from_checkpoint: str | None = None,
                  **qevolve_kwds,
                  ) -> np.ndarray:
    '''
    Parameters
    ----------
    system: ChromophoreSystem | ExcitonicSystem
        The system to simulate.

    spectroscopy: Spectroscopy
        The `Spectroscopy` object to simulate.

    shots: int
        Number of measurements per circuit.

    verbose: bool
        If `True`, print during the execution of the code.

    checkpoint: bool
        If `True`, save checkpoint during execution.

    restart_from_checkpoint: str | None
        The name of the checkpoint folder to continue execution of.

    qevolve_kwds: dict
        Keywords for the `qevolve` method.

    Returns
    -------
    numpy.ndarray
        An array with the results of the simulation. The dimensions are that of `spectroscopy.delay_time.shape`.
    '''
    # Checking the presence of pseudomodes
    if type(system) is ChromophoreSystem and system.mode_dict is None:
        warnings.warn("Pseudomodes not specified. Executing the dynamics with system as an ExcitonicSystem.")
        return qspectroscopy(system.extract_ExcitonicSystem(),
                             spectroscopy,
                             shots,
                             qevolve_kwds,
                             verbose,
                             )
    
    # Removing special keywords from qevolve_kwds
    qevolve_kwds.pop('system', None)
    qevolve_kwds.pop('time', None)
    qevolve_kwds.pop('shots', None)
    qevolve_kwds.pop('initialize_circuit', None)
    qevolve_kwds.pop('verbose', None)

    # Checking Spectroscopy
    if spectroscopy.side_seq == '' or spectroscopy.direction_seq == '':
        raise ValueError('Spectroscopy object is not correctly defined. Spectroscopy.side_seq or Spectroscopy.direction_seq are missing.')

    signal = __run(system, spectroscopy, shots, checkpoint, restart_from_checkpoint, qevolve_kwds)  
    ending_sentence(verbose)
    return signal