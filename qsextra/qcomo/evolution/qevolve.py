from qsextra import ExcitonicSystem, ChromophoreSystem
from qsextra.tools import if_scalar_to_list, gray_code_list
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.result.result import Result
from qiskit_aer import AerSimulator, AerError
import numpy as np
import itertools
from math import prod
import warnings
from qsextra.tools.oracle import ending_sentence

def __create_circuit(system: ChromophoreSystem | ExcitonicSystem,
                     bool_ancillae: bool,
                     ):
    qr_e = QuantumRegister(system.system_size, 'sys_e')
    register_list = [qr_e]
    if type(system) is ChromophoreSystem:
        qr_p_list = [QuantumRegister(len(system.mode_dict['lvl_mode']), 'mode({},{})'.format(i,k)) for i in range(system.system_size) for k in range(len(system.mode_dict['omega_mode']))]
        register_list += qr_p_list
    if bool_ancillae:
        qr_a = AncillaRegister(1, 'a')
        register_list.append(qr_a)
    qc = QuantumCircuit(*register_list)
    return qc, register_list

def __circuit_init(qc: QuantumCircuit,
                   register_list: list[QuantumRegister],
                   system: ChromophoreSystem | ExcitonicSystem,
                   ) -> QuantumCircuit:
    qr_e = register_list[0]
    if system.state_type == 'state':
        qc.initialize(system.state, qr_e)
    elif system.state_type == 'delocalized excitation':
        relevant_qubits = np.where(np.array(system.state) != 0)[0].tolist()
        relevant_state = np.zeros(2**len(relevant_qubits), dtype = complex)
        for ni, i in enumerate(relevant_qubits):
            relevant_state[2**ni] = system.state[i]
        qc_relevant = QuantumCircuit(len(relevant_qubits))
        qc_relevant.initialize(relevant_state)
        qc.compose(qc_relevant,
                   qubits = [qr_e[i] for i in relevant_qubits],
                   inplace = True,
                   )
    elif system.state_type == 'localized excitation':
        qc.x(qr_e[system.state])
    return qc

def __Trotter_class(Trotter_order: int):
    if Trotter_order == 1:
        return LieTrotter()
    else:
        warnings.warn('At the moment, the evolution is implemented with a division between the Hamiltonian evolutions of the system, pseudomodes (if present), system-pseudomode interactions, and collision interactions. This division corresponds to a partial first-order Trotterization. Subsequent orders of Suzuki-Trotter apply only to the evolutions of the mentioned Hamiltonians.')
        return SuzukiTrotter(order=Trotter_order)

def __sys_evolution(system: ChromophoreSystem | ExcitonicSystem,
                    dt: float,
                    Trotter_order: int,
                    ) -> QuantumCircuit:
    energies = system.e_el
    couplings = system.coupl_el
    N = system.system_size
    qc = QuantumCircuit(N)

    # Create the dictionary of Pauli operations
    op_dict = {}
    for i in range(N):
        if energies[i] != 0.:
            op_str = ''.join([*['I']*(N-i-1), 'Z', *['I']*i])
            op_dict[op_str] = - energies[i] / 2
    for i in range(N):
        for j in range(i+1, N):
            if couplings[i,j] != 0.:
                for pauli in ['X', 'Y']:
                    op_str = ''.join([*['I']*(N-j-1), pauli, *['I']*(j-i-1), pauli, *['I']*i])
                    op_dict[op_str] = couplings[i,j] / 2
    
    # Selecting the Trotter class
    Trotter = __Trotter_class(Trotter_order)

    # Creating the gates
    pauli_gates = SparsePauliOp(list(op_dict.keys()), coeffs=list(op_dict.values()))
    evolution_gate = PauliEvolutionGate(pauli_gates, time=dt, synthesis=Trotter)

    # Appending to qc and returning qc
    qc.append(evolution_gate, range(N))
    return qc.decompose(reps=2)

def __mode_operators(system: ChromophoreSystem | ExcitonicSystem,
                     k: int,
                     ):
    ''' Generates dictionaries for mode operators a, a^dagger, a+a^dagger, and a^dagger*a.
    Gray code is used.
    Dictionaries are created such that:
    - keys represent the Pauli strings into which the operator is decomposed;
    - values represent the coefficients associated with the keys.
    '''
    def add_str_and_coefs_to_dict(target_dict: dict,
                                  Pauli_ops: list[list[str]],
                                  Pauli_coefs: list[list[complex]],
                                  mul: float,
                                  ):
        ''' A method for __mode_operators.
        Take: A target dictionary, Pauli_ops, Pauli_coefs and a multiplicative factor.
        Compose: The Pauli strings by combining Pauli_ops and the coefficients by combining Pauli_coefs and multiplying by mul.
        Add: The strings and coeffcients to the target dictionary.
        Return: The updated target dictionary.
        '''
        Pauli_strings = list(itertools.product(*Pauli_ops))
        Pauli_strings = [''.join(Pauli_strings[i]) for i in range(len(Pauli_strings))]
        Pauli_coefs_comb = list(itertools.product(*Pauli_coefs))
        Pauli_coefs_comb = [mul*prod(Pauli_coefs_comb[i]) for i in range(len(Pauli_coefs_comb))]
        for i, Pauli_string in enumerate(Pauli_strings):
            if Pauli_string in target_dict:
                target_dict[Pauli_string] += Pauli_coefs_comb[i]
            else:
                target_dict[Pauli_string] = Pauli_coefs_comb[i]
        return target_dict

    # Checking the number of qubits of the pseudomode
    d_k = system.mode_dict['lvl_mode'][k]
    n_qubits = np.log2(d_k)
    tollerance = 1e-10
    if np.abs(np.round(n_qubits) - n_qubits) > tollerance:
        # This instruction occurs when d is not a power of 2
        n_qubits = np.ceil(n_qubits).astype(int)
    else:
        n_qubits = np.round(n_qubits).astype(int)

    # Constructing a list of Gray codes for the pseudomode
    pseudo_encoded = gray_code_list(n_qubits)
    
    # a
    a_dict = {}
    for index in range(1, d_k):
        Pauli_ops = []
        Pauli_coefs = []
        for i in range(n_qubits):
            if pseudo_encoded[index -1][i] == pseudo_encoded[index][i]:
                Pauli_ops.append(['I','Z'])
                Pauli_coefs.append([1./2,1./2]) if pseudo_encoded[index - 1][i] == '0' else Pauli_coefs.append([1./2,-1./2])
            else:
                Pauli_ops.append(['X','Y'])
                Pauli_coefs.append([1/2,1.j/2]) if pseudo_encoded[index - 1][i] == '0' else Pauli_coefs.append([1./2,-1.j/2])
        a_dict = add_str_and_coefs_to_dict(a_dict, Pauli_ops, Pauli_coefs, np.sqrt(index))
    
    # a^dagger
    a_dagger_dict = {}
    for index in range(d_k - 1):
        Pauli_ops = []
        Pauli_coefs = []
        for i in range(n_qubits):
            if pseudo_encoded[index + 1][i] == pseudo_encoded[index][i]:
                Pauli_ops.append(['I','Z'])
                Pauli_coefs.append([1./2,1./2]) if pseudo_encoded[index + 1][i] == '0' else Pauli_coefs.append([1./2,-1./2])
            else:
                Pauli_ops.append(['X','Y'])
                Pauli_coefs.append([1/2,1.j/2]) if pseudo_encoded[index + 1][i] == '0' else Pauli_coefs.append([1./2,-1.j/2])
        a_dagger_dict = add_str_and_coefs_to_dict(a_dagger_dict, Pauli_ops, Pauli_coefs, np.sqrt(index+1))

    # a + a^dagger
    a_plus_a_dagger_dict = {}
    for key in a_dict:
        if a_dict[key] != -a_dagger_dict[key]:
            a_plus_a_dagger_dict[key] = a_dict[key] + a_dagger_dict[key]
    
    # a^dagger * a
    a_dagger_a_dict = {}
    for index in range(d_k):
        Pauli_ops = []
        Pauli_coefs = []
        for i in range(n_qubits):
            Pauli_ops.append(['I','Z'])
            Pauli_coefs.append([1./2,1./2]) if pseudo_encoded[index][i] == '0' else Pauli_coefs.append([1./2,-1./2])
        a_dagger_a_dict = add_str_and_coefs_to_dict(a_dagger_a_dict, Pauli_ops, Pauli_coefs, index)

    return a_dict, a_dagger_dict, a_plus_a_dagger_dict, a_dagger_a_dict

def __mode_evolution(frequency_mode: float,
                     a_dagger_a_dict: dict,
                     dt: float,
                     Trotter_order: int,
                     ) -> QuantumCircuit:
    # Setting the number of qubits by measuring the length of the first Pauli string
    n_qubits = len(list(a_dagger_a_dict.keys())[0])

    # Initializing the circuit
    qr_p_k = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr_p_k)

    # Checking the frequency of the mode
    if frequency_mode != 0:
        # Selecting the Trotter class
        Trotter = __Trotter_class(Trotter_order)

        # Creating the gates
        pauli_gates = SparsePauliOp(list(a_dagger_a_dict.keys()), coeffs = frequency_mode * np.array(list(a_dagger_a_dict.values())))
        evolution_gate = PauliEvolutionGate(pauli_gates, time = dt, synthesis = Trotter)
        qc.append(evolution_gate, qr_p_k)
    return qc.decompose(reps=2)

def __sys_mode_interaction(interaction_strength: ChromophoreSystem,
                           a_plus_a_dagger_dict: dict,
                           dt: float,
                           Trotter_order: int,
                           ) -> QuantumCircuit:
    qr_e = QuantumRegister(1)

    # Setting the number of qubits by measuring the length of the first Pauli string
    n_qubits = len(list(a_plus_a_dagger_dict.keys())[0])

    # Initializing the circuit
    qr_p_k = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr_e, qr_p_k)

    # Useful gates
    I = SparsePauliOp('I')
    Z = SparsePauliOp('Z')

    # Checking the interaction strength
    if interaction_strength != 0:
        # Selecting the Trotter class
        Trotter = __Trotter_class(Trotter_order)

        # Creating the gates
        a_plus_a_dagger_gates = SparsePauliOp(list(a_plus_a_dagger_dict.keys()), coeffs = np.array(list(a_plus_a_dagger_dict.values())))
        # Here we take into account that operator |e><e| = (1 - Z) / 2
        H_coll_gate = a_plus_a_dagger_gates ^ ((0.5 * I) - (0.5 * Z))
        H_coll_gate = H_coll_gate.simplify()
        H_coll_gate = interaction_strength * H_coll_gate
        evolution_gate = PauliEvolutionGate(H_coll_gate, time = dt, synthesis = Trotter)
        qubit_list = [qr_e[0]]
        qubit_list += [qr_p_k[q] for q in range(n_qubits)]
        qc.append(evolution_gate, qubit_list)
    return qc.decompose(reps=2)

def __Trotter_step(system: ChromophoreSystem | ExcitonicSystem,
                   dt: float,
                   Trotter_order: int,
                   bool_anciallae: bool,
                   ) -> QuantumCircuit:
    ''' Return the circuit implementing a single Trotter step.
    '''
    qc, register_list = __create_circuit(system, bool_anciallae)
    qr_e = register_list[0]
    qc.compose(__sys_evolution(system, dt, Trotter_order),
               qubits = qr_e,
               inplace = True,
               )
    if type(system) is ChromophoreSystem:
        N = system.system_size
        W = len(system.mode_dict['omega_mode'])
        qr_p_list = register_list[1:-1] if bool_anciallae else register_list[1:]
        for k in range(W):
            _, _, a_plus_a_dagger_dict, a_dagger_a_dict = __mode_operators(system, k)   # The pseudomodes with different i but same k has the same operators
            n_qubits_mode_k = len(list(a_dagger_a_dict.keys())[0])                      # Setting the number of qubits by measuring the length of the first Pauli string

            qc_mode_evo_k = __mode_evolution(system.mode_dict['omega_mode'][k], a_dagger_a_dict, dt, Trotter_order)
            qc_sysmode_int_k = __sys_mode_interaction(system.mode_dict['coupl_ep'][k], a_plus_a_dagger_dict, dt, Trotter_order)
            for i in range(N):
                qr_p_ik = qr_p_list[i*W + k]
                qc.compose(qc_mode_evo_k,
                           qubits = qr_p_ik,
                           inplace = True,
                           )
                qubit_list = [qr_e[i]]
                qubit_list += [qr_p_ik[q] for q in range(n_qubits_mode_k)]
                qc.compose(qc_sysmode_int_k,
                           qubits = qubit_list,
                           inplace = True,
                           )
    return qc

def __collisions_exsys(system: ExcitonicSystem,
                       dt: float,
                       coll_rates: float,
                       ) -> QuantumCircuit:
    ''' Generate collisions between system qubits and the ancilla qubit.
    In the future, we may add the possibility of using not just one ancilla, but N ancillae for parallelizing the collisions.
    '''
    qc, register_list = __create_circuit(system, coll_rates is not None)
    qr_e = register_list[0]
    qr_a = register_list[-1]
    for i in range(system.system_size):
        qc.rzx(2 * np.sqrt(coll_rates) * np.sqrt(dt), qr_e[i], qr_a[0])
        qc.reset(qr_a[0])
    return qc

def __collisions_cpsys(system: ChromophoreSystem,
                       dt: float,
                       Trotter_order: int,
                       coll_rates: list,
                       ) -> QuantumCircuit:
    ''' Generate collisions between the pseudomodes and the ancilla qubit.
    In the future, we may add the possibility of using not just one ancilla, but more ancillae for parallelizing the collisions.
    '''
    N = system.system_size
    W = len(system.mode_dict['omega_mode'])
    qc, register_list = __create_circuit(system, coll_rates is not None)
    qr_p_list = register_list[1:-1]
    qr_a = register_list[-1]

    # Selecting the Trotter class
    Trotter = __Trotter_class(Trotter_order)

    # Useful gates
    X = SparsePauliOp('X')
    Y = SparsePauliOp('Y')

    for k in range(W):
        if coll_rates[k] != 0:
            a_dict, a_dagger_dict, _, _ = __mode_operators(system, k)                   # The pseudomodes with different i but same k has the same operators
            n_qubits_mode_k = len(list(a_dict.keys())[0])                               # Setting the number of qubits by measuring the length of the first Pauli string

            # Creating the gates
            a_pauli_gates = SparsePauliOp(list(a_dict.keys()), coeffs = np.array(list(a_dict.values())))
            a_dagger_pauli_gates = SparsePauliOp(list(a_dagger_dict.keys()), coeffs = np.array(list(a_dagger_dict.values())))
            # Here we take into account that ancilla operator \simga^+ = (X - iY) / 2 and \sigma^- = (X + iY) / 2
            H_coll_gate = (a_pauli_gates ^ ((0.5 * X) - (0.5j * Y))) + (a_dagger_pauli_gates ^ ((0.5 * X) + (0.5j * Y)))
            H_coll_gate = H_coll_gate.simplify()                                        # Sum repeated Pauli strings and remove zeros
            H_coll_gate = np.sqrt(coll_rates[k] / dt) * H_coll_gate
            evolution_gate = PauliEvolutionGate(H_coll_gate, time = dt, synthesis = Trotter)

            for i in range(N):
                qr_p_ik = qr_p_list[i*W + k]
                qubit_list = [qr_a[0]]
                qubit_list += [qr_p_ik[q] for q in range(n_qubits_mode_k)]
                qc.append(evolution_gate, qubit_list)
                qc.reset(qr_a[0])
    return qc.decompose(reps=2)

def __collision_circuit(system: ChromophoreSystem | ExcitonicSystem,
                        dt: float,
                        Trotter_order: int,
                        coll_rates: float | list[float],
                        ) -> QuantumCircuit:
    ''' Return the circuit implementing the scheme of the collision model.
    '''
    if type(system) is ExcitonicSystem:
        return __collisions_exsys(system, dt, coll_rates)
    elif type(system) is ChromophoreSystem:
        return __collisions_cpsys(system, dt, Trotter_order, coll_rates)   

def __propagation(qc: QuantumCircuit,
                  register_list: list[QuantumRegister],
                  system: ChromophoreSystem | ExcitonicSystem,
                  time: float | list[float] | np.ndarray,
                  Trotter_number: int,
                  Trotter_order: int,
                  dt: float,
                  coll_rates: float | list[float] | None,
                  shots: int,
                  ):
    # Setting the time step
    if dt is None:
        if np.isscalar(time):
            dt_Trotter = time / Trotter_number
        else:
            dt_Trotter = (time[1] - time[0]) / Trotter_number # Tacitly assuming that time is equally spaced.
        dt = dt_Trotter
        Trotter_number = 1
    else:
        dt_Trotter = dt / Trotter_number
    
    # Making sure time is itarable
    time = if_scalar_to_list(time)
    # Checking if every time entry is a multiple of dt_collisions
    tolerance = 1e-10
    if all(abs(t - np.round(t / dt) * dt) < tolerance for t in time):
        # The condition above should be equal to all(t % dt_collisions < tolerance for t in time), but for example 0.5 % 0.1 = 0.1 for python because he say floor(0.5 / 0.1 = 0.4).
        number_of_dts = np.round(np.array(time) / dt).astype(int)
    else:
        raise ValueError('Entries in time are not multiples of dt_collisions.')

    # Constructing the container for the resulting circuits
    qc_list = []

    # Composing the circuit for a Trotter step
    qc_ts = __Trotter_step(system,
                           dt_Trotter,
                           Trotter_order,
                           coll_rates is not None,
                           )
    
    # Composing the circuit for the collisions
    if coll_rates is not None:
        qc_collision = __collision_circuit(system,
                                           dt,
                                           Trotter_order,
                                           coll_rates,
                                           )

    # Adding a classical register when needed
    if shots != 0 and shots is not None:
        creg = ClassicalRegister(system.system_size)
        qc.add_register(creg)

    # Propagation
    for ndt in range(np.max(number_of_dts) + 1):
        if ndt != 0:
            for _ in range(Trotter_number):
                qc.compose(qc_ts, inplace=True)
            if coll_rates is not None:
                qc.compose(qc_collision, inplace=True)
        if ndt in number_of_dts:
            qc_copy = qc.copy()
            if shots != 0 and shots is not None:
                qc_copy.measure(register_list[0], creg)
            qc_list.append(qc_copy)
    return qc_list

def qevolve(system: ChromophoreSystem | ExcitonicSystem,
            time: float | list[float] | np.ndarray,
            shots: int | None = None,
            initialize_circuit: bool = True,
            Trotter_number: int = 1,
            Trotter_order: int = 1,
            dt: float = None,
            coll_rates: float | list[float] | None = None,
            GPU: bool = False,
            verbose: bool = True,
            ) -> QuantumCircuit | list[QuantumCircuit] | Result:
    ''' Quantum algorithm for the simulation of the dynamics of the system.

    Parameters
    ----------
    system: ChromophoreSystem | ExcitonicSystem
        The system to evolve.

    time: float | list[float] | numpy.ndarray
        The target time(s).

    shots: int | None
        If `None` or `0`, does not add measurements to the quantum circuits. If positive, add measurement operations and perform that number of measurements on the circuits.

    initialize_circuit: bool
        If `True`, initialize the circuit to the system state.

    Trotter_number: int
        Number of Trotter steps in a time interval (`dt`) for the system evolution.

    Trotter_order: int
        Order of the Trotter decomposition. If `1`, use `LieTrotter` decomposition from `qiskit.synthesis`. If more than `1`, use `SuzukiTrotter` decomposition from `qiskit.synthesis`.

    dt: float
        Time interval. If `coll_rates` is specified, the collisions are applied at each dt. If not specified, `dt = (time[1] - time[0]) / Trotter_number`.

    coll_rates: float | list[float] | None
        If `None`, a Schrödinger dynamics is returned. Else, if system is an `ExcitonicSystem` object, a float value dictating the system relaxations due to a Markovian environment. If system is a `ChromophoreSystem` object, a float or list of floats with the relaxation rates of pseudomodes due to the interaction with a Markovian environment.

    GPU: bool
        If `True`, use GPU for the simulation.

    verbose: bool
        If `True`, print during the execution of the code.

    Returns
    -------
    qiskit.QuantumCircuit | list[qiskit.QuantumCircuit] | qiskit.result.result.Result
        If shots is `None`, return the `QuantumCircuit` or the `list[QuantumCircuits]` implementing the dynamics. If `shots` is `0`, return the qiskit's `Result` object resulting from `AerSimulation(method='statevector')`. If shots is a positive number, return the qiskit's `Result` object resulting from `AerSimulation()`.
    '''
    # Define a function to print if verbose
    verboseprint = print if verbose else lambda *a, **k: None

    # Checking the validity of the system
    if type(system) is not ChromophoreSystem and type(system) is not ExcitonicSystem:
        raise TypeError('system must be an instance of class ChromophoreSystem or ExcitonicSystem.')
    if not system._validity:
        raise Exception('Make sure all the system parameters are specified and consistent.')
    
    # Checking the presence of pseudomodes
    if type(system) is ChromophoreSystem and system.mode_dict is None:
        warnings.warn("Pseudomodes not specified. Executing the dynamics with system as an ExcitonicSystem.")
        return qevolve(system.extract_ExcitonicSystem(),
                       time,
                       shots,
                       initialize_circuit,
                       Trotter_number,
                       Trotter_order,
                       dt,
                       coll_rates,
                       GPU,
                       verbose,
                       )

    # Checking the collision
    if type(system) is ExcitonicSystem:
        if not np.isscalar(coll_rates) and coll_rates is not None:
            raise TypeError('For an ExcitonicSystem, coll_rates must be a scalar.')
    elif type(system) is ChromophoreSystem:
        if coll_rates is not None:
            coll_rates = if_scalar_to_list(coll_rates)
            if len(coll_rates) != len(system.mode_dict['omega_mode']):
                raise ValueError('The number of dephasing rates is different from the number of pseudomodes per chromophore.')

    # Constructing the circuit
    qc, register_list = __create_circuit(system, coll_rates is not None)

    # Circuit initialization
    if initialize_circuit:
        qc = __circuit_init(qc,
                            register_list,
                            system,
                            )

    verboseprint('Start creating the circuits...')

    # Propagation
    qc_list = __propagation(qc,
                            register_list,
                            system,
                            time,
                            Trotter_number,
                            Trotter_order,
                            dt,
                            coll_rates,
                            shots,
                            )

    verboseprint('Circuits created...')

    # Return the circuit
    if shots is None:
         ending_sentence(verbose)
         return qc_list
    # Run the statevector simulation
    elif shots == 0:
        simulator = AerSimulator(method = 'statevector')
        if GPU:
            try:
                simulator.set_option(device = 'GPU')
            except AerError as e:
                print(e)
        results = simulator.run(qc_list).result()
    # Run the aer simulation
    else:
        verboseprint('Start measuring the circuits...')
        simulator = AerSimulator()
        if GPU:
            try:
                simulator.set_option(device = 'GPU')
            except AerError as e:
                print(e)
        results = simulator.run(qc_list, shots = shots).result()
    ending_sentence(verbose)
    return results
