from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, assemble
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliSumOp, X, Y, Z, PauliTrotterEvolution
from qiskit.circuit import Parameter
import numpy as np
import itertools
from math import prod
from qsextra.utils import CNOT_staircase_circuit


def evolve(Hamiltonian, t_max, dt, Gamma, tau, shots=10000, method='classical noise', mapping='physical', qubits_per_pseudomode=1):
    '''Quantum algorithm to compute the dynamics of the open system.

    Input parametes:
    - Hamiltonian [list, np.array]: the Hamiltonian of the open system in an algorithmic mapping
    - t_max [float, double]: maximum time of the dynamics (included)    
    - dt [float, double]: time step of the dynamics
    - Gamma [float, double]: equivalent noise strength
    - tau [float, double]: memory time of the fluctuations
    - shots [int]: number of shots
    - method ['classical noise', 'collision model']: quantum algorithm used to compute the dynamics
    - mapping ['algorithmic', 'physical']: mapping of the system on the qubits
    - qubits_per_pseudomode [int]: number of qubits to use in the collision model to implement a pseudomode

    Return
    ---- if method == 'classical noise' ----
    List[float]
        The populations of the target site during time.
    int
        The number of trajectories executed.
    ---- if method == 'collision model' ----
    List[float]
        The populations of the target site during time.
    float
        Final time.
    '''
    
    if method not in ['classical noise', 'collision model']:
        raise Exception('Method must be "classical noise" or "collision model"')

    if mapping not in ['algorithmic', 'physical']:
        raise Exception('Method must be "algorithmic" or "physical"')

    H = np.array(Hamiltonian)
    if (H != H.conj().T).all() or H.shape[0] != H.shape[1]:
        raise Exception('Hamiltonian must be an Hermitian square matrix')
    N = H.shape[0]

    if method == 'classical noise':
        tlist = np.arange(0,t_max+dt,dt)
        populations = np.zeros([len(tlist),N])
        trajectories = 0
        for pops_iter, trajs_iter in _CNA(mapping).solve(H, N, dt, t_max, tau, Gamma, shots):
            populations = pops_iter
            trajectories = trajs_iter
        return populations, trajectories
    if method == 'collision model':
        populations = []
        final_time = 0
        for pops_iter, final_time_iter in _CA(mapping).solve(H, N, dt, t_max, tau, Gamma, shots, qubits_per_pseudomode):
            populations = pops_iter
            final_time = final_time_iter
        return populations, final_time

def minimal_circuit(Hamiltonian, method='classical noise', mapping='physical', qubits_per_pseudomode=1, Gamma = 1, tau = 1, dt = 1, backend=None, transpiled = True):
    '''Returns the quantum circuit for a time step evolution.
    Input parametes:
    - Hamiltonian [list, np.array]: the Hamiltonian of the open system in an algorithmic mapping
    - method ['classical noise', 'collision model']: quantum algorithm used to compute the dynamics
    - mapping ['algorithmic', 'physical']: mapping of the system on the qubits
    - qubits_per_pseudomode [int]: number of qubits to use in the collision model to implement a pseudomode
    - Gamma [float, double]: equivalent noise strength
    - tau [float, double]: memory time of the fluctuations
    - dt [float, double]: time step of the dynamics
    - backend [IBMQ_backend]: backend used for transpilation
    - transpile [boolean]: if True returne the transpiled circuit
    '''
    
    if method not in ['classical noise', 'collision model']:
        raise Exception('Method must be "classical noise" or "collision model"')

    if mapping not in ['algorithmic', 'physical']:
        raise Exception('Method must be "algorithmic" or "physical"')

    H = np.array(Hamiltonian)
    if (H != H.conj().T).all() or H.shape[0] != H.shape[1]:
        raise Exception('Hamiltonian must be an Hermitian square matrix')
    N = H.shape[0]

    if method == 'classical noise':
        return _CNA(mapping).minimal_circuit(H, N, dt, backend, transpiled)
    if method == 'collision model':
        return _CA(mapping).minimal_circuit(H, N, dt, tau, Gamma, qubits_per_pseudomode, backend, transpiled)

class _CNA():
    def __init__(self, mapping):
        '''Working class for Classical Noise Algorithm, algorithmic mapping'''
        self.mapping = mapping

    def __sys_free_evolution(self, H, N, dt):
        energy_params = [Parameter('H{}{}'.format(i,i)) for i in range(N)]

        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
            qc = QuantumCircuit(q_reg)
            H = np.pad(H, ((0,2**qubits-N), (0,2**qubits-N)), mode='constant', constant_values=0)
            sys_encoded = []
            site_energies_dict = {}
            for index in range(N):
                sys_encoded.append('{:b}'.format(index).zfill(qubits)) #Creating a list with the bit strings representing the sites of the system
                #Creating the Pauli strings corresponding to the projectors
                Pauli_ops = []
                Pauli_coefs = []
                for i in range(qubits):
                    Pauli_ops.append(['I','Z'])
                    Pauli_coefs.append([1./2,1./2]) if sys_encoded[index][i] == '0' else Pauli_coefs.append([1./2,-1./2])
                Pauli_strings = list(itertools.product(*Pauli_ops))
                Pauli_strings = [''.join(Pauli_strings[i]) for i in range(len(Pauli_strings))]
                Pauli_coefs_comb = list(itertools.product(*Pauli_coefs))
                Pauli_coefs_comb = [energy_params[index]*prod(Pauli_coefs_comb[i]) for i in range(len(Pauli_coefs_comb))]
                for i, Pauli_string in enumerate(Pauli_strings):
                    try:
                        site_energies_dict[Pauli_string] = site_energies_dict[Pauli_string] + Pauli_coefs_comb[i]
                    except:
                        site_energies_dict[Pauli_string] = Pauli_coefs_comb[i]
            qc.compose(CNOT_staircase_circuit(qubits, dt, site_energies_dict),inplace=True)

            interactions_dict = {}
            for index in range(N-1):
                for index2 in range(index+1, N):
                    #Creating the Pauli strings corresponding to the interaction operators
                    Pauli_ops = []
                    Pauli_coefs = []
                    for i in range(qubits):
                        if sys_encoded[index][i] == sys_encoded[index2][i]:
                            Pauli_ops.append(['I','Z'])
                            Pauli_coefs.append([1./2,1./2]) if sys_encoded[index][i] == '0' else Pauli_coefs.append([1./2,-1./2])
                        else:
                            Pauli_ops.append(['X'])
                            Pauli_coefs.append([1])
                    Pauli_strings = list(itertools.product(*Pauli_ops))
                    Pauli_strings = [''.join(Pauli_strings[i]) for i in range(len(Pauli_strings))]
                    Pauli_coefs_comb = list(itertools.product(*Pauli_coefs))
                    Pauli_coefs_comb = [H[index,index2]*prod(Pauli_coefs_comb[i]) for i in range(len(Pauli_coefs_comb))]
                    for i, Pauli_string in enumerate(Pauli_strings):
                        try:
                            interactions_dict[Pauli_string] = interactions_dict[Pauli_string] + Pauli_coefs_comb[i]
                        except:
                            interactions_dict[Pauli_string] = Pauli_coefs_comb[i]
            interactions_op = PauliTrotterEvolution().convert((PauliSumOp(SparsePauliOp(list(interactions_dict.keys()),coeffs=list(interactions_dict.values()))) * dt).exp_i())
            qc.compose(interactions_op.to_circuit(),range(qubits),inplace=True)

        else:
            q_reg = QuantumRegister(N)
            qc = QuantumCircuit(q_reg)
            for i in range(N):
                qc.rz(-energy_params[i]*dt,q_reg[i])
            for i in range(N-1):
                for j in range(i+1,N):
                    if H[i,j] != 0:
                        qc.rxx(H[i,j]*dt,q_reg[i],q_reg[j])
                        qc.ryy(H[i,j]*dt,q_reg[i],q_reg[j])
        
        return qc, energy_params
    
    def __fluctuation_update(self, H_fluc, rand_increment, tau, dt):
        H_fluc = H_fluc*np.exp(-dt/tau)+np.array(rand_increment)*np.sqrt(1-np.exp(-2*dt/tau))
        return H_fluc

    def solve(self, H, N, dt, t_max, tau, Gamma, shots):
        #Initializing the registers
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
            cl_reg = ClassicalRegister(qubits)
        else:
            q_reg = QuantumRegister(N)
            cl_reg = ClassicalRegister(N)

        #Selecting the backend
        backend = Aer.get_backend('qasm_simulator',
                                  shots = 1,
                                  max_parallel_threads = 0,
                                  max_parallel_experiments = 0,
                                  max_parallel_shots = 1)

        #Create the time list and the list (of lists) that will contain the values of the populations
        tlist = np.arange(0,t_max+dt,dt)
        populations = np.zeros([len(tlist),N])

        #To simplify the work divide the dynamics into trajectory chunks of length
        chunk_len = 100

        #Creating the random numbers for the energy fluctuations
        random_increments = np.sqrt(Gamma/tau)*np.random.randn(N,len(tlist),shots)

        qc_Trotter_step, energy_params = self.__sys_free_evolution(H, N, dt)
        qc_Trotter_step = transpile(qc_Trotter_step, backend=backend)

        
        for traj_chunk in range(0, shots, chunk_len):
            qcs = []
            for traj in range(chunk_len):
                H_fluc = np.array(random_increments[:, 0, traj_chunk + traj])

                qc_lead = QuantumCircuit(q_reg, cl_reg) #Lead of the evlution
                if self.mapping == 'physical':
                    qc_lead.x(0)

                for nt in range(len(tlist)):
                    if nt > 0:
                        energies = np.diag(H) + H_fluc
                        qc_lead.compose(qc_Trotter_step.bind_parameters(dict(zip(energy_params, energies.tolist()))), inplace=True)
                        H_fluc = self.__fluctuation_update(H_fluc, random_increments[:, nt, traj_chunk + traj], tau, dt)
                    qc_copy = qc_lead.copy(name = 'circuit_{}'.format(nt))
                    qc_copy.measure(q_reg, cl_reg)
                    qcs.append(qc_copy)

            #Solving the circuits
            qobjs = assemble(qcs, backend=backend)
            options = {'max_parallel_threads':0, 'max_parallel_experiments':0, 'max_parallel_shots':0}
            job_info = backend.run(qobjs, shots = 1, options = options)

            #Append the results to populations
            for nq, qc in enumerate(qcs):
                counts = job_info.result().get_counts(qc)
                ntime = nq%len(tlist)
                for i in range(N):
                    try:
                        populations[ntime,i] = (populations[ntime,i] + counts['{:b}'.format(1<<i).zfill(N)]/shots) if self.mapping == 'physical' else (populations[ntime,i] + counts['{:b}'.format(i).zfill(qubits)]/shots)
                    except:
                        pass

            yield populations, traj_chunk + chunk_len

    def minimal_circuit(self, H, N, dt, backend, transpiled):
        qc = self.__sys_free_evolution(H, N, dt)[0]
        if transpiled == True:
            qc = transpile(qc, backend=backend, basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset'])
        return qc

class _CA():
    def __init__(self, mapping):
        '''Working class for Classical Noise Algorithm, algorithmic mapping'''
        self.mapping = mapping

    def __sys_free_evolution(self, H, N, dt):
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
            qc = QuantumCircuit(q_reg)
            H = np.pad(H, ((0,2**qubits-N), (0,2**qubits-N)), mode='constant', constant_values=0)
            sys_encoded = []
            site_energies_dict = {}
            for index in range(N):
                sys_encoded.append('{:b}'.format(index).zfill(qubits)) #Creating a list with the bit strings representing the sites of the system
                #Creating the Pauli strings corresponding to the projectors
                Pauli_ops = []
                Pauli_coefs = []
                for i in range(qubits):
                    Pauli_ops.append(['I','Z'])
                    Pauli_coefs.append([1./2,1./2]) if sys_encoded[index][i] == '0' else Pauli_coefs.append([1./2,-1./2])
                Pauli_strings = list(itertools.product(*Pauli_ops))
                Pauli_strings = [''.join(Pauli_strings[i]) for i in range(len(Pauli_strings))]
                Pauli_coefs_comb = list(itertools.product(*Pauli_coefs))
                Pauli_coefs_comb = [H[index,index]*prod(Pauli_coefs_comb[i]) for i in range(len(Pauli_coefs_comb))]
                for i, Pauli_string in enumerate(Pauli_strings):
                    try:
                        site_energies_dict[Pauli_string] = site_energies_dict[Pauli_string] + Pauli_coefs_comb[i]
                    except:
                        site_energies_dict[Pauli_string] = Pauli_coefs_comb[i]
            projectors_op = PauliTrotterEvolution().convert((PauliSumOp(SparsePauliOp(list(site_energies_dict.keys()),coeffs=list(site_energies_dict.values()))) * dt).exp_i())
            qc.compose(projectors_op.to_circuit(),range(qubits),inplace=True)

            interactions_dict = {}
            for index in range(N-1):
                for index2 in range(index+1, N):
                    #Creating the Pauli strings corresponding to the interaction operators
                    Pauli_ops = []
                    Pauli_coefs = []
                    for i in range(qubits):
                        if sys_encoded[index][i] == sys_encoded[index2][i]:
                            Pauli_ops.append(['I','Z'])
                            Pauli_coefs.append([1./2,1./2]) if sys_encoded[index][i] == '0' else Pauli_coefs.append([1./2,-1./2])
                        else:
                            Pauli_ops.append(['X'])
                            Pauli_coefs.append([1])
                    Pauli_strings = list(itertools.product(*Pauli_ops))
                    Pauli_strings = [''.join(Pauli_strings[i]) for i in range(len(Pauli_strings))]
                    Pauli_coefs_comb = list(itertools.product(*Pauli_coefs))
                    Pauli_coefs_comb = [H[index,index2]*prod(Pauli_coefs_comb[i]) for i in range(len(Pauli_coefs_comb))]
                    for i, Pauli_string in enumerate(Pauli_strings):
                        try:
                            interactions_dict[Pauli_string] = interactions_dict[Pauli_string] + Pauli_coefs_comb[i]
                        except:
                            interactions_dict[Pauli_string] = Pauli_coefs_comb[i]
            interactions_op = PauliTrotterEvolution().convert((PauliSumOp(SparsePauliOp(list(interactions_dict.keys()),coeffs=list(interactions_dict.values()))) * dt).exp_i())
            qc.compose(interactions_op.to_circuit(),range(qubits),inplace=True)

        else:
            q_reg = QuantumRegister(N)
            qc = QuantumCircuit(q_reg)
            for i in range(N):
                qc.rz(-H[i,i]*dt,q_reg[i])
            for i in range(N-1):
                for j in range(i+1,N):
                    if H[i,j] != 0:
                        qc.rxx(H[i,j]*dt,q_reg[i],q_reg[j])
                        qc.ryy(H[i,j]*dt,q_reg[i],q_reg[j])
        
        return qc

    def __Pseudomodes_ops(self, qubits_per_pseudomode):
        #Gray-code encoding for a pseudomode. Every pseudomode will be taken with the same encoding
        pseudo_encoded = []
        for i in range(1 << qubits_per_pseudomode):
            gray=i^(i>>1)
            pseudo_encoded.append('{0:0{1}b}'.format(gray,qubits_per_pseudomode))
        
        #Creating dictionaries for c, c^dagger and c+c^dagger operators of the pseudomodes
        c_dagger_dict = {}
        for index in range(2**qubits_per_pseudomode-1):
            Pauli_ops = []
            Pauli_coefs = []
            for i in range(qubits_per_pseudomode):
                if pseudo_encoded[index + 1][i] == pseudo_encoded[index][i]:
                    Pauli_ops.append(['I','Z'])
                    Pauli_coefs.append([1./2,1./2]) if pseudo_encoded[index + 1][i] == '0' else Pauli_coefs.append([1./2,-1./2])
                else:
                    Pauli_ops.append(['X','Y'])
                    Pauli_coefs.append([1/2,1.j/2]) if pseudo_encoded[index + 1][i] == '0' else Pauli_coefs.append([1./2,-1.j/2])
            Pauli_strings = list(itertools.product(*Pauli_ops))
            Pauli_strings = [''.join(Pauli_strings[i]) for i in range(len(Pauli_strings))]
            Pauli_coefs_comb = list(itertools.product(*Pauli_coefs))
            Pauli_coefs_comb = [np.sqrt(index+1)*prod(Pauli_coefs_comb[i]) for i in range(len(Pauli_coefs_comb))]
            for i, Pauli_string in enumerate(Pauli_strings):
                try:
                    c_dagger_dict[Pauli_string] = c_dagger_dict[Pauli_string] + Pauli_coefs_comb[i]
                except:
                    c_dagger_dict[Pauli_string] = Pauli_coefs_comb[i]
        
        c_dict = {}
        for index in range(1, 1 << qubits_per_pseudomode):
            Pauli_ops = []
            Pauli_coefs = []
            for i in range(qubits_per_pseudomode):
                if pseudo_encoded[index -1][i] == pseudo_encoded[index][i]:
                    Pauli_ops.append(['I','Z'])
                    Pauli_coefs.append([1./2,1./2]) if pseudo_encoded[index - 1][i] == '0' else Pauli_coefs.append([1./2,-1./2])
                else:
                    Pauli_ops.append(['X','Y'])
                    Pauli_coefs.append([1/2,1.j/2]) if pseudo_encoded[index - 1][i] == '0' else Pauli_coefs.append([1./2,-1.j/2])
            Pauli_strings = list(itertools.product(*Pauli_ops))
            Pauli_strings = [''.join(Pauli_strings[i]) for i in range(len(Pauli_strings))]
            Pauli_coefs_comb = list(itertools.product(*Pauli_coefs))
            Pauli_coefs_comb = [np.sqrt(index)*prod(Pauli_coefs_comb[i]) for i in range(len(Pauli_coefs_comb))]
            for i, Pauli_string in enumerate(Pauli_strings):
                try:
                    c_dict[Pauli_string] = c_dict[Pauli_string] + Pauli_coefs_comb[i]
                except:
                    c_dict[Pauli_string] = Pauli_coefs_comb[i]
        
        c_plus_c_dagger_dict = {}
        for key in c_dict:
            if c_dict[key] != -c_dagger_dict[key]:
                c_plus_c_dagger_dict[key] = c_dict[key] + c_dagger_dict[key]
        
        return c_dagger_dict, c_dict, c_plus_c_dagger_dict

    def __sys_pseudo_interaction(self, N, dt, Gamma, tau, c_plus_c_dagger_dict, qubits_per_pseudomode):
        #Initializing the registers
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
            pseudo_reg = [QuantumRegister(qubits_per_pseudomode,name='pseudomode_{}'.format(i)) for i in range(N)]
            qc = QuantumCircuit(q_reg, *pseudo_reg)
            sys_encoded = []
            for index in range(N):
                sys_encoded.append('{:b}'.format(index).zfill(qubits)) #Creating a list with the bit strings representing the sites of the system
                #Creating the Pauli strings corresponding to the projectors
                Pauli_ops = []
                Pauli_coefs = []
                for i in range(qubits):
                    Pauli_ops.append(['I','Z'])
                    Pauli_coefs.append([1./2,1./2]) if sys_encoded[index][i] == '0' else Pauli_coefs.append([1./2,-1./2])
                Pauli_strings_sys = list(itertools.product(*Pauli_ops))
                Pauli_strings_sys = [''.join(Pauli_strings_sys[i]) for i in range(len(Pauli_strings_sys))]
                Pauli_coefs_comb_sys = list(itertools.product(*Pauli_coefs))
                Pauli_coefs_comb_sys = [prod(Pauli_coefs_comb_sys[i]) for i in range(len(Pauli_coefs_comb_sys))]
                Pauli_interaction_dict = {}
                for i, Pauli_string in enumerate(Pauli_strings_sys):
                    for Pauli_string_pseudo, Pauli_coef_pseudo in c_plus_c_dagger_dict.items():
                        Pauli_interaction_dict[Pauli_string + Pauli_string_pseudo] = Pauli_coefs_comb_sys[i] * Pauli_coef_pseudo
                pseudo_sys_op = PauliTrotterEvolution().convert((PauliSumOp(SparsePauliOp(list(Pauli_interaction_dict.keys()), coeffs=list(Pauli_interaction_dict.values()))) * dt).exp_i())
                qubit_list = pseudo_reg[index][0:qubits_per_pseudomode]
                qubit_list.extend(q_reg[0:qubits])
                qc.compose(pseudo_sys_op.to_circuit(),qubit_list,inplace=True)

        else:
            q_reg = QuantumRegister(N)
            pseudo_reg = [QuantumRegister(qubits_per_pseudomode,name='pseudomode_{}'.format(i)) for i in range(N)]
            qc = QuantumCircuit(q_reg, *pseudo_reg)
            c_plus_c_dagger = PauliSumOp(SparsePauliOp(list(c_plus_c_dagger_dict.keys()),coeffs=list(c_plus_c_dagger_dict.values())))
            pseudo_sys_Hamiltonian = - np.sqrt(Gamma/tau) / 2 * Z ^ c_plus_c_dagger
            pseudo_sys_op = PauliTrotterEvolution().convert((pseudo_sys_Hamiltonian * dt).exp_i())
            for i in range(N):
                qubit_list = pseudo_reg[i][0:qubits_per_pseudomode]
                qubit_list.append(q_reg[i])
                qc.compose(pseudo_sys_op.to_circuit(),qubit_list,inplace=True)

        return qc

    def __Trotter_step(self, H, N, dt, tau, Gamma, qubits_per_pseudomode):
        #Initializing the registers
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
        else:
            q_reg = QuantumRegister(N)
        pseudo_reg = [QuantumRegister(qubits_per_pseudomode,name='pseudomode_{}'.format(i)) for i in range(N)]
        ancilla = QuantumRegister(1,name='ancilla')
        qc = QuantumCircuit(q_reg, *pseudo_reg, ancilla)

        #System self-Hamiltonian free propagation
        qc.compose(self.__sys_free_evolution(H,N,dt),q_reg,inplace=True)

        #Pseudomodes operators
        c_dagger_dict, c_dict, c_plus_c_dagger_dict = self.__Pseudomodes_ops(qubits_per_pseudomode)
        #Converting dictionaries to PauliOp objects
        c_dagger = PauliSumOp(SparsePauliOp(list(c_dagger_dict.keys()),coeffs=list(c_dagger_dict.values())))
        c = PauliSumOp(SparsePauliOp(list(c_dict.keys()),coeffs=list(c_dict.values())))
        #Pseudomodes-System interaction
        qc.compose(self.__sys_pseudo_interaction(N, dt, Gamma, tau, c_plus_c_dagger_dict, qubits_per_pseudomode), inplace=True)
        #Pseudomodes-ancilla collisions
        pseudo_ancilla_Hamiltonian = ((X + (1.j * Y)) ^ c_dagger) + ((X - (1.j * Y)) ^ c)
        pseudo_ancilla_Hamiltonian = pseudo_ancilla_Hamiltonian.reduce() #Simplify the primitive SparsePauliOp by combining duplicates and removing zeros
        pseudo_ancilla_op = PauliTrotterEvolution().convert((pseudo_ancilla_Hamiltonian * np.sqrt(2/tau/dt) / 2 *dt).exp_i())
        for i in range(N):
            qubit_list = pseudo_reg[i][0:qubits_per_pseudomode]
            qubit_list.append(ancilla[0])
            qc.compose(pseudo_ancilla_op.to_circuit(),qubit_list,inplace=True)
            qc.reset(ancilla[0])

        return qc

    def solve(self, H, N, dt, t_max, tau, Gamma, shots, qubits_per_pseudomode):
        #Initializing the registers
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
            cl_reg = ClassicalRegister(qubits)
        else:
            q_reg = QuantumRegister(N)
            cl_reg = ClassicalRegister(N)
        pseudo_reg = [QuantumRegister(qubits_per_pseudomode) for _ in range(N)]
        ancilla = QuantumRegister(1)

        #Selecting the backend
        backend = Aer.get_backend('qasm_simulator',
                                  max_parallel_threads = 0,
                                  max_parallel_shots = 0)

        #Create the time list and the list (of lists) that will contain the values of the populations
        tlist = np.arange(0,t_max+dt,dt)
        populations = [[] for i in range(N)]

        #To simplify the work divide the dynamics into time chunks of length
        chunk_len = 100

        #Creating the quanutm circuits
        qc_lead = QuantumCircuit(q_reg, *pseudo_reg, ancilla, cl_reg) #Lead of the evlution
        qc_Trotter_step = self.__Trotter_step(H, N, dt, tau, Gamma, qubits_per_pseudomode) #Incremental block of the evolution
        qc_Trotter_step = transpile(qc_Trotter_step, backend=backend)

        #Initializing the circuit to the first site (in case of the algorithmic mapping no actions are needed)
        if self.mapping == 'physical':
            qc_lead.x(0)

        #Propagating in time
        for nt_chunk in range(0, len(tlist), chunk_len):
            #Creating a list with all the quantum circuit for a parallelized evaluation
            qcs = []
            for nt in range(chunk_len):
                qc_copy = qc_lead.copy(name = 'circuit_{}'.format(nt))
                qc_copy.measure(q_reg, cl_reg)
                qcs.append(qc_copy)
                qc_lead.compose(qc_Trotter_step, inplace=True)

            #Solving the circuits
            qobjs = assemble(qcs, backend=backend)
            options = {'max_parallel_threads':0, 'max_parallel_shots':0}
            job_info = backend.run(qobjs, shots = shots, options = options)

            #Append the results to populations
            for qc in qcs:
                counts = job_info.result().get_counts(qc)
                for i in range(N):
                    try:
                        populations[i].append(counts['{:b}'.format(1<<i).zfill(N)]/shots) if self.mapping == 'physical' else populations[i].append(counts['{:b}'.format(i).zfill(qubits)]/shots)
                    except:
                        populations[i].append(0)

            yield populations, tlist(nt_chunk + chunk_len - 1)

    def minimal_circuit(self, H, N, dt, tau, Gamma, qubits_per_pseudomode, backend, transpiled):
        if transpiled == True:
            qc = transpile(self.__Trotter_step(H, N, dt, tau, Gamma, qubits_per_pseudomode), backend=backend, basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset'])
        else:
            qc = self.__Trotter_step(H, N, dt, tau, Gamma, qubits_per_pseudomode)
        return qc
