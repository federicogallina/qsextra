from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, assemble
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliSumOp, X, Y, Z, PauliTrotterEvolution
from qiskit.circuit import Parameter
import numpy as np
import itertools
from math import prod


def evolve(Hamiltonian, t_max, dt, gamma, shots=10000, method='classical noise', mapping='physical'):
    '''Quantum algorithm to compute the dynamics of the open system. Return the populations of the target site during time.
    Input parametes:
    - Hamiltonian [list, np.array]: the Hamiltonian of the open system in an algorithmic mapping
    - t_max [float, double]: maximum time of the dynamics (included)    
    - dt [float, double]: time step of the dynamics
    - gamma [float, double]: noise strength
    - shots [int]: number of shots
    - method ['classical noise', 'collision model']: quantum algorithm used to compute the dynamics
    - mapping ['algorithmic', 'physical']: mapping of the system on the qubits
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
        return _CNA(mapping).solve(H, N, dt, t_max, gamma, shots)
    if method == 'collision model':
        return _CA(mapping).solve(H, N, dt, t_max, gamma, shots)

def minimal_circuit(Hamiltonian, method='classical noise', mapping='physical', backend=None, gamma = 1, dt = 1, transpiled = True):
    '''Returns the quantum circuit for a time step evolution.
    Input parametes:
    - Hamiltonian [list, np.array]: the Hamiltonian of the open system in an algorithmic mapping
    - method ['classical noise', 'collision model']: quantum algorithm used to compute the dynamics
    - mapping ['algorithmic', 'physical']: mapping of the system on the qubits
    - gamma [float, double]: noise strength
    - dt [float, double]: time step of the dynamics
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
        return _CA(mapping).minimal_circuit(H, N, dt, gamma, backend, transpiled)

class _CNA():
    def __init__(self, mapping):
        '''Working class for Classical Noise Algorithm, algorithmic mapping'''
        self.mapping = mapping

    def __CNOT_staircase_method(self, N, dt, Pauli_dict):
        qubits = int(np.ceil(np.log2(N)))
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

    def __sys_free_evolution(self, H, N, dt, backend):
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
            qc.compose(self.__CNOT_staircase_method(N, dt, site_energies_dict),inplace=True)

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
            
        qc = transpile(qc, backend=backend)
        
        return qc, energy_params
    
    def solve(self, H, N, dt, t_max, tau, gamma, shots):
        #Initializing the registers
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
            cl_reg = ClassicalRegister(qubits)
        else:
            q_reg = QuantumRegister(N)
            cl_reg = ClassicalRegister(N)

        #Selecting the backend
        backend = Aer.get_backend('qasm_simulator')

        #Create the time list and the list (of lists) that will contain the values of the populations
        tlist = np.arange(0,t_max+dt,dt)
        populations = np.zeros([len(tlist),N])

        #Creating the quanutm circuits and initializing the circuit to the first site (in case of the algorithmic mapping no extra actions are needed)
        qc_lead = QuantumCircuit(q_reg, cl_reg) #Lead of the evlution
        if self.mapping == 'physical':
            qc_lead.x(0)

        #Creating the random numbers for the energy fluctuations
        random_increments = np.sqrt(gamma/dt)*np.random.randn(N,len(tlist),shots)

        qc_Trotter_step, energy_params = self.__sys_free_evolution(H, N, dt, backend)

        for traj in range(shots):
            qcs = []
            H_fluc = np.array(random_increments[:,0,traj])

            for nt, t in enumerate(tlist):
                qc_copy = qc_lead.copy(name = 'circuit_{}'.format(nt))
                qc_copy.measure(q_reg,cl_reg)
                qcs.append(qc_copy)
                energies = np.diag(H) + random_increments[:,nt,traj]
                qc_lead.compose(qc_Trotter_step.bind_parameters(dict(zip(energy_params, energies.tolist()))), inplace=True)

            #Solving the circuits
            qobjs = assemble(qcs, backend=backend)
            options = {'max_parallel_threads':0, 'max_parallel_experiments':0, 'max_parallel_shots':0}
            job_info = backend.run(qobjs, shots = 1, options = options)

            #Getting results
            for nq, qc in enumerate(qcs):
                counts = job_info.result().get_counts(qc)
                for i in range(N):
                    try:
                        populations[nq,i] = (populations[nq,i] + counts['{:b}'.format(1<<i).zfill(N)]/shots) if self.mapping == 'physical' else (populations[nq,i] + counts['{:b}'.format(i).zfill(qubits)]/shots)
                    except:
                        pass

        return populations

    def minimal_circuit(self, H, N, dt, backend, transpiled):
        qc, params = self.__sys_free_evolution(H, N, dt, backend)
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

    def __sys_anc_interaction(self, N, dt, gamma):
        #Initializing the registers
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
            ancilla = QuantumRegister(1,name='ancilla')
            qc = QuantumCircuit(q_reg, ancilla)
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
                    Pauli_interaction_dict[Pauli_string + 'X'] = Pauli_coefs_comb_sys[i] * np.sqrt(gamma/dt)
                pseudo_sys_op = PauliTrotterEvolution().convert((PauliSumOp(SparsePauliOp(list(Pauli_interaction_dict.keys()), coeffs=list(Pauli_interaction_dict.values()))) * dt).exp_i())
                qubit_list = [ancilla[0]]
                qubit_list.extend(q_reg[0:qubits])
                qc.compose(pseudo_sys_op.to_circuit(),qubit_list,inplace=True)
                qc.reset(ancilla[0])

        else:
            q_reg = QuantumRegister(N)
            ancilla = QuantumRegister(1,name='ancilla')
            qc = QuantumCircuit(q_reg, ancilla)
            sys_ancilla_Hamiltonian = np.sqrt(gamma/4/dt) * X ^ Z
            sys_ancilla_op = PauliTrotterEvolution().convert((sys_ancilla_Hamiltonian * dt).exp_i())
            for i in range(N):
                qubit_list = [ancilla[0]]
                qubit_list.append(q_reg[i])
                qc.compose(sys_ancilla_op.to_circuit(),qubit_list,inplace=True)
                qc.reset(ancilla[0])

        return qc

    def __Trotter_step(self, H, N, dt, gamma, backend):
        #Initializing the registers
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
        else:
            q_reg = QuantumRegister(N)
        ancilla = QuantumRegister(1,name='ancilla')
        qc = QuantumCircuit(q_reg, ancilla)

        #System self-Hamiltonian free propagation
        qc.compose(self.__sys_free_evolution(H,N,dt),q_reg,inplace=True)

        #Pseudomodes-ancilla collisions
        qc.compose(self.__sys_anc_interaction(N, dt, gamma),inplace=True)
        
        qc = transpile(qc, backend=backend)

        return qc

    def solve(self, H, N, dt, t_max, gamma, shots):
        #Initializing the registers
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
            cl_reg = ClassicalRegister(qubits)
        else:
            q_reg = QuantumRegister(N)
            cl_reg = ClassicalRegister(N)
        ancilla = QuantumRegister(1)

        #Selecting the backend
        backend = Aer.get_backend('qasm_simulator')

        #Create the time list and the list (of lists) that will contain the values of the populations
        tlist = np.arange(0,t_max+dt,dt)
        populations = [[] for i in range(N)]

        #Creating the quanutm circuits
        qc_lead = QuantumCircuit(q_reg, ancilla, cl_reg) #Lead of the evlution
        qc_Trotter_step = self.__Trotter_step(H, N, dt, gamma, backend) #Incremental block of the evolution
        
        #Initializing the circuit to the first site (in case of the algorithmic mapping no actions are needed)
        if self.mapping == 'physical':
            qc_lead.x(0)

        #Creating a list with all the quantum circuit for a parallelized evaluation
        qcs = []

        #Propagating in time
        for nt, t in enumerate(tlist):
            qc_copy = qc_lead.copy(name = 'circuit_{}'.format(nt))
            qc_copy.measure(q_reg,cl_reg)
            qcs.append(qc_copy)
            qc_lead.compose(qc_Trotter_step, inplace=True)

        #Solving the circuits
        qobjs = assemble(qcs, backend=backend)
        options = {'max_parallel_threads':0, 'max_parallel_shots':0}
        job_info = backend.run(qobjs, shots = shots, options = options)

        #Getting results
        for qc in qcs:
            counts = job_info.result().get_counts(qc)
            for i in range(N):
                try:
                    populations[i].append(counts['{:b}'.format(1<<i).zfill(N)]/shots) if self.mapping == 'physical' else populations[i].append(counts['{:b}'.format(i).zfill(qubits)]/shots)
                except:
                    populations[i].append(0)

        return populations

    def minimal_circuit(self, H, N, dt, gamma, backend, transpiled):
        if transpiled == True:
            qc = transpile(self.__Trotter_step(H, N, dt, gamma, backend), backend=backend, basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset'])
        else:
            qc = self.__Trotter_step(H, N, dt, gamma, backend)
        return qc
