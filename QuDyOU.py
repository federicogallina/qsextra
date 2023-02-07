from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.extensions import HamiltonianGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliSumOp, X, Y, Z, PauliTrotterEvolution
from qiskit.compiler import transpile, assemble
from qiskit import Aer, execute
import numpy as np
import scipy as sp
import itertools
from math import prod


class system():
    def __init__(self, Hamiltonian):
        '''Create the system object with basic information.
        Input:
        - Hamiltonian [list, np.array]: the Hamiltonian of the open system in an algorithmic mapping
        '''
        self.H = np.array(Hamiltonian)
        self.N = self.H.shape[0]
    
    def evolve(self, t_max, dt, Gamma, tau, shots=10000, method='classical noise', mapping='physical', qubits_per_pseudomode=1):
        '''Quantum algorithm to compute the dynamics of the open system. Return the populations of the target site during time.
        Input parametes:
        - t_max [float, double]: maximum time of the dynamics (included)    
        - dt [float, double]: time step of the dynamics
        - Gamma [float, double]: equivalent noise strength
        - tau [float, double]: memory time of the fluctuations
        - shots [int]: number of shots
        - method ['classical noise', 'collision model']: quantum algorithm used to compute the dynamics
        - mapping ['algorithmic', 'physical']: mapping of the system on the qubits
        - qubits_per_pseudomode [int]: number of qubits to use in the collision model to implement a pseudomode
        '''
        
        if method not in ['classical noise', 'collision model']:
            raise Exception('Method must be "classical noise" or "collision model"')

        if mapping not in ['algorithmic', 'physical']:
            raise Exception('Method must be "algorithmic" or "physical"')

        if method == 'classical noise':
            return _CNA(mapping).solve(self.H, self.N, dt, t_max, tau, Gamma, shots)
        if method == 'collision model':
            return _CA(mapping).solve(self.H, self.N, dt, t_max, tau, Gamma, shots, qubits_per_pseudomode)

class _CNA():
    def __init__(self, mapping):
        '''Working class for Classical Noise Algorithm, algorithmic mapping'''
        self.mapping = mapping

    def __sys_free_evolution(self, H, N, dt):
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
            qc = QuantumCircuit(q_reg)
            H = np.pad(H, ((0,qubits-N), (0,qubits-N)), mode='constant', constant_values=0)
            sys_encoded = []
            projectors_dict = {}
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
                        projectors_dict[Pauli_string] = projectors_dict[Pauli_string] + Pauli_coefs_comb[i]
                    except:
                        projectors_dict[Pauli_string] = Pauli_coefs_comb[i]
            projectors_op = PauliTrotterEvolution().convert((PauliSumOp(SparsePauliOp(list(projectors_dict.keys()),coeffs=list(projectors_dict.values()))) * dt).exp_i())
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
    
    def fluctuation_update(self, H_fluc, rand_increment, tau, dt):
        H_fluc = H_fluc*np.exp(-dt/tau)+np.diag(rand_increment)*np.sqrt(1-np.exp(-2*dt/tau))
        return H_fluc

    def solve(self, H, N, dt, t_max, tau, Gamma, shots):
        #Initializing the registers
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
        else:
            q_reg = QuantumRegister(N)
        cl_reg = ClassicalRegister(N)

        #Selecting the backend
        backend = Aer.get_backend('qasm_simulator')

        #Create the time list and the list (of lists) that will contain the values of the populations
        tlist = np.arange(0,t_max+dt,dt)
        populations = np.zeros(len(tlist),N)

        #Creating the quanutm circuits and initializing the circuit to the first site (in case of the algorithmic mapping no extra actions are needed)
        qc = QuantumCircuit(q_reg, cl_reg) #Lead of the evlution
        if self.mapping == 'physical':
            qc.x(0)

        #Creating the random numbers for the energy fluctuations
        random_increments = np.sqrt(Gamma/tau)*np.random.randn(N,len(tlist),shots)

        for traj in range(shots):
            qcs = []
            H_fluc = np.diag(random_increments[:,0,traj])

            for nt, t in enumerate(tlist):
                qc_copy = qc.copy(name = 'circuit_{}'.format(nt))
                qc_copy.measure(q_reg,cl_reg)
                qcs.append(qc_copy)
                qc.compose(self.__sys_free_evolution(H + H_fluc, N, dt), inplace=True)
                H_fluc = self.fluctuation_update(H_fluc, random_increments[:,nt,traj], tau, dt)
            
            #Solving the circuits
            transpiled_qcs = transpile(qcs, backend=backend)
            qobjs = assemble(transpiled_qcs, backend=backend)
            options = {'max_parallel_threads':0, 'max_parallel_experiments':0}
            job_info = backend.run(qobjs, shots = 1, options = options)

            #Getting results
            for ntq, transpiled_qc in enumerate(transpiled_qcs):
                counts = job_info.result().get_counts(transpiled_qc)
                for i in range(N):
                    try:
                        populations[ntq,i] = (populations[ntq,i] + counts['{:b}'.format(2**i).zfill(N)]/shots) if self.mapping == 'physical' else (populations[ntq,i] + counts['{:b}'.format(i).zfill(qubits)]/shots)
                    except:
                        pass

        return populations

class _CNA_phys():
    def __init__(self):
        '''Working class for Classical Noise Algorithm, algorithmic mapping'''

    def solve(H, N, dt, t_max, tau, Gamma, shots):
        return 0

class _CA():
    def __init__(self, mapping):
        '''Working class for Classical Noise Algorithm, algorithmic mapping'''
        self.mapping = mapping

    def __sys_free_evolution(self, H, N, dt):
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
            qc = QuantumCircuit(q_reg)
            H = np.pad(H, ((0,qubits-N), (0,qubits-N)), mode='constant', constant_values=0)
            sys_encoded = []
            projectors_dict = {}
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
                        projectors_dict[Pauli_string] = projectors_dict[Pauli_string] + Pauli_coefs_comb[i]
                    except:
                        projectors_dict[Pauli_string] = Pauli_coefs_comb[i]
            projectors_op = PauliTrotterEvolution().convert((PauliSumOp(SparsePauliOp(list(projectors_dict.keys()),coeffs=list(projectors_dict.values()))) * dt).exp_i())
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
        #Grey-code encoding for a pseudomode. Every pseudomode will be taken with the same encoding
        pseudo_encoded = []
        for i in range(0, 1<<qubits_per_pseudomode):
            gray=i^(i>>1)
            pseudo_encoded.append("{0:0{1}b}".format(gray,qubits_per_pseudomode))
        
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
        for index in range(1,2**qubits_per_pseudomode):
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
        
        #Converting dictionaries to PauliOp objects
        c_dagger = PauliSumOp(SparsePauliOp(list(c_dagger_dict.keys()),coeffs=list(c_dagger_dict.values())))
        c = PauliSumOp(SparsePauliOp(list(c_dict.keys()),coeffs=list(c_dict.values())))
        c_plus_c_dagger = PauliSumOp(SparsePauliOp(list(c_plus_c_dagger_dict.keys()),coeffs=list(c_plus_c_dagger_dict.values())))

        return c_dagger, c, c_plus_c_dagger

    def __sys_pseudo_interaction(self, N, dt, Gamma, tau, c_plus_c_dagger, qubits_per_pseudomode):
        #Initializing the registers
        if self.mapping == 'algorithmic':
            qubits = int(np.ceil(np.log2(N)))
            q_reg = QuantumRegister(qubits)
            pseudo_reg = [QuantumRegister(qubits_per_pseudomode,name='pseudomode_{}'.format(i)) for i in range(N)]
            qc = QuantumCircuit(q_reg, *pseudo_reg)

        else:
            q_reg = QuantumRegister(N)
            pseudo_reg = [QuantumRegister(qubits_per_pseudomode,name='pseudomode_{}'.format(i)) for i in range(N)]
            qc = QuantumCircuit(q_reg, *pseudo_reg)
            pseudo_sys_Hamiltonian = - np.sqrt(Gamma/tau) / 2 * Z ^ c_plus_c_dagger
            pseudo_sys_op = PauliTrotterEvolution().convert((pseudo_sys_Hamiltonian * dt).exp_i())
            for i in range(N):
                qubit_list = pseudo_reg[i][0:qubits_per_pseudomode]
                qubit_list.append(q_reg[i])
                qc.compose(pseudo_sys_op.to_circuit(),qubit_list,inplace=True)

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
        qc.compose(self.__sys_free_evolution(H,N,dt),q_reg[0:-1],inplace=True)

        #Pseudomodes operators
        c_dagger, c, c_plus_c_dagger = self.__Pseudomodes_ops(qubits_per_pseudomode)
        #Pseudomodes-System interaction
        qc.compose(self.__sys_pseudo_interaction(self, N, dt, Gamma, tau, c_plus_c_dagger, qubits_per_pseudomode),inplace=True)
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
        else:
            q_reg = QuantumRegister(N)
        pseudo_reg = [QuantumRegister(qubits_per_pseudomode) for i in range(N)]
        ancilla = QuantumRegister(1)
        cl_reg = ClassicalRegister(N)

        #Selecting the backend
        backend = Aer.get_backend('qasm_simulator')

        #Create the time list and the list (of lists) that will contain the values of the populations
        tlist = np.arange(0,t_max+dt,dt)
        populations = [[] for i in range(N)]

        #Creating the quanutm circuits
        qc = QuantumCircuit(q_reg, *pseudo_reg, ancilla, cl_reg) #Lead of the evlution
        qc_Trotter_step = self.__Trotter_step(H, N, dt, tau, Gamma, qubits_per_pseudomode) #Incremental block of the evolution
        
        #Initializing the circuit to the first site (in case of the algorithmic mapping no actions are needed)
        if self.mapping == 'physical':
            qc.x(0)

        #Creating a list with all the quantum circuit for a parallelized evaluation
        qcs = []

        #Propagating in time
        for nt, t in enumerate(tlist):
            qc_copy = qc.copy(name = 'circuit_{}'.format(nt))
            qc_copy.measure(q_reg,cl_reg)
            qcs.append(qc_copy)
            qc.compose(qc_Trotter_step, inplace=True)

        #Solving the circuits
        transpiled_qcs = transpile(qcs, backend=backend)
        qobjs = assemble(transpiled_qcs, backend=backend)
        options = {'max_parallel_threads':0, 'max_parallel_shots':0}
        job_info = backend.run(qobjs, shots = shots, options = options)

        #Getting results
        for transpiled_qc in transpiled_qcs:
            counts = job_info.result().get_counts(transpiled_qc)
            for i in range(N):
                try:
                    populations[i].append(counts['{:b}'.format(2**i).zfill(N)]/shots) if self.mapping == 'physical' else populations[i].append(counts['{:b}'.format(i).zfill(qubits)]/shots)
                except:
                    populations[i].append(0)

        return populations