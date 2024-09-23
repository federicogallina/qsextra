import numpy as np
from scipy.linalg import ishermitian
import warnings
from typing import Literal
from qutip import (Qobj,
                   sigmaz,
                   identity,
                   destroy,
                   basis,
                   zero_ket,
                   qzero,
                   )
from qsextra.tools import kron, if_scalar_to_list

class ExcitonicSystem():

    @property
    def e_el(self):
        return self._e_el
    
    @property
    def system_size(self):
        return self._system_size
    
    @property
    def dipole_moments(self):
        return self._dipole_moments
    
    @property
    def coupl_el(self):
        return self._coupl_el
    
    @property
    def state_type(self):
        return self._state_type

    @property
    def state(self):
        return self._state

    @e_el.setter
    def e_el(self,
             energies: list[float] | float | np.ndarray,
             ):
        if self._state_type is None:
            if np.isscalar(energies):
                energies = np.array([energies])
            elif isinstance(energies, list):
                energies = np.array(energies)
            if not isinstance(energies, np.ndarray):
                raise TypeError('energies is in a wrong format ({}).'.format(type(energies)))
            if len(energies.shape) > 1:
                raise Exception('energies must be a list, a number (for a single chromophore) or a monodimensional array.')
            self._e_el = energies
            self._system_size = energies.size
            self.__update_validity()
        else:
            raise Exception('{} parameters cannot be changed after a state has been set.'.format(type(self)))

    @system_size.setter
    def system_size(self,
                    size: int,
                    ):
        if self._state_type is None:
            self._system_size = size
            self.__update_validity()
        else:
            raise Exception('{} parameters cannot be changed after a state has been set.'.format(type(self)))

    @dipole_moments.setter
    def dipole_moments(self,
                       dipole_moments: list[float] | float | np.ndarray | None,
                       ):
        if self._state_type is None:
            if dipole_moments is None:
                warnings.warn('Equal dipole moments are considered.')
                self.dipole_moments = [1.] * self.system_size
            else:
                if np.isscalar(dipole_moments):
                    dipole_moments = np.array([dipole_moments])
                elif isinstance(dipole_moments, list):
                    dipole_moments = np.array(dipole_moments)
                if not isinstance(dipole_moments, np.ndarray):
                    raise TypeError('dipole_moments is in a wrong format ({}).'.format(type(dipole_moments)))
                if len(dipole_moments.shape) > 1:
                    raise Exception('dipole_moments must be a list, a number (for a single chromophore) or a monodimensional array.')
                self._dipole_moments = dipole_moments
            self.__update_validity()
        else:
            raise Exception('{} parameters cannot be changed after a state has been set.'.format(type(self)))

    @coupl_el.setter
    def coupl_el(self,
                 couplings: list[list[float]] | float | np.ndarray | None,
                 ):
        if self._state_type is None:
            if couplings is None:
                warnings.warn('No coupling between chromophores is considered.')
                self.coupl_el = [[0 for _ in range(self.system_size)] for _ in range(self.system_size)]
            else:
                if np.isscalar(couplings) and self.system_size == 2:
                    couplings = np.array([[0, couplings], [couplings, 0]])
                if isinstance(couplings, list) and isinstance(couplings[0], list):
                    couplings = np.array(couplings)
                elif not isinstance(couplings, np.ndarray):
                    raise TypeError('couplings is in a wrong format ({}).'.format(type(couplings)))
                couplings -= np.diag(np.diag(couplings))
                if ishermitian(couplings) == False:
                    raise Exception('couplings is not Hermitian, but it must be.')
                self._coupl_el = couplings
            self.__update_validity()
        else:
            raise Exception('{} parameters cannot be changed after a state has been set.'.format(type(self)))

    def __init__(self,
                 energies: list[float] | float | np.ndarray = None,
                 couplings: list[list[float]] | float | np.ndarray = None,
                 dipole_moments: list[float] | float | np.ndarray = None,
                 ):
        r""" Create an `ExcitonicSystem` object.
        An `ExcitonicSystem` object contains the information about the excitonic part of a chromophore system intended as a collection of two-level electronic systems with a groung |0> and an excited |1> states.
        The exciton Hamiltonian is

        .. math:: H^{e} = - \sum_{i=1}^{N} \epsilon_{i}/2 \sigma^{z}_{i} + \sum_{i=1}^{N-1} \sum_{j>1}^{N} ( J_{ij} \sigma^{+}_{i} \sigma^{-}_{j} + h.c. )

        where :math:`\epsilon` are the energies and :math:`J` are the couplings.
        :math:`\sigma^{z}` is the Pauli-z operator (defined as |0><0| - |1><1|), while :math:`\sigma^{+}` (:math:`\sigma^{-}`) are the rising (lowering) ladder operators.
        
        Parameters
        ----------
        energies: list[float] | float | numpy.ndarray
            The electronic energy gaps of the chromphores of the network. It is a list, a number (for a single chromophore) or a monodimensional array.

        couplings: list[list[float]] | float | numpy.ndarray
            The electronic couplings between different chromophores of the network. It is in form of an Hermitian matrix with zeros in the diagonal. An off-diagonal element (i,j) represents the coupling that chromophore i experiences due to chromophore j. If the system is composed of only 2 chromophores, a float value is enough.

        dipole_moments: list[float] | float | numpy.ndarray
            The amplitude of the transition dipole moments of the chromophores. It is a list, a number (for a single chromophore) or a monodimensional array.
        """
        self._system_size = 0
        self._e_el = None
        self._dipole_moments = None
        self._coupl_el = None
        self._validity = False
        self._state_type = None
        self._state = None
        if energies is not None:
            self.electronics(energies = energies,
                             couplings = couplings,
                             dipole_moments = dipole_moments,
                             )

    def __update_validity(self,
                          ):
        try:
            if self.e_el.size == self.system_size and self.dipole_moments.size == self.system_size and self.coupl_el.shape[0] == self.system_size:
                self._validity = True
            else:
                if self.e_el.size != self.system_size:
                    warnings.warn('The number of elements in energies ({}) is different from the system size (number of chromophores = {}). Make sure they match in order to have a valid {}.'.format(self.e_el.size, self.system_size, type(self)))
                if self.dipole_moments.size != self.system_size:
                    warnings.warn('The number of elements in dipole_moments ({}) is different from the system size (number of chromophores = {}). Make sure they match in order to have a valid {}.'.format(self.dipole_moments.size, self.system_size, type(self)))
                if self.coupl_el.shape[0] != self.system_size:
                    warnings.warn('The number of elements in couplings ({}x{}) is different from that expected ({}x{}). Make sure they match in order to have a valid {}.'.format(self.coupl_el.shape[0], self.coupl_el.shape[0], self.system_size, self.system_size, type(self)))
                self._validity = False
        except:
            self._validity = False

    def electronics(self,
                    energies: list[float] | float | np.ndarray,
                    couplings: list[list[float]] | float | np.ndarray = None,
                    dipole_moments: list[float] | float | np.ndarray = None,
                    ):
        self.e_el = energies
        self.system_size = self.e_el.size
        self.dipole_moments = dipole_moments
        if self.dipole_moments.size != self.system_size:
            raise Exception('The number of elements in dipole_moment ({}) is different from the system size (number of chromophores = {}).'.format(self.dipole_moments.size, self.system_size))
        self.coupl_el = couplings
        if self.coupl_el.shape[0] != self.system_size:
            raise Exception('The number of elements in couplings ({}x{}) is different from that expected ({}x{}).'.format(self.coupl_el.shape[0], self.coupl_el.shape[0], self.system_size, self.system_size))

    def set_state(self,
                  state_type: Literal['state', 'delocalized excitation', 'localized excitation', 'ground'] = 'ground',
                  state: list | np.ndarray | int = 0,
                  ):
        r""" Set the `ExcitonicSystem` electronic ket state.
        Note that in this version of the code only pure states are accepted.

        Parameters
        ----------
        state_type: str
            Type of the input state. It must be one of the following: 'state', 'delocalized excitation', 'localized excitation', 'ground'.
        
        state: list | np.ndarray | int
            The state of the system. The input is given by the selected state_type:
            - `'state'`: It requires a list of coefficients for the :math:`2^{N}` states that compose the electronic Hilbert space. A standard binary ordering and local basis set are intended, that is to say, for N=3 state `[0., 0., 0., 1.+0.j, 0., 0., 0., 0.] = |011>` denotes a double excitation state with both chromophore 0 and 1 excited. 
            - `'delocalized excitation'`: It takes a list of N (complex) coefficients that describe the superposition of single excitation states. For example, for N=3, `[c_001, c_010, c_100] = [0., 1./sqrt(2), 1./sqrt(2)] = 1/sqrt(2) |010> + 1/sqrt(2) |100>`.
            - `'localized excitation'`: It takes an integer that denotes the excited chromophore. Chromophores are numbered from 0 to N-1.
            - `'ground'`: It does not require any input. The system is in the global ground state.
        """
        valid_state_types = ('state', 'delocalized excitation', 'localized excitation', 'ground')
        if not self._validity:
            raise Exception('Make sure system electronic parameters (energies, dipole moment intensities and couplings) are specified and consistent before setting the state.')
        if state_type not in valid_state_types:
            raise ValueError('state_type must be one of {}'.format(valid_state_types))
        if state_type == 'state':
            if not isinstance(state, list) and not isinstance(state, np.ndarray):
                raise TypeError('For a state, state must be an list.')
            state = np.array(state, dtype='complex')
            if len(state.shape) > 1:
                raise Exception('state must be a monodimensional array.')
            if state.size != 2**self.system_size:
                raise Exception('The number of elements in state ({}) is different from the Hilbert space dimension ({}).'.format(state.size, 2**self.system_size))
            norm = np.sum(np.abs(state)**2)
            if norm != 1.:
                state = state/np.sqrt(norm)
                warnings.warn('Coefficients have been normalized.')
            state = state.tolist()
        elif state_type == 'delocalized excitation':
            if not isinstance(state, list) and not isinstance(state, np.ndarray):
                raise TypeError('For a delocalized excitation, state must be an list.')
            state = np.array(state, dtype='complex')
            if len(state.shape) > 1:
                raise Exception('state must be a monodimensional array.')
            if state.size != self.system_size:
                raise Exception('The number of elements in state ({}) is different from the system size (number of chromophores = {}).'.format(state.size, self.system_size))
            norm = np.sum(np.abs(state)**2)
            if norm != 1.:
                state = state/np.sqrt(norm)
                warnings.warn('Coefficients have been normalized.')
            state = state.tolist()
        elif state_type == 'localized excitation':
            if not isinstance(state, int):
                raise TypeError('For a localized excitation, state must be an integer.')
            if state >= self.system_size or state < 0:
                raise ValueError('state must be an int between 0 and {}'.format(self.system_size))
        elif state_type == 'ground':
            state = 0
        self._state_type = state_type
        self._state = state
    
    def todict(self) -> dict:
        '''
        Returns
        -------
        dict
            A dictionary with object data.
        '''
        initial_dict = self.__dict__
        sys_dict = {}
        sys_dict['class'] = type(self)
        for key in initial_dict:
            new_key = key if key[0] != '_' else key[1:]
            sys_dict[new_key] = initial_dict[key]
            if isinstance(initial_dict[key], np.ndarray):
                sys_dict[new_key] = initial_dict[key].tolist()
        return sys_dict

    def get_e_Hamiltonian(self):
        ''' Returns the Frenkel-exciton Hamiltonian.
        '''
        if self._validity:
            sz = sigmaz()
            sm = destroy(2)
            sp = destroy(2).dag()
            I = identity(2)
            H = qzero(dimensions = [2]*self.system_size)
            for i in range(self.system_size):
                for j in range(i, self.system_size):
                    if i != j:
                        H += self.coupl_el[i,j] * (kron(*[I]*(self.system_size-j-1), sm, *[I]*(j-i-1), sp, *[I]*i) +
                                                   kron(*[I]*(self.system_size-j-1), sp, *[I]*(j-i-1), sm, *[I]*i))
                    else:
                        H += - self.e_el[i]/2 * kron(*[I]*(self.system_size-i-1), sz, *[I]*i)
            return H
        else:
            raise Exception("The system is not valid. Please, reset the parameters.")

    def get_e_state(self):
        ''' Returns the ket state of the system.
        '''
        if self._validity:
            if self.state_type == 'ground':
                state = basis([2] * self.system_size, [0] * self.system_size)
            elif self.state_type == 'state':
                state = Qobj(np.array(self.state), dims = [[2] * self.system_size, [1] * self.system_size], type = 'ket')
            elif self.state_type == 'delocalized excitation':
                state = zero_ket(dimensions = [2] * self.system_size)
                for nc, c in enumerate(self.state):
                    position = position = [0] * self.system_size
                    position[nc] = 1
                    position.reverse()
                    state += c * basis([2] * self.system_size, position)
            elif self.state_type == 'localized excitation':
                position = [0 for _ in range(self.system_size)]
                position[self.state] = 1
                position.reverse()
                state = basis([2] * self.system_size, position)
            return state
        else:
            raise Exception("The system is not valid. Please, reset the parameters.")

class ChromophoreSystem(ExcitonicSystem):

    @property
    def mode_dict(self):
        return self._mode_dict
    
    @mode_dict.setter
    def mode_dict(self,
                  md,
                  ):
        self._mode_dict = md

    def __init__(self,
                 energies: list[float] | float | np.ndarray = None,
                 couplings: list[list[float]] | float | np.ndarray = None,
                 dipole_moments: list[float] | float | np.ndarray = None,
                 excitonic_system: ExcitonicSystem = None,
                 frequencies_pseudomode: list[float] | float = None,
                 levels_pseudomode: int = None,
                 couplings_ep: list[float] | float = None,
                 ):
        r""" Create a `ChromophoreSystem` object.
        A `ChromophoreSystem` object contains the information about the whole chromophore intended as a collection of two-level electronic systems with a groung |0> and an excited |1> states.
        The exciton Hamiltonian is

        .. math:: H^{e} = - \sum_{i=1}^{N} \epsilon_{i}/2 \sigma^{z}_{i} + \sum_{i=1}^{N-1} \sum_{j>1}^{N} ( J_{ij} \sigma^{+}_{i} \sigma^{-}_{j} + h.c. )

        where :math:`\epsilon` are the energies and :math:`J` are the couplings.
        :math:`\sigma^{z}` is the Pauli-z operator (defined as |0><0| - |1><1|), while :math:`\sigma^{+}` (:math:`\sigma^{-}`) are the rising (lowering) ladder operators.

        Parameters
        ----------
        energies: list[float] | float | numpy.ndarray
            The electronic energy gaps of the chromphores of the network. It is a list, a number (for a single chromophore) or a monodimensional array.

        couplings: list[list[float]] | float | numpy.ndarray
            The electronic couplings between different chromophores of the network. It is in form of an Hermitian matrix with zeros in the diagonal. An off-diagonal element (i,j) represents the coupling that chromophore i experiences due to chromophore j. If the system is composed of only 2 chromophores, a float value is enough.

        dipole_moments: list[float] | float | numpy.ndarray
            The amplitude of the transition dipole moments of the chromophores. It is a list, a number (for a single chromophore) or a monodimensional array.

        excitonic_system: ExcitonicSystem
            Initialize the electronic part of the `ChromophoreSystem` to the given `ExcitonicSystem`.

        frequencies_pseudomode: list[float] | float
            The frequencies of the pseudomodes. If a single pseudomode per chromophore is considered, a float value is accepted.

        levels_pseudomode: int
            Number of energy levels per pseudomode. Introduces a necessary truncation of the Hilbert space.

        couplings_ep: list[float] | float
            The system-pseudomode coupings between electronic and pseudomode degrees of freedom. If a single pseudomode per chromophore is considered, a float value is accepted.
        """
        if type(excitonic_system) is ExcitonicSystem:
            super().__init__(energies = excitonic_system.e_el,
                             couplings = excitonic_system.coupl_el,
                             dipole_moments = excitonic_system.dipole_moments,
                             )
            try:
                self.set_state(excitonic_system.state_type, excitonic_system.state)
            except:
                pass
        else:
            super().__init__(energies = energies,
                             couplings = couplings,
                             dipole_moments = dipole_moments,
                             )
        self._mode_dict = None
        if frequencies_pseudomode is not None:
                self.pseudomodes(frequencies_pseudomode = frequencies_pseudomode,
                                 levels_pseudomode = levels_pseudomode,
                                 couplings_ep = couplings_ep,
                                 )

    def pseudomodes(self,
                    frequencies_pseudomode: list[float] | float = None,
                    levels_pseudomode: list[int] | int = None,
                    couplings_ep: list[float] | float = None,
                    ):
        r"""
        It is possible to couple harmonic pseudomodes to the electronic degrees of freedom.
        The pseudomode environment are assumed to be identical for every chromophore.
        In this version of the code, when this function is called, the state of the pseudomodes is set as the state where only the lower level is populated.

        Parameters
        ----------
        frequencies_pseudomode: list[float] | float
            The frequencies of the pseudomodes. If a single pseudomode per chromophore is considered, a float value is accepted.

        levels_pseudomode: int
            Number of energy levels per pseudomode. Introduces a necessary truncation of the Hilbert space.

        couplings_ep: list[float] | float
            The system-pseudomode coupings between electronic and pseudomode degrees of freedom. If a single pseudomode per chromophore is considered, a float value is accepted.
        """
        # Checking frequencies_pseudomode
        if frequencies_pseudomode is None:
            raise Exception('frequencies_pseudomode is not specified.')
        else:
            frequencies_pseudomode = if_scalar_to_list(frequencies_pseudomode)
            if not isinstance(frequencies_pseudomode, list):
                raise TypeError('frequencies_pseudomode is in a wrong format ({}).'.format(type(frequencies_pseudomode)))

        # Checking levels_pseudomode
        if levels_pseudomode is None:
            raise Exception('levels_pseudomode is not specified.')
        else:
            levels_pseudomode = if_scalar_to_list(levels_pseudomode)
            if not isinstance(levels_pseudomode, list):
                raise TypeError('levels_pseudomode is in a wrong format ({}).'.format(type(levels_pseudomode)))
            if len(levels_pseudomode) != len(frequencies_pseudomode):
                raise Exception('The number of elements in levels_pseudomode ({}) is different from the nummer of elements in frequencies_pseudomode ({}).'.format(len(levels_pseudomode), len(frequencies_pseudomode)))
            if min(levels_pseudomode) < 2:
                raise Exception('All the entries in levels_pseudomode must be >= 2.')

        # Checking couplings_ep
        if couplings_ep is None:
            raise Exception('couplings_ep is not specified.')
        else:
            couplings_ep = if_scalar_to_list(couplings_ep)
            if not isinstance(couplings_ep, list):
                raise TypeError('couplings_ep is in a wrong format ({}).'.format(type(couplings_ep)))
            if len(couplings_ep) != len(frequencies_pseudomode):
                raise Exception('The number of elements in couplings_ep ({}) is different from the nummer of elements in frequencies_pseudomode ({}).'.format(len(couplings_ep), len(frequencies_pseudomode)))
            
        # Setting the pseudomode state
        state_mode = self.__set_mode_state(levels_pseudomode)

        # Setting the dictionary
        pseudomodes_dict = {'omega_mode': frequencies_pseudomode,
                            'lvl_mode': levels_pseudomode,
                            'coupl_ep': couplings_ep,
                            'state_mode': state_mode,
                            }
        
        self.mode_dict = pseudomodes_dict
    
    def __set_mode_state(self,
                         levels_pseudomode: list[int],
                         ):
        """ In this version of the code set a default state in which only the |0> state is occupied.
        """
        state_mode = [[] for _ in levels_pseudomode]
        for i in range(len(levels_pseudomode)):
            state_mode[i] = [1.+0.j]
            state_mode[i] += [0.+0.j] * (levels_pseudomode[i] - 1)
        return state_mode
    
    def get_state_mode(self,
                       mode_number: int | None = None,
                       ):
        """ If mode_number is specified, return the state of the relative pseudomode. Otherwise, return a list with all the states.
        """
        if mode_number is None:
            states = []
            for i in range(len(self.mode_dict['state_mode'])):
                state = zero_ket(dimensions = self.mode_dict['lvl_mode'][i])
                for j in range(len(self.mode_dict['state_mode'][i])):
                    state += self.mode_dict['state_mode'][i][j] * basis(self.mode_dict['lvl_mode'][i], j)
                states.append(state)
            return states
        else:
            if not isinstance(mode_number, int):
                raise TypeError('mode_number must be int or None.')
            if mode_number >= len(self.mode_dict['lvl_mode']):
                raise ValueError('mode_number must be lower than {} or None.'.format(len(self.mode_dict['lvl_mode'])))
            state = zero_ket(dimensions = self.mode_dict['lvl_mode'][mode_number])
            for j in len(self.mode_dict['state_mode'][mode_number]):
                state += self.mode_dict['state_mode'][mode_number][j] * basis(self.mode_dict['lvl_mode'][mode_number], j)
            return state
    
    def get_p_Hamiltonian(self):
        W = len(self.mode_dict['omega_mode'])
        d = self.mode_dict['lvl_mode']
        Id = [identity(d[k]) for k in range(W)]
        H = qzero(dimensions = d)
        for k in range(W):
            a = destroy(d[k])
            H += self.mode_dict['omega_mode'][k] * kron(*Id[:k],
                                                        a.dag() * a,
                                                        *Id[k+1:],
                                                        )
        return H

    def get_ep_Hamiltonian(self):
        N = self.system_size
        W = len(self.mode_dict['omega_mode'])
        d = self.mode_dict['lvl_mode']
        Id = [identity(d[k]) for k in range(W)]
        I = identity(2)
        sz = sigmaz()
        H = qzero(dimensions = [2]*N + d*N)
        for i in range(N):
            for k in range(W):
                a = destroy(d[k])
                H += self.mode_dict['coupl_ep'][k]/2 * kron(*[I]*(N-i-1),
                                                            I-sz,
                                                            *[I]*i,
                                                            *Id*(N-i-1),
                                                            *Id[:k],
                                                            a.dag() + a,
                                                            *Id[k+1:],
                                                            *Id*i,
                                                            )
        return H
    
    def get_global_Hamiltonian(self):
        N = self.system_size
        W = len(self.mode_dict['omega_mode'])
        d = self.mode_dict['lvl_mode']
        Id = [identity(d[k]) for k in range(W)]
        I = identity(2)
        H_e = self.get_e_Hamiltonian()
        H_p = self.get_p_Hamiltonian()
        H_ep = self.get_ep_Hamiltonian()
        H = (kron(H_e, *Id*N) +
             H_ep)
        for i in range(N):
            H += kron(*[I]*N,
                      *Id*(N-i-1),
                      H_p,
                      *Id*i,
                      )
        return H

    def extract_ExcitonicSystem(self):
        e_sys = ExcitonicSystem(energies = self.e_el,
                                couplings = self.coupl_el,
                                dipole_moments = self.dipole_moments,
                                )
        try:
            e_sys.set_state(self.state_type, self.state)
        except:
            warnings.warn('State is not defined.')
        return e_sys
    
    