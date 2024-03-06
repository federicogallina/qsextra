from qsextra import ExcitonicSystem, ChromophoreSystem
import numpy as np
import warnings
from qutip import (Qobj,
                   mesolve,
                   sesolve,
                   sigmaz,
                   identity,
                   basis,
                   destroy,
                   )
from qutip.solver import Result
from qsextra.tools import kron, if_scalar_to_list
from qsextra.tools.oracle import oracle_word

def ending_sentence(verbose: bool):
    # Define a function to print if verbose
    verboseprint = print if verbose else lambda *a, **k: None
    verboseprint(oracle_word())

def __evolve_exsys(system: ExcitonicSystem,
                   state: Qobj,
                   time: float | list[float] | np.ndarray,
                   rates: float | None,
                   measure_populations: bool,
                   verbose: bool,
                   ):
    def e_ops(N, op):
        I = identity(2)
        ops_list = []
        for i in range(N):
            ops_list.append(kron(*[I]*(N-i-1), op, *[I]*i))
        return ops_list
    if rates is None:
        results = sesolve(system.get_e_Hamiltonian(),
                          state,
                          time,
                          e_ops = e_ops(system.system_size, basis(2,1).proj()) if measure_populations else None,
                          )
    else:
        results = mesolve(system.get_e_Hamiltonian(),
                          state,
                          time,
                          c_ops = e_ops(system.system_size, np.sqrt(rates) * sigmaz()),
                          e_ops = e_ops(system.system_size, basis(2,1).proj()) if measure_populations else None,
                          )
    ending_sentence(verbose)
    return results

def __evolve_cpsys(system: ChromophoreSystem,
                   state: Qobj,
                   time: float | list[float] | np.ndarray,
                   rates: float| list[float] | None,
                   measure_populations: bool,
                   verbose: bool,
                   ):
    def e_ops(N, op, Id_pseudomodes):
        I = identity(2)
        ops_list = []
        for i in range(N):
            ops_list.append(kron(*[I]*(N-i-1), op, *[I]*i, *Id_pseudomodes*N))
        return ops_list
    def c_ops(N, rates, d, Id_pseudomodes):
        I = identity(2)
        ops_list = []
        for i in range(N):
            for k in range(len(rates)):
                a = destroy(d[k])
                ops_list.append(np.sqrt(rates[k]) * kron(*[I]*N,
                                                         *Id_pseudomodes*(N-i-1),
                                                         *Id_pseudomodes[:k],
                                                         a,
                                                         *Id_pseudomodes[k+1:],
                                                         *Id_pseudomodes*i,
                                                         ))
        return ops_list
    W = len(system.mode_dict['omega_mode'])
    d = system._mode_dict['lvl_mode']
    Id = [identity(d[k]) for k in range(W)]
    state0 = kron(state,
                  *[Qobj(np.array(system.mode_dict['state_mode'][k]), type='ket') for k in range(W)]*system.system_size,
                  )
    if rates is None:
        results = sesolve(system.get_global_Hamiltonian(),
                          state0,
                          time,
                          e_ops = e_ops(system.system_size, basis(2,1).proj(), Id) if measure_populations else None,
                          )
    else:
        results = mesolve(system.get_global_Hamiltonian(),
                          state0,
                          time,
                          c_ops = c_ops(system.system_size, rates, d, Id),
                          e_ops = e_ops(system.system_size, basis(2,1).proj(), Id) if measure_populations else None,
                          )
    ending_sentence(verbose)
    return results

def clevolve(system: ChromophoreSystem | ExcitonicSystem,
             time: float | list[float] | np.ndarray,
             rates: float | list[float] | None = None,
             measure_populations: bool = True,
             state_overwrite: Qobj = None,
             verbose: bool = True,
             ) -> Result:
    ''' Quantum algorithm for the simulation of the dynamics of the system.

    Parameters
    ----------
    system: ChromophoreSystem | ExcitonicSystem
        The system to evolve.

    time: float | list[float] | numpy.ndarray
        The target time(s).

    rates: float | list[float] | None
        If None, a Schrödinger dynamics is returned. Else, if system is an ExcitonicSystem object, a float value dictating the system relaxations due to a Markovian environment. If system is a ChromophoreSystem object, a float or list of floats with the relaxation rates of pseudomodes due to the interaction with a Markovian environment.

    measure_populations: bool
        If True, return the expectation values of the populations of the chromophores at the specified times.

    state_overwrite: Qobj
        Specify the (excitonic) state to propagate instead of system.get_e_state().

    verbose: bool
        If True, print during the execution of the code.

    Returns
    -------
    qutip.solver.Result
        Results of the dynamics using Qutip.
    '''
    # Checking the validity of the system
    if not type(system) is ChromophoreSystem and not type(system) is ExcitonicSystem:
        raise TypeError('system must be an instance of class ChromophoreSystem or ExcitonicSystem.')
    if not system._validity:
        raise Exception('Make sure all the system parameters are specified and consistent.')
    
    # Checking the presence of pseudomodes
    if type(system) is ChromophoreSystem and system.mode_dict is None:
        warnings.warn("Pseudomodes not specified. Executing the dynamics with system as an ExcitonicSystem.")
        return clevolve(system.extract_ExcitonicSystem(),
                        time,
                        rates,
                        measure_populations,
                        )

    # Making sure time is itarable
    time = if_scalar_to_list(time)

    # Checking the rates
    if type(system) is ExcitonicSystem:
        if not np.isscalar(rates) and rates is not None:
            raise TypeError('For an ExcitonicSystem, rates must be a scalar.')
    elif type(system) is ChromophoreSystem:
        if rates is not None:
            rates = if_scalar_to_list(rates)
            if len(rates) != len(system.mode_dict['omega_mode']):
                raise ValueError('The number of dephasing rates is different from the number of pseudomodes per chromophore.')

    # Checking the initial state
    if state_overwrite is None:
        state = system.get_e_state()
    else:
        state = state_overwrite

    if type(system) is ExcitonicSystem:
        return __evolve_exsys(system,
                              state,
                              time,
                              rates,
                              measure_populations,
                              verbose,
                              )
    elif type(system) is ChromophoreSystem:
        return __evolve_cpsys(system,
                              state,
                              time,
                              rates,
                              measure_populations,
                              verbose,
                              )