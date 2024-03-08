from qsextra import ExcitonicSystem, ChromophoreSystem
from qsextra.tools import kron
from qsextra.qcomo import clevolve
from qsextra.spectroscopy.spectroscopy import Spectroscopy
import numpy as np
from qutip import (Qobj,
                   sigmax,
                   identity,
                   destroy,
                   expect,
                   basis,
                   ket2dm,
                   )
from itertools import product
import warnings
from qsextra.tools.oracle import ending_sentence

def __run(system: ChromophoreSystem | ExcitonicSystem,
          spectroscopy: Spectroscopy,
          clevolve_kwds: dict = {},
          ) -> np.ndarray:
    
    def combinations_of_indices(sizes: list[int],
                                ):
        indices = [range(size) for size in sizes]
        return list(product(*indices))
    
    def dipole_operator(spectroscopy: Spectroscopy,
                        first_k: bool,
                        first_b: bool,
                        n_interaction: int,
                        ):
        if first_b or first_k:
            dipole_op = sx
            if spectroscopy.side_seq[n_interaction] == 'b':
                first_b = False
            elif spectroscopy.side_seq[n_interaction] == 'k':
                first_k = False
        elif spectroscopy.side_seq[n_interaction] == 'i':
            dipole_op = sp
        elif spectroscopy.side_seq[n_interaction] == 'o':
            dipole_op = sm
        return dipole_op, first_k, first_b

    def build_operator(N: int,
                       site_index: int,
                       operator: Qobj,
                       Id_pseudomodes: list[Qobj] | None,
                       ):
        I = identity(2)
        if Id_pseudomodes is None:
            return kron(*[I] * (N - site_index - 1), operator, *[I] * site_index)
        else:
            return kron(*[I] * (N - site_index - 1), operator, *[I] * site_index, *Id_pseudomodes)

    def dipole_application(dm: Qobj,
                           dipole_op: Qobj,
                           side: str,
                           ):
        if side == 'b':
            return dm * dipole_op
        elif side == 'k':
            return dipole_op * dm

    # Defining system operators
    N = system.system_size
    sx = sigmax()
    sm = destroy(2)
    sp = destroy(2).dag()

    # Defining the initial (ground) state of the system
    state_0 = basis([2]*N, [0]*N)
    # Adding pseudomodes if ChromophoreSystem
    if type(system) is ChromophoreSystem:
        W = len(system.mode_dict['omega_mode'])
        d = system._mode_dict['lvl_mode']
        Id_pseudomodes = [identity(d[k]) for k in range(W)]
        state_0 = kron(state_0, basis(d, [0]*W))
    else:
        Id_pseudomodes = None
    dm = ket2dm(state_0)

    # Defining the list with the expectation operators (terminal dipole moment which can be applied to any chromophore)
    e_op_list = [build_operator(N, j, sx, Id_pseudomodes) for j in range(N)]

    # Defining the output variable (signal)
    size_signal = [len(T) for T in spectroscopy.delay_time]
    signal = np.zeros(size_signal, dtype = complex)
    # Defining a list with all the indices of signal
    time_indices = combinations_of_indices(size_signal)

    # Doing a similar thing to define the site pathways
    site_pathways_indices = combinations_of_indices([N] * len(spectroscopy.delay_time))

    for site_pathways_index in site_pathways_indices:
        for time_index in time_indices:
            first_k = True
            first_b = True
            for n_interaction in range(len(spectroscopy.delay_time)):
                dipole_op, first_k, first_b = dipole_operator(spectroscopy, first_k, first_b, n_interaction)
                dipole_op = build_operator(N, site_pathways_index[n_interaction], dipole_op, Id_pseudomodes)
                dm = dipole_application(dm, dipole_op, spectroscopy.side_seq[n_interaction])
                results = clevolve(system,
                                spectroscopy.delay_time[n_interaction][time_index[n_interaction]],
                                measure_populations = False,
                                state_overwrite = dm,
                                **clevolve_kwds,
                                )
                dm = results.states
            signal[*time_index] += np.sum(expect(e_op_list, dm))
    return signal

def clspectroscopy(system: ChromophoreSystem | ExcitonicSystem,
                 spectroscopy: Spectroscopy,
                 clevolve_kwds: dict = {},
                 verbose: bool = True,
                 ) -> np.ndarray:
    '''
    Parameters
    ----------
    system: ChromophoreSystem | ExcitonicSystem
        The system to simulate.

    spectroscopy: Spectroscopy
        The `Spectroscopy` object to simulate.

    clevolve_kwds: dict
        A dictionary with keywords for the `clevolve` method.

    verbose: bool
        If `True`, print during the execution of the code.

    Returns
    -------
    numpy.ndarray
        An array with the results of the simulation. The dimensions are that of `spectroscopy.delay_time.shape`.
    '''
    # Checking the presence of pseudomodes
    if type(system) is ChromophoreSystem and system.mode_dict is None:
        warnings.warn("Pseudomodes not specified. Executing the dynamics with system as an ExcitonicSystem.")
        return clspectroscopy(system.extract_ExcitonicSystem(),
                              spectroscopy,
                              clevolve_kwds,
                              verbose,
                              )
    signal = __run()    
    ending_sentence(verbose)
    return signal
