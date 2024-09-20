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
        sx = sigmax()
        sm = destroy(2)
        sp = destroy(2).dag()
        # First interaction
        if spectroscopy.side_seq[n_interaction] == 'b' and first_b:
            dipole_op = sx
            first_b = False
        elif spectroscopy.side_seq[n_interaction] == 'k' and first_k:
            dipole_op = sx
            first_k = False
        # ket out | bra in
        elif (spectroscopy.side_seq[n_interaction] == 'k') ^ (spectroscopy.direction_seq[n_interaction] == 'i'):
            dipole_op = sm
        # ket in | bra out
        else:
            dipole_op = sp
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
            return kron(*[I] * (N - site_index - 1), operator, *[I] * site_index, *Id_pseudomodes * N)

    def dipole_application(dm: Qobj,
                           dipole_op: Qobj,
                           side: str,
                           ):
        if side == 'b':
            dm_output = dm * dipole_op
        elif side == 'k':
            dm_output = dipole_op * dm
        return dm_output

    # Defining system operators
    N = system.system_size
    sx = sigmax()

    # Defining the initial (ground) state of the system
    state_0 = basis([2]*N, [0]*N)
    # Adding pseudomodes if ChromophoreSystem
    if type(system) is ChromophoreSystem:
        W = len(system.mode_dict['omega_mode'])
        d = system._mode_dict['lvl_mode']
        Id_pseudomodes = [identity(d[k]) for k in range(W)]
        state_modes = system.get_state_mode()
        state_0 = kron(state_0,
                      *[state_modes[k] for k in range(W)] * N,
                      )
    else:
        Id_pseudomodes = None

    # Defining the list with the expectation operators (terminal dipole moment which can be applied to any chromophore)
    e_op_list = [system.dipole_moments[i] * build_operator(N, i, sx, Id_pseudomodes) for i in range(N)]

    # Defining the output variable (signal)
    size_signal = [len(T) for T in spectroscopy.delay_time]
    signal = np.zeros(size_signal, dtype = complex)
    # Defining a list with all the indices of signal
    time_indices_list = combinations_of_indices(size_signal)

    # Doing a similar thing to define the site pathways
    site_pathways_indices_list = combinations_of_indices([N] * len(spectroscopy.delay_time))

    # Running over the site pathways
    for site_pathways_indices in site_pathways_indices_list:
        # Running over all the possibile combinations of times
        for time_indices in time_indices_list:
            first_k = True
            first_b = True
            dm = ket2dm(state_0)
            # Running over the light-matter interaction events
            for n_interaction in range(len(site_pathways_indices)):
                dipole_op, first_k, first_b = dipole_operator(spectroscopy, first_k, first_b, n_interaction)    # Creating the dipole operator (sigma_x, sigma_+, sigma_-)
                dipole_op = build_operator(N, site_pathways_indices[n_interaction], dipole_op, Id_pseudomodes)    # Adapting dipole operator to the global Hilbert space
                dm = dipole_application(dm, dipole_op, spectroscopy.side_seq[n_interaction])                    # Applying the operator to the density matrix
                try:
                    results = clevolve(system,
                                       [0, spectroscopy.delay_time[n_interaction][time_indices[n_interaction]]],   # Qutip like to start dynamics from t=0
                                       measure_populations = False,
                                       state_overwrite = dm,
                                       verbose = False,
                                       **clevolve_kwds,
                                       )
                except:
                    results = clevolve(system,
                                       spectroscopy.delay_time[n_interaction][0:time_indices[n_interaction] + 1], # Qutip like to start dynamics from t=0, sometimes it also want more points.
                                       measure_populations = False,
                                       state_overwrite = dm,
                                       verbose = False,
                                       **clevolve_kwds,
                                       )
                dm = results.states[-1]
            signal[*time_indices] += np.sum(expect(e_op_list, dm)) * np.prod([system.dipole_moments[i] for i in site_pathways_indices])     # Sum of the expectation values of the dipole operators
    return signal

def clspectroscopy(system: ChromophoreSystem | ExcitonicSystem,
                   spectroscopy: Spectroscopy,
                   verbose: bool = True,
                   **clevolve_kwds,
                   ) -> np.ndarray:
    '''
    Parameters
    ----------
    system: ChromophoreSystem | ExcitonicSystem
        The system to simulate.

    spectroscopy: Spectroscopy
        The `Spectroscopy` object to simulate.

    verbose: bool
        If `True`, print during the execution of the code.

    clevolve_kwds: dict
        Keywords for the `clevolve` method.

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
    
    # Removing special keywords from clevolve_kwds
    clevolve_kwds.pop('system', None)
    clevolve_kwds.pop('time', None)
    clevolve_kwds.pop('measure_populations', None)
    clevolve_kwds.pop('state_overwrite', None)
    clevolve_kwds.pop('verbose', None)

    # Checking Spectroscopy
    if spectroscopy.side_seq == '' or spectroscopy.direction_seq == '':
        raise ValueError('Spectroscopy object is not correctly defined. Spectroscopy.side_seq or Spectroscopy.direction_seq are missing.')

    signal = __run(system, spectroscopy, clevolve_kwds)
    ending_sentence(verbose)
    return signal
