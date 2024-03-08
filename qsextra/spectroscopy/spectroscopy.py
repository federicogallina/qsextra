'''
Part of this code was originally included in the qudofedi package (https://github.com/federicogallina/qudofedi).
'''

import numpy as np
from qsextra.tools import if_scalar_to_list

class Spectroscopy():

    @property
    def label(self)->str:
        return self._label

    @label.setter
    def label(self,
              label,
              ):
        self._label = label

    @property
    def delay_time(self)->list[list]:
        return self._delay_time

    @delay_time.setter
    def delay_time(self,
                   delays,
                   ):
        ''' Convert the input to a list of lists:

        Parameters
        ----------
        delays: float | list[float] | numpy.ndarray | list[numpy.ndarray] | list[list[float]]
            - `float`: `T` is converted to `[[T]]`.
            - `list[float]`: `[Ta, Tb, Tc]` is converted to `[[Ta, Tb, Tc]]`.
            - `numpy.ndarray`: `numpy.array([Ta, Tb, Tc])` is converted to `[[Ta, Tb, Tc]]`.
            - `list[numpy.ndarray]`: `[numpy.array([T1a, T1b, T1c]), numpy.array([T2a, T2b, T2c]), numpy.array([T3a, T3b, T3c])]` is converted to `[[T1a, T1b, T1c], [T2a, T2b, T2c], [T3a, T3b, T3c]]`.
            - `list[list[float]]`: `[[T1a, T1b, T1c], [T2a, T2b, T2c], [T3a, T3b, T3c]]` is preserved.
        '''
        if np.isscalar(delays):
            dys = [[delays]]
        elif isinstance(delays, np.ndarray):
            dys = [delays.tolist()]
        elif isinstance(delays, list):
            if all([np.isscalar(delay) for delay in delays]):
                dys = [delays]
            else:
                dys = []
                for delay in delays:
                    if np.isscalar(delay):
                        dy = [delay]
                    elif isinstance(delay, np.ndarray):
                        dy = delay.tolist()
                    elif isinstance(delay, list):
                        dy = delay
                    dys.append(dy)
        else:
            raise TypeError('delay_time is a {}. However, a list, numpy.ndarray or a float are expected.'.format(type(delays)))
        self._delay_time = dys

    @property
    def side_seq(self)->str:
        return self._side_seq

    @side_seq.setter
    def side_seq(self,
                 side_sequence,
                 ):
        self._side_seq = side_sequence

    @property
    def direction_seq(self)->str:
        return self._direction_seq

    @direction_seq.setter
    def direction_seq(self,
                      direction_sequence,
                      ):
        self._direction_seq = direction_sequence
    
    def __init__(self,
                 delay_time: np.ndarray | list[list[float]] | list[float] | float,
                 label: str = "",
                 ):
        self.label = label
        self.delay_time = delay_time
        self.side_seq = ''
        self.direction_seq = ''

    def todict(self) -> dict:
        '''
        Returns
        -------
        dict
            A dictionary with object data.
        '''
        initial_dict = self.__dict__
        sp_dict = {}
        sp_dict['class'] = type(self)
        for key in initial_dict:
            new_key = key if key[0] != '_' else key[1:]
            sp_dict[new_key] = initial_dict[key]
            if isinstance(initial_dict[key], np.ndarray):
                sp_dict[new_key] = initial_dict[key].tolist()
        return sp_dict

class FeynmanDiagram(Spectroscopy):
    _linear_list = ["a", "abs"]
    _thirdorder_list = ["gsb", "se", "esa"]

    def __init__(self,
                 FD_label: str,
                 delay_time: float | list[float] | np.ndarray | list[np.ndarray] | list[list[float]],
                 ):
        '''
        Create an object that contains the information about the contribution of the response function to be simulated.

        Parameters
        ----------
        FD_label: str
            The type of Feynman diagram. At the moment, only linear absorption and the components of the third-order rephasing signal are implemented:
            - `"a"`: Linear absorption
            - `"abs"`: Linear absorption (alias)
            - `"gsb"`: Third-order rephasing Ground State Bleaching
            - `"se"`:  Third-order rephasing Stimulated Emission
            - `"esa"`: Third-order rephasing Excited State Absorption

        delay_time: float | list[float] | numpy.ndarray | list[numpy.ndarray] | list[list[float]]
            A list with the delay times.
            For the linear absorption the input is: `float`, `list[float]` or `numpy.ndarray`.
            For third-order responses the input is a list with 3 lists (i.e.: `[[T1], [T2], [T3]]`), therefore the accepted types are: `list[list[float]]` or `list[numpy.ndarray]`.
            - `float`: `T` is converted to `[[T]]`.
            - `list[float]`: `[Ta, Tb, Tc]` is converted to `[[Ta, Tb, Tc]]`.
            - `numpy.ndarray`: `numpy.array([Ta, Tb, Tc])` is converted to `[[Ta, Tb, Tc]]`.
            - `list[numpy.ndarray]`: `[numpy.array([T1a, T1b, T1c]), numpy.array([T2a, T2b, T2c]), numpy.array([T3a, T3b, T3c])]` is converted to `[[T1a, T1b, T1c], [T2a, T2b, T2c], [T3a, T3b, T3c]]`.
            - `list[list[float]]`: `[[T1a, T1b, T1c], [T2a, T2b, T2c], [T3a, T3b, T3c]]` is preserved.
        '''
        self.set(FD_label, delay_time)

    def __FD_to_sequence(self):
        if self.label == "a" or self.label == "abs":
            self.side_seq = 'kk'
            self.direction_seq = 'io'
        elif self.label == "gsb":
            self.side_seq = 'bbkk'
            self.direction_seq = 'ioio'
        elif self.label == "se":
            self.side_seq = 'bkbk'
            self.direction_seq = 'iioo'
        elif self.label == "esa":
            self.side_seq = 'bkkk'
            self.direction_seq = 'iiio'
        
    def set(self,
            FD_label: str,
            delay_time: float | list[float] | np.ndarray | list[np.ndarray] | list[list[float]],
            ):
        '''
        Create an object that contains the information about the contribution of the response function to be simulated.

        Parameters
        ----------
        FD: str
            The type of Feynman diagram. At the moment, only linear absorption and the components of the third-order rephasing signal are implemented:
            - `"a"`: Linear absorption
            - `"abs"`: Linear absorption (alias)
            - `"gsb"`: Third-order rephasing Ground State Bleaching
            - `"se"`:  Third-order rephasing Stimulated Emission
            - `"esa"`: Third-order rephasing Excited State Absorption

        delay_time: float | list[float] | numpy.ndarray | list[numpy.ndarray] | list[list[float]]
            A list with the delay times.
            For the linear absorption the input is: `float`, `list[float]` or `numpy.ndarray`.
            For third-order responses the input is a list with 3 lists (i.e.: `[[T1], [T2], [T3]]`), therefore the accepted types are: `list[list[float]]` or `list[numpy.ndarray]`.
            - `float`: `T` is converted to `[[T]]`.
            - `list[float]`: `[Ta, Tb, Tc]` is converted to `[[Ta, Tb, Tc]]`.
            - `numpy.ndarray`: `numpy.array([Ta, Tb, Tc])` is converted to `[[Ta, Tb, Tc]]`.
            - `list[numpy.ndarray]`: `[numpy.array([T1a, T1b, T1c]), numpy.array([T2a, T2b, T2c]), numpy.array([T3a, T3b, T3c])]` is converted to `[[T1a, T1b, T1c], [T2a, T2b, T2c], [T3a, T3b, T3c]]`.
            - `list[list[float]]`: `[[T1a, T1b, T1c], [T2a, T2b, T2c], [T3a, T3b, T3c]]` is preserved.
        '''
        # Checking the Feynman diagram.
        FD_label = FD_label.casefold()
        if not isinstance(FD_label, str):
            raise TypeError()
        if FD_label not in self._linear_list and FD_label not in self._thirdorder_list:
            raise ValueError('Unsupported Feynman diagram.')
        self.label = FD_label
        
        # Checking the correct input format for delay_time. Converting float to list if necessary.
        self.delay_time = delay_time
        if self.label in self._linear_list:
            if len(self.delay_time) != 1:
                raise ValueError('delay_time contains {} items. However, 1 is expected.'.format(len(self.delay_time)))
        if self.label in self._thirdorder_list:
            if len(self.delay_time) != 3:
                raise ValueError('delay_time contains {} items. However, 3 are expected.'.format(len(self.delay_time)))
            
        # Generate the associated sequences
        self.__FD_to_sequence()
