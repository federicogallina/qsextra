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
    def delay_time(self)->np.ndarray:
        return self._delay_time

    @delay_time.setter
    def delay_time(self,
                   delay,
                   ):
        self._delay_time = np.array(delay)

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
        self._label = label
        self._delay_time = delay_time
        self._side_seq = None
        self._direction_seq = None

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
    _linear_name_list = ["a", "abs"]
    _thirdorder_name_list = ["gsb", "se", "esa"]

    def __init__(self,
                 FD_label: str,
                 delay_time: np.ndarray | list[list[float]] | list[float] | float,
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

        delay_time: numpy.ndarray | list[list[float]] | list[float] | float
            A list with the delay times. For the linear absorption the input is: `list[float]` or `float`. For third-order responses the input is a list with 3 entres (i.e.: `[T1_list, T2_list, T3_list]`), therefore the accepted types are: `list[list[float]]` or `list[float]`.
        '''
        super.__init__()
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
            delay_time: np.ndarray | list[list[float]] | list[float] | float,
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

        delay_time: numpy.ndarray | list[list[float]] | list[float] | float
            A list with the delay times. For the linear absorption the input is: `list[float]` or `float`. For third-order responses the input is a list with 3 entres (i.e.: `[T1_list, T2_list, T3_list]`), therefore the accepted types are: `list[list[float]]` or `list[float]`.
        '''
        # Checking the Feynman diagram.
        FD_label = FD_label.casefold()
        if not isinstance(FD_label, str):
            raise TypeError()
        if FD_label not in self._linear_name_list and FD_label not in self._thirdorder_name_list:
            raise ValueError('Unsupported Feynman diagram.')
        # Checking the correct input format for delay_time. Converting float to list if necessary.
        if FD_label in self._linear_name_list:
            delay_time = if_scalar_to_list(delay_time)
            if isinstance(delay_time, np.ndarray):
                delay_time = delay_time.tolist()
            elif not isinstance(delay_time, list):
                raise TypeError('delay_time is a {}. However, a list, numpy.ndarray or a float are expected.'.format(type(delay_time)))
        elif FD_label in self._thirdorder_name_list:
            if isinstance(delay_time, np.ndarray):
                delay_time = delay_time.tolist()
            if not isinstance(delay_time, list):
                raise TypeError('delay_time is a {}. However, a list is expected.'.format(type(delay_time)))
            if len(delay_time) != 3:
                raise ValueError('delay_time contains {} items. However, 3 are expected.'.format(len(delay_time)))
            for i, T_i in enumerate(delay_time):
                delay_time[i] = if_scalar_to_list(T_i)
                if isinstance(T_i, np.ndarray):
                    delay_time[i] = T_i.tolist()
                elif not isinstance(T_i, list):
                    raise TypeError('delay_time[{}] is a {}. However, a list, numpy.ndarray or a float are expected.'.format(i, type(T_i)))
        
        # Saving type of Feynamn diagram and the list of delay times.
        self.label = FD_label
        self.delay_time = delay_time

        # Generate the associated sequences
        self.__FD_to_sequence()
