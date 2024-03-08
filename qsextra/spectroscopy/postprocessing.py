'''
Part of this code was originally included in the qudofedi package (https://github.com/federicogallina/qudofedi).
'''

from qsextra.spectroscopy import Spectroscopy
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift

def __linear_rf(response_function,
                delay_time,
                RF_freq = 0,
                damping_rate = 0,
                ):
    '''
    Method that apply the rotating frame to the linear response function.
    '''
    T = np.array(delay_time[0])
    rf_response_function = response_function * np.exp(+ 1.j * RF_freq * T) * np.exp(- damping_rate * T)
    return rf_response_function

def __2D_rf(response_function,
            delay_time,
            RF_freq = 0,
            damping_rate = 0,
            T2_index = 0,
            ):
    '''
    Method that apply the rotating frame to the non-linear response function.
    '''
    T1 = np.array(delay_time[0])
    T2 = np.array(delay_time[1])
    T3 = np.array(delay_time[2])
    if (T2_index >= len(T2)):
        raise ValueError('T2_index exceed length of T2')
    T1, T3 = np.meshgrid(T1, T3, indexing='ij') 
    rf_response_function = response_function[:,T2_index,:] * np.exp(-1.j * RF_freq * (T1 - T3)) * np.exp(- damping_rate * (T1 + T3))
    return rf_response_function

def rotating_frame(response_function,
                   delay_time,
                   RF_freq = 0,
                   damping_rate = 0,
                   T2_index = 0,
                   ):
    if len(delay_time) == 1:
        return __linear_rf(response_function, delay_time, RF_freq, damping_rate)
    elif len(delay_time) == 3:
        return __2D_rf(response_function, delay_time, RF_freq, damping_rate, T2_index)

def __linear_ft(response_function,
                delay_time,
                RF_freq = 0,
                pad_extension = 1,
                ):
    '''
    Method that apply the Fourier Transform to the linear response function.
    '''
    delay_time = delay_time[0]
    dt = delay_time[1] - delay_time[0]
    omega = fftshift(2*np.pi*fftfreq(len(delay_time) * pad_extension, dt)) + RF_freq
    freq_spectra = ifftshift(ifft(np.pad(response_function, ((0, len(delay_time) * (pad_extension-1))),'constant')))
    return omega, freq_spectra

def __2D_ft(response_function,
            delay_time,
            RF_freq = 0,
            T2_index = 0,
            pad_extension = 1,
            ):
    '''
    Method that apply the Fourier Transform to the non-linear response function.
    '''
    T1 = delay_time[0]
    dt1 = T1[1] - T1[0]
    T2 = delay_time[1]
    T3 = delay_time[2]
    dt3 = T3[1] - T3[0]
    if (T2_index >= len(T2)):
        raise ValueError('T2_index exceed length of T2')
    omega1 = fftshift(2*np.pi*fftfreq(len(T1) * pad_extension, dt1)) + RF_freq
    omega3 = fftshift(2*np.pi*fftfreq(len(T3) * pad_extension, dt3)) + RF_freq
    omega1, omega3 = np.meshgrid(omega1, omega3, indexing='ij')
    omega = [omega1, omega3]
    response_function_pad = np.pad(response_function, ((0,len(T1) * (pad_extension-1)), (0,len(T3) * (pad_extension-1))), 'constant')
    freq_spectra = len(response_function_pad)/len(response_function)**2 * fftshift(ifft(fftshift(fft(response_function_pad, axis=0), axes=0), axis=1), axes=1)
    return omega, freq_spectra

def fourier_transform(response_function,
                      delay_time,
                      RF_freq = 0,
                      T2_index = 0,
                      pad_extension = 1,
                      ):
    if len(delay_time) == 1:
        return __linear_ft(response_function, delay_time, RF_freq, pad_extension)
    elif len(delay_time) == 3:
        return __2D_ft(response_function, delay_time, RF_freq, T2_index, pad_extension)
    
def postprocessing(spectroscopy: Spectroscopy,
                   signal: np.ndarray,
                   RF_freq: float = 0,
                   damping_rate: float = 0,
                   pad_extension: int = 1,
                   T2_index: int = 0,
                   ):
    '''
    '''
    response_function = signal
    delay_time = spectroscopy.delay_time

    RF_response_function = rotating_frame(response_function,
                                          delay_time,
                                          RF_freq,
                                          damping_rate, 
                                          T2_index)

    omega, freq_spectrum = fourier_transform(RF_response_function,
                                            delay_time,
                                            RF_freq,
                                            T2_index,
                                            pad_extension)

    return omega, freq_spectrum