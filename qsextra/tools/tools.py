from qutip import tensor, Qobj
from itertools import permutations
import numpy as np

def if_scalar_to_list(a):
    ''' Convert a scalar input to a list whose only item is that scalar. If non scalar objects are given as an input, return the object without changes.
    '''
    if np.isscalar(a):
        a = [a]
    return a

def kron(*args) -> Qobj:
    ''' Kronecker product between multiple Qutip Qobj objects.

    Parameters
    ----------

    args: Qobj
        An ordered sequence of qutip.Qobj objects.
    '''
    if len(args) == 1:
        return args[0]
    return tensor(args[0], kron(*args[1:]))

"""
def permutations_of_func(iterable: list,
                         r: int = None,
                         func = None,
                         mul: float = 1,
                         ):
    ''' Given a certain function, some iterables and r, return a list with the results of the function applied to successive r length permutations of elements in the iterable.        
    To multiply a value to each element in the list, please specify mul.
    '''
    if func is None:
        return permutations(iterable, r)
    pof_list = []
    for perm in permutations(iterable, r):
        pof_list.append(mul * func(*perm))
    return pof_list
"""

def spectral_function(central_frequencies: float | list[float],
                      height: float | list[float],
                      width: float | list[float],
                      frequency_range: list[float] | np.ndarray | None = None,
                      negative_frequencies: bool = True,
                      ):
    ''' Return a spectral function composed of the sum of Lorentzians.

    Parameters
    ----------

    central_frequencies: float | list[float]
        The central frequencies of the Lorentzians. Positive values are expected.

    height: float | list[float]
        The heights at the central frequencies.

    width: float | list[float]
        The width of the Lorentzians such that: y = height/2 --> x = central_frequency +- width.

    frequency_range: list[float] | np.ndarray | None
        The range of (positive) frequencies at which the spectral function is calculated.

    negative_frequencies: bool
        If True, the returned frequency range extends to negative frequencies. The spectral function at negative frequencies is specular to the one at positive frequencies.

    Returns
    -------
    fr: np.ndarray
        The frequency range.
    sf: np.ndarray
        The spectral function evaluated at fr.
    '''
    central_frequencies = if_scalar_to_list(central_frequencies)
    height = if_scalar_to_list(height)
    width = if_scalar_to_list(width)
    if frequency_range is None:
        f_max = np.max(np.array(central_frequencies)) + 4 * np.max(np.array(width))
        df = np.min(np.array(width))/100
        fr = np.arange(0, f_max + df, df, dtype = float)
    else:
        fr = frequency_range
    sf = np.zeros_like(fr, dtype = float)
    for k in range(len(height)):
        sf += height[k] * width[k] ** 2 / ((fr - central_frequencies[k]) ** 2 + width[k] ** 2)
    if negative_frequencies:
        fr = np.concatenate((-np.flip(fr[1:]), fr))
        sf = np.concatenate((np.flip(sf[1:]), sf))
    return fr, sf

def to_gray(input: int | str,
            binary: bool = False,
            ) -> str:
    ''' Make the Grey encoding of a number given as an input.

    Parameters
    ----------
    input: int | str
        The input number. If input is a decimal, int and str are accepted. If input is a binary, str is expected.

    binary: bool
        If True, the input is supposed to be a binary. Otherwise, it is supposed to be a decimal.

    Returns
    -------
    str
        The Gray code string of the input.
    '''
    if binary:
        input = int(input, 2)
    else:
        input = int(input)
    return input ^ (input >> 1)

def gray_code_list(n: int) -> list[str]:
    ''' Return a list of Gray codes for numbers from 0 to 2**n.

    Parameters
    ----------
    n: int
        Length of the Gray code strings (total number of bits).

    Returns
    -------
    list[str]
        A list of Gray codes.
    '''
    pseudo_encoded = []
    for i in range(1 << n):
        gray = to_gray(i)
        pseudo_encoded.append('{0:0{1}b}'.format(gray,n))
    return pseudo_encoded