""" This module contains basic utility functions that are used in the lighthouse package. 

*Author: Etienne de Montalivet*
"""

import itertools


def flatten_list(l):
    """Flatten a list of lists.

    Parameters
    ----------
    l : list
        list to flatten

    Returns
    -------
    list
        flatten list
    """
    return list(itertools.chain.from_iterable(l))


def add_prefix(prefix, l):
    """
    Add a prefix to each element of a list.

    Parameters
    ----------
    prefix : str
        The prefix to be added.
    l : list
        The list of strings.

    Returns
    -------
    list
        The resulting list.

    Examples
    --------
    >>> add_prefix('a', ['b', 'c', 'd'])
    ['ab', 'ac', 'ad']
    """
    return [prefix + s for s in l]


def ascii_to_string(l):
    """
    Convert a list of ASCII values to a string.

    Parameters
    ----------
    l : list
        A list of integers representing ASCII values.

    Returns
    -------
    str
        The resulting string.

    Examples
    --------
    >>> ascii_to_string([72, 101, 108, 108, 111])
    'Hello'
    """
    s = ""
    for i in range(len(l)):
        s += chr(l[i])
    return s
