""" This module contains functions to find directories, files, and paths. """

from pathlib import Path


def find_absd_dirs(path):
    """Find all subdirectories with 'ABSD' or 'absd' in their name.

    Parameters
    ----------
    path : Path
        The path to search for subdirectories

    Returns
    -------
    list
        A list of paths to subdirectories with 'ABSD' or 'absd' in their name
    """
    return list(p.resolve() for p in Path(path).glob("**") if p.is_dir() and "ABSD" in p.name or "absd" in p.name)
