""" Data loader module for Lighthouse. dt5, gdp, smr, ... enjoy!

.. note::

    If you want to add a new data loader, please try to follow the same structure as the 
    existing ones. Especially, the function has to return data with the following structure:
    
    ``signals, times, ch_names``
    
    they can be lists, np.ndarray, or a function can return multiple tuples of them.
"""
