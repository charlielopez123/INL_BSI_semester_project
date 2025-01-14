Lighthouse lib
==============

Installation
------------

If you already have poetry installed and configured:

.. code-block:: bash

    poetry install

Then to activate/exit the virtual env:

.. code-block:: bash

    poetry shell # to activate environment
    exit # to exit

Configuration
-------------

To prevent hardcoded paths, we use environment variables at a user level that are easy 
to get in python with :code:`os.environ[<VARIABLE>]`.

Please set up the following:

.. code-block:: bash
    
    DATA_DIR=/path/to/data_directory
    MLFLOW_DIR=/path/to/mlflow_directory

If you need to set a new environment variable, please update this doc.

Code Guidelines
---------------

The repository enforces that code is formatted by 
`black <https://black.readthedocs.io/en/stable/>`_ and 
`isort` on dev and main branches.


`poetry`
----------

`poetry <https://python-poetry.org/docs/>`_ is a tool that leverages `venv` 
and is IMO the best tool when *authoring* Python packages that will be shared. 
`poetry` makes it easy to create a virtual environment for a specific package 
and to manage the list of dependencies for that package.

If there is a chance that you will modify this project in a way that will 
affect dependencies, then I strongly suggest that you use `poetry`.

Follow the `official instruction to install poetry <https://python-poetry.org/docs/#installing-with-the-official-installer>`_.


.. note::

    If using poetry then I also recommend configuring it to put the 
    `.venv in-project <https://python-poetry.org/docs/configuration/#virtualenvsin-project>`_. 
    This makes it easier to clean up and easier to find by IDEs:

.. code-block:: bash

    poetry config virtualenvs.in-project true

.. note::
    
    Poetry users should prefix each non-poetry command with `poetry run` 
    (e.g. `poetry run python path/to/script.py`) to use the poetry-managed environment
    , or semi-permanently activate an environment context with `poetry shell`; 
    `exit` leaves the context. 


Documentation
-------------

The documentation is automatically built on `dev` and `main` branches. It mainly uses \*.rst format for 
the documentation (`cheatsheet here <https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`_).
You can find the documentation in the `docs` directory.

To build the documentation manually, run:

.. code-block:: bash

    cd docs
    make html # or make.bat html on Windows

And launch the `index.html` file in the `build/html` directory.


`mlflow`
--------

`mlflow <https://mlflow.org/docs/latest/index.html>`_ is a tool that helps manage
machine learning experiments. It is a good idea to use it to track experiments
and to save models.

To use it, you need to create a directory where the experiments will be saved.
Then you need to set the environment variable `MLFLOW_DIR` to point to
that directory.

To start the mlflow server, run:

.. code-block:: bash

    mlflow server --backend-store-uri /path/to/mlflow_dir --host 127.0.0.1 --port 8080

and browse to the address `http://127.0.0.1:8080` to see the mlflow UI.
