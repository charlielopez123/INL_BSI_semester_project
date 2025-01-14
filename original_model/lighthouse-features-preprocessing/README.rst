Lighthouse documentation
=========================

Installation
------------

If you already have poetry installed and configured:

.. code-block:: bash

    poetry install

Then to activate/exit the virtual env:

.. code-block:: bash

    poetry shell # to activate environment
    exit # to exit


.. warning::

    **Please install pytorch, torchvision and torchaudio manually.**

    This repo aims to be as flexible as possible. Setting up pytorch and torchvision
    would constrain the user to a specific version of pytorch, which
    is not the spirit of this repo. Check the `official pytorch website <https://pytorch.org/get-started/locally/>`_ 
    for the installation instructions.


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

`poetry <https://python-poetry.org/docs/>`_ is a tool that leverages ``venv`` 
and is IMO the best tool when *authoring* Python packages that will be shared. 
``poetry`` makes it easy to create a virtual environment for a specific package 
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
    
    Poetry users should prefix each non-poetry command with ``poetry run`` 
    (e.g. ``poetry run python path/to/script.py``) to use the poetry-managed environment
    , or semi-permanently activate an environment context with ``poetry shell``; 
    ``exit`` leaves the context. 


Documentation
-------------

You can find the documentation in the ``docs/`` directory AND in the code itself.
Doc relies on\*.rst format 
(`cheatsheet here <https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`_).

Docstring code-style
~~~~~~~~~~~~~~~~~~~~	

We use `numpy style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.

In VScode (and probably other IDEs), you can set the docstring style in the settings so
that the IDE will automatically generate the docstrings in the correct style.

Build and view the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The documentation is automatically built on ``dev`` and ``main`` branches. 
To build the documentation manually, run:

.. code-block:: bash

    cd docs
    make html # or make.bat html on Windows

And launch the ``index.html`` file in the ``docs/build/html`` directory.


Workflow
--------

If you're working on a new feature or new analysis, create a new branch from ``dev``:

.. code-block:: bash

    git checkout dev
    git pull
    git checkout -b feature/my_new_feature

Check that your code is correctly formatted and documented. Check the rendering of the documentation
by locally build it (see :ref:`index:Build and view the documentation`).

When you're done, push the branch to the repository and create a pull request on GitHub.  
When the pull request is approved, merge it into ``dev`` and delete the branch.

.. note:: if and only if you're making a minor update such as:

    * minor doc update
    * minor change that does not affect the existing repo

    If you have any doubt, please create a branch and merge it with ``dev``



Authorship
----------

* When you create a new file, add your name to the list of authors in the docstring header.
* When you modify a file, add your name to the list of authors in the docstring header.
    * if major contribution, go first in list
    * if minor contribution, go last in list
* Merge authors when merging files
* If very minor contribution, you can skip adding your name to the list of authors

If any doubt, please ask other contributors.


`mlflow`
--------

`mlflow <https://mlflow.org/docs/latest/index.html>`_ is a tool that helps manage
machine learning experiments. It is a good idea to use it to track experiments
and to save models.

To use it, you need to create a directory where the experiments will be saved.
Then you need to set the environment variable ``MLFLOW_DIR`` to point to
that directory.

To start the mlflow server, run:

.. code-block:: bash

    mlflow server --backend-store-uri /path/to/mlflow_dir --host 127.0.0.1 --port 8080

and browse to the address `http://127.0.0.1:8080` to see the mlflow UI.
