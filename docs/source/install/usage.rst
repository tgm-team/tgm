Usage
=====

.. _installation:

Installation
------------

To use openDG, first install it using pip:

.. code-block:: console

   (.venv) $ pip install opendg

Creating recipes
----------------

To load a temporal graph from an edgelist,
you can use the ``opendg.CTDG(data)`` function:

.. autofunction:: opendg.CTDG(data)

For example:

>>> import opendg
>>> opendg.CTDG(data)

.. The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
.. or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
.. will raise an exception.

.. .. autoexception:: lumache.InvalidKindError

.. For example:

.. >>> import lumache
.. >>> lumache.get_random_ingredients()
.. ['shells', 'gorgonzola', 'parsley']

