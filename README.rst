PyAutoLens Workspace
====================

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/HEAD

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.02825/status.svg
   :target: https://doi.org/10.21105/joss.02825

|binder| |JOSS|

`Installation Guide <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautolens.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Binder <https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/master?filepath=introduction.ipynb>`_ |
`HowToLens <https://pyautolens.readthedocs.io/en/latest/howtolens/howtolens.html>`_

Welcome to the **PyAutoLens** Workspace. You can get started right away by going to the `autolens workspace
Binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/HEAD>`_.
Alternatively, you can get set up by following the installation guide on our `readthedocs <https://pyautolens.readthedocs.io/>`_.

Getting Started
---------------

If you haven't already, install `PyAutoLens via pip or conda <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_.

Next, clone the ``autolens workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autolens_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
   git clone https://github.com/Jammy2211/autolens_workspace --depth 1
   cd autolens_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py

Workspace Structure
-------------------

The workspace includes the following main directories:

- ``notebooks``: **PyAutoLens** examples written as Jupyter notebooks.
- ``scripts``: **PyAutoLens** examples written as Python scripts.
- ``config``: Configuration files which customize **PyAutoLens**'s behaviour.
- ``dataset``: Where data is stored, including example datasets distributed with **PyAutoLens**.
- ``output``: Where the **PyAutoLens** analysis and visualization are output.
- ``slam``: The Source, Light and Mass (SLaM) lens modeling pipelines, which are scripts for experienced users.

The examples in the ``notebooks`` and ``scripts`` folders are structured as follows:

- ``overview``: Examples giving an overview of **PyAutoLens**'s core features.
- ``howtolens``: Detailed step-by-step Jupyter notebook tutorials on how to use **PyAutoLens**.
- ``imaging``: Examples for analysing and simulating CCD imaging data.
- ``interferometer``: Examples for analysing and simulating interferometer datasets.
- ``group``: Group-scale lens modeling and simulations examples.
- ``cluster``: Cluster-scale lens modeling and simulation exampless.
- ``database``: Examples for using database tools which load libraries of model-fits to large datasets.
- ``plot``: An API reference guide for **PyAutoLens**'s plotting tools.
- ``misc``: Miscelaneous scripts for specific lens analysis.

In the ``imaging``, ``interferometer``, ``point_source``, ``group`` and ``cluster`` folders you'll find the following
packages:

- ``modeling``: Examples of how to fit a lens model to data via a non-linear search.
- ``chaining``: Advanced modeling scripts which chain together multiple non-linear searches.
- ``simulators``: Scripts for simulating realistic imaging and interferometer data of strong lenses.
- ``preprocess``: Tools to preprocess ``data`` before an analysis (e.g. convert units, create masks).

The ``chaining`` sections are for users familiar with **PyAutoLens** and contain:

- ``pipelines``: Example pipelines for modeling strong lenses using non-linear search chaining.
- ``hyper_mode``: Examples using hyper-mode, which adapts the lens model to the data being fitted.
- ``slam``: Example scripts that fit lens datasets using the SLaM pipelines.

Getting Started
---------------

We recommend new users begin with the example notebooks / scripts in the *overview* folder and the **HowToLens**
tutorials.

Workspace Version
-----------------

This version of the workspace is built and tested for using **PyAutoLens v2021.6.04.1**.

HowToLens
---------

Included with **PyAutoLens** is the ``HowToLens`` lecture series, which provides an introduction to strong gravitational
lens modeling with **PyAutoLens**. It can be found in the workspace & consists of 5 chapters:

- ``Introduction``: An introduction to strong gravitational lensing & **PyAutoLens**.
- ``Lens Modeling``: How to model strong lenses, including a primer on Bayesian non-linear analysis.
- ``Pipelines``: How to build model-fitting pipelines & tailor them to your own science case.
- ``Inversions``: How to perform pixelized reconstructions of the source-galaxy.
- ``Hyper-Mode``: How to use **PyAutoLens** advanced modeling features that adapt the model to the strong lens being analysed.


Contribution
------------
To make changes in the tutorial notebooks, please make changes in the the corresponding python files(.py) present in the
``scripts`` folder of each chapter. Please note that  marker ``# %%`` alternates between code cells and markdown cells.


Support
-------

Support for installation issues, help with lens modeling and using **PyAutoLens** is available by
`raising an issue on the autolens_workspace GitHub page <https://github.com/Jammy2211/autolens_workspace/issues>`_. or
joining the **PyAutoLens** `Slack channel <https://pyautolens.slack.com/>`_, where we also provide the latest updates on
**PyAutoLens**.

Slack is invitation-only, so if you'd like to join send an `email <https://github.com/Jammy2211>`_ requesting an
invite.
