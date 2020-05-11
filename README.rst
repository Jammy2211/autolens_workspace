PyAutoLens Workspace
====================

Welcome to the **PyAutoLens** Workspace. If you haven't already, you should install **PyAutoLens**, following the
instructions at `the PyAutoLens readthedocs <https://pyautolens.readthedocs.io/en/master/installation.html>`_.

Workspace Version
=================

This version of the workspace are built and tested for using **PyAutoLens v0.46.2**.

.. code-block:: python

    pip install autolens==0.46.2

Getting Started
===============

To begin, check out the 'introduction' in the quickstart folder. This will explain the best way to get started with
**PyAutoLens**.

Workspace Contents
==================

The workspace includes the following:

- **Aggregator** - Manipulate large suites of modeling results via Jupyter notebooks, using **PyAutoFit**'s in-built results database.
- **API** - Illustrative scripts of the **PyAutoLens** interface, for examples on how to make plots, peform lensing calculations, etc.
- **Config** - Configuration files which customize **PyAutoLens**'s behaviour.
- **Dataset** - Where data is stored, including example datasets distributed with **PyAutoLens**.
- **HowToLens** - The **HowToLens** lecture series.
- **Output** - Where the **PyAutoLens** analysis and visualization are output.
- **Pipelines** - Example pipelines for modeling strong lenses.
- **Preprocess** - Tools to preprocess data before an analysis (e.g. convert units, create masks).
- **Quick Start** - A quick start guide, so you can begin modeling your lenses within hours.
- **Runners** - Scripts for running a **PyAutoLens** pipeline.
- **Simulators** - Scripts for simulating strong lens datasets with **PyAutoLens**.

Issues, Help and Problems
=========================

If the installation below does not work or you have issues running scripts in the workspace, please post an issue on
the `issues section of the autolens_workspace <https://github.com/Jammy2211/autolens_workspace/issues>`_.

Setup
=====

The workspace is independent from the autolens install (e.g. the 'site-packages' folder), meaning you can edit
workspace scripts and not worry about conflicts with new **PyAutoLens** installs.

Python therefore must know where the workspace is located so that it can import modules / scripts. This is done by 
setting the PYTHONPATH:

.. code-block:: python

    export PYTHONPATH=/path/to/autolens_workspace/

**PyAutoLens** additionally needs to know the default location of config files, which is done by setting the WORKSPACE.
Clone autolens workspace & set WORKSPACE enviroment variable:

.. code-block:: python

    export WORKSPACE=/path/to/autolens_workspace/

Matplotlib uses the default backend on your computer, as set in the config file 
autolens_workspace/config/visualize/general.ini:
 
.. code-block:: python

    [general]
    backend = default

There have been reports that using the default backend causes crashes when running the test script below (either the 
code crashes without a error or your computer restarts). If this happens, change the config's backend until the test
works (TKAgg has worked on Linux machines, Qt5Agg has worked on new MACs). For example:

.. code-block:: python

    [general]
    backend = TKAgg

You can test everything is working by running the example pipeline runner in the autolens_workspace

.. code-block:: python

    python3 /path/to/autolens_workspace/runners/beginner/no_lens_light/lens_sie__source_inversion.py

 Support & Discussion
=====================

If you haven't already, go ahead and `email <https://github.com/Jammy2211>`_ me to get on our
`Slack channel <https://pyautolens.slack.com/>`_.
