PyAutoLens Workspace
====================

Welcome to the **PyAutoLens** Workspace. If you haven't already, you should install **PyAutoLens**, following the
instructions at `the PyAutoLens readthedocs <https://pyautolens.readthedocs.io/en/master/installation.html>`_.

Workspace Version
=================

This version of the workspace are built and tested for using **PyAutoLens v1.0.10**.

.. code-block:: python

    pip install autolens==1.0.10

Getting Started
===============

To get started checkout the `workspace tour on our readthedocs <https://pyautolens.readthedocs.io/en/latest/workspace.html>`_.

Workspace Contents
==================

The workspace includes the following:

- **Examples** - Illustrative scripts of the **PyAutoLens** interface, for examples on how to perform lensing
                 calculations, model lenses, etc.
- **Config** - Configuration files which customize **PyAutoLens**'s behaviour.
- **Dataset** - Where data is stored, including example datasets distributed with **PyAutoLens**.
- **HowToLens** - The **HowToLens** lecture series.
- **Output** - Where the **PyAutoLens** analysis and visualization are output.
- **Pipelines** - Example pipelines for modeling strong lenses.
- **Preprocess** - Tools to preprocess data before an analysis (e.g. convert units, create masks).
- **Runners** - Scripts for running a **PyAutoLens** pipeline.
- **Simulators** - Scripts for simulating strong lens datasets with **PyAutoLens**.

The **advanced** folder of the workspace includes:

- **Aggregator** - Manipulate large suites of modeling results via Jupyter notebooks, using **PyAutoFit**'s in-built
                   results database.

HowToLens
---------

Included with **PyAutoLens** is the **HowToLens** lecture series, which provides an introduction to strong gravitational
lens modeling with **PyAutoLens**. It can be found in the workspace & consists of 5 chapters:

- **Introduction** - An introduction to strong gravitational lensing & **PyAutolens**.
- **Lens Modeling** - How to model strong lenses, including a primer on Bayesian non-linear analysis.
- **Pipelines** - How to build model-fitting pipelines & tailor them to your own science case.
- **Inversions** - How to perform pixelized reconstructions of the source-galaxy.
- **Hyper-Mode** - How to use **PyAutoLens** advanced modeling features that adapt the model to the strong lens being analysed.

Issues, Help and Problems
=========================

If the installation below does not work or you have issues running scripts in the workspace, please post an issue on
the `issues section of the autolens_workspace <https://github.com/Jammy2211/autolens_workspace/issues>`_.

Support & Discussion
=====================

If you haven't already, go ahead and `email <https://github.com/Jammy2211>`_ me to get on our
`Slack channel <https://pyautolens.slack.com/>`_.
