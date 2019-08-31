# PyAutoLens Workspace

Welcome to the **PyAutoLens** Workspace. If you haven't already, you should install **PyAutoLens**, following the instructions at [PyAutoLens](https://github.com/Jammy2211/PyAutoLens).

# Workspace Version

This version of the workspace are built and tested for using **PyAutoLens v0.30.0** and **PyAutoFit v0.31.4** If you have any errors or issues using the workspace, try using these versions:

```
pip install autolens==0.30.0
pip install autofit==0.31.4
```

## Getting Started

To begin, I'd check out the 'introduction' in the quickstart folder. This will explain the best way to get started with
**PyAutoLens**.

# Workspace Contents

The workspace includes the following:

- **Aggregator** - Manipulate large suites of model-fits via Jupyter notebooks, using **PyAutoFit**'s in-built results database.
- **Config** - Configuration files which customize the **PyAutoLens** analysis.
- **Data** - Your data folder, including example data-sets distributed with **PyAutoLens**.
- **Quick Start** - A quick start guide, so you can begin modeling your own lens data within hours.
- **HowToLens** - The **HowToLens** lecture series.
- **Output** - Where the **PyAutoLens** analysis and visualization are output.
- **Pipelines** - Example pipelines for modeling strong lenses or to use a template for your own pipeline.
- **Plotting** - Scripts enabling customized figures and images.
- **Runners** - Scripts for running a **PyAutoLens** pipeline.
- **Tools** - Tools for simulating strong lens data, creating masks and using many other **PyAutoLens** features.

## Setup

The workspace is independent from the autolens install (e.g. the 'site-packages' folder), meaning you can edit workspace 
scripts and not worry about conflicts with new **PyAutoLens** installs.

Python therefore must know where the workspace is located so that it can import modules / scripts. This is done by 
setting the PYTHONPATH:
```
export PYTHONPATH=/path/to/autolens_workspace/
```

**PyAutoLens** additionally needs to know the default location of config files, which is done by setting the WORKSPACE.
Clone autolens workspace & set WORKSPACE enviroment variable:
```
export WORKSPACE=/path/to/autolens_workspace/
```

You can test everything is working by running the example pipeline runner in the autolens_workspace
```
python3 /path/to/autolens_workspace/runners/simple/runner__lens_sie__source_sersic.py
```

## Support & Discussion

If you haven't already, go ahead and [email](https://github.com/Jammy2211) me to get on our [Slack channel](https://pyautolens.slack.com/).