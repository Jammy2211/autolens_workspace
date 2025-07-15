"""
__Log Likelihood Function: Linear Light Profile__

!!!
THIS DOES NOT CURRENTLY WORK, LINEAR LIGHT PROFILES ARE NOT SUPPORTED FOR INTERFEROMETRY YET!!!!
!!!!

This script provides a step-by-step guide of the `log_likelihood_function` which is used to fit `Interferometer`
data with parametric linear light profiles (e.g. a Sersic bulge and Exponential disk).

A "linear light profile" is a variant of a standard light profile where the `intensity` parameter is solved for
via linear algebra every time the model is fitted to the data. This uses a process called an "inversion" and it
always computes the `intensity` values that give the best fit to the data (e.g. maximize the likelihood)
given the light profile's other parameters.

This script has the following aims:

 - To provide a resource that authors can include in papers, so that readers can understand the likelihood
 function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

Accompanying this script is the `contributor_guide.py` which provides URL's to every part of the source-code that
is illustrated in this guide. This gives contributors a sequential run through of what source-code functions, modules and
packages are called when the likelihood is evaluated.

__Prerequisites__

The likelihood function of linear light profiles builds on that used for standard parametric light profiles, therefore
you should read the `light_profile/log_likelihood_function.py` script before this script.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

# import matplotlib.pyplot as plt
# import numpy as np
# from pathlib import Path
#
# import autogalaxy as ag
# import autogalaxy.plot as aplt
