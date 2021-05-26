"""
Modeling: Model Maker
=====================

For cluster modeling, there can be tens or hundreds of lens and source galaxies. Manually writing the lens model of
each in the Python scripts we used to perform lens model becomes highly cumbersome, and it better for us to manage
the models separate and store them in .json files.

This script makes the model's that are fitted throughout the example script in the `cluster/modeling` package and
writes the model to a .json file, which is loaded in each script.

To write your own cluster lens model, you can either easily adapt this script or edit the .json files that are
produced.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import json
import autofit as af
import autolens as al

"""
The path where the models are output.
"""
model_path = path.join("scripts", "clusters", "modeling", "models")
model_file = path.join(model_path, "lens_x3__source_x1.json")

"""
__Lens_x3__Source_x1__

 - The cluster consists of three lens galaxies whose total mass distributions are `SphIsothermal` models.
 - A single source galaxy is observed whose `LightProfile` is an `EllSersic`.
"""

mass_0 = af.Model(al.mp.SphIsothermal)
mass_0.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
mass_0.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)

lens_0 = af.Model(al.Galaxy, redshift=0.5, mass=mass_0)

mass_1 = af.Model(al.mp.SphIsothermal)
mass_1.centre_0 = af.GaussianPrior(mean=3.5, sigma=0.5)
mass_1.centre_1 = af.GaussianPrior(mean=2.5, sigma=0.5)

lens_1 = af.Model(al.Galaxy, redshift=0.5, mass=mass_1)

mass_2 = af.Model(al.mp.SphIsothermal)
mass_2.centre_0 = af.GaussianPrior(mean=-4.4, sigma=0.5)
mass_2.centre_1 = af.GaussianPrior(mean=-5.0, sigma=0.5)

lens_2 = af.Model(al.Galaxy, redshift=0.5, mass=mass_2)

point_0 = af.Model(al.ps.PointFlux)
point_0.centre_0 = af.GaussianPrior(mean=0.0, sigma=3.0)
point_0.centre_1 = af.GaussianPrior(mean=0.0, sigma=3.0)

source_0 = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

model = af.Collection(
    galaxies=af.Collection(
        lens_0=lens_0, lens_1=lens_1, lens_2=lens_2, source_0=source_0
    )
)

with open(model_file, "w+") as f:
    json.dump(model.dict, f, indent=4)
