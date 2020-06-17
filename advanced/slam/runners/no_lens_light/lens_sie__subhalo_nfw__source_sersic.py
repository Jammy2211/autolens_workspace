import os

"""
__SLaM (Source, Light and Mass)__

Welcome to the SLaM pipeline runner, which loads a strong lens dataset and analyses it using a SLaM lens modeling 
pipeline. For a complete description of SLaM, checkout ? and ?.

__THIS RUNNER__

Using 1 source pipeline, a mass pipeline and a subhalo pipeline we will fit a lens model where: 

    - The lens galaxy's light is omitted from the data and model.
    - The lens galaxy's *MassProfile* is fitted with an *EllipticalIsothermal*.
    - A dark matter subhalo's within the lens galaxy is fitted with a *SphericalNFWMCRLudLow*.
    - The source galaxy is fitted with an *EllipticalSersic*.

We'll use the SLaM pipelines:

    'slam/no_lens_light/source/parametric/lens_bulge_disk_sie__source_sersic.py'.
    'slam/no_lens_light/mass/sie/lens_power_law__source.py'.
    'slam/no_lens_light/subhalo/lens_mass__subhalo_nfw__source.py'.

Check them out now for a detailed description of the analysis!
"""

""" AUTOFIT + CONFIG SETUP """

from autoconf import conf
import autofit as af

"""Setup the path to the autolens_workspace, using a relative directory name."""
workspace_path = "{}/../../../..".format(os.path.dirname(os.path.realpath(__file__)))

"""Use this path to explicitly set the config path and output path."""
conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

""" AUTOLENS + DATA SETUP """
import autolens as al
import autolens.plot as aplt

"""Specify the dataset label and name, which we use to determine the path we load the data from."""
dataset_label = "imaging"
dataset_name = "lens_sie__subhalo_nfw__source_sersic__low_res"
pixel_scales = 0.1

"""
Create the path where the dataset will be loaded from, which in this case is
'/autolens_workspace/dataset/imaging/lens_bulge_disk_mlr_nfw__source_sersic'
"""

dataset_path = af.util.create_path(
    path=workspace_path, folders=["dataset", dataset_label, dataset_name]
)

"""Using the dataset path, load the data (image, noise map, PSF) as an imaging object from .fits files."""
imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
)

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

settings = al.PhaseSettingsImaging(
    grid_class=al.Grid,
    sub_size=2,
)

"""
__PIPELINE SLaM SETUP__

Advanced pipelines still use hyper settings, which customize the hyper-mode features and inclusion of a shear.
"""

hyper = al.slam.Hyper(
    hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False
)

source = al.slam.Source(no_shear=True)

mass = al.slam.Mass(no_shear=True)

slam = al.slam.SLaM(hyper=hyper, source=source, mass=mass)

"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!
"""

from autolens_workspace.advanced.slam.pipelines.no_lens_light.source.parametric import (
    lens_sie__source_sersic,
)

source__parametric = lens_sie__source_sersic.make_pipeline(
    slam=slam, settings=settings, phase_folders=["slam", dataset_label]
)

from autolens_workspace.advanced.slam.pipelines.no_lens_light.mass.sie import (
    lens_sie__source,
)

mass__sie = lens_sie__source.make_pipeline(
    slam=slam, settings=settings, phase_folders=["slam", dataset_label]
)

from autolens_workspace.advanced.slam.pipelines.no_lens_light.subhalo import (
    lens_mass__subhalo_nfw__source,
)

subhalo__nfw = lens_mass__subhalo_nfw__source.make_pipeline(
    slam=slam,
    settings=settings,
    phase_folders=["slam", dataset_label],
    grid_size=2,
)

"""
__PIPELINE COMPOSITION__

We finally add the pipelines above together, meaning they will run back-to-back, passing information from earlier 
phases to later phases.
"""

pipeline = source__parametric + mass__sie + subhalo__nfw

# %%
"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""

pipeline.run(dataset=imaging, mask=mask)
