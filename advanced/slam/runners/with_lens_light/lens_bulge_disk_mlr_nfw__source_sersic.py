import os

"""
__SLaM (Source, Light and Mass)__

Welcome to the SLaM pipeline runner, which loads a strong lens dataset and analyses it using a SLaM lens modeling 
pipeline. For a complete description of SLaM, checkout ? and ?.

__THIS RUNNER__

Using two source pipelines, a light pipeline and a mass pipeline we will fit a lens model where: 

    - The lens galaxy's *LightProfile*'s are fitted with an EllipticalSersic + EllipticalExponential, representing
      a bulge + disk model.
    - The lens galaxy's stellar *MassProfile* is fitted using the EllipticalSersic + EllipticalExponential of the 
      *LightProfile*, where it is converted to a stellar mass distribution via constant mass-to-light ratios.
    - The lens galaxy's nfw *MassProfile* is fitted with a SphericalNFW.
    - The source galaxy's *LightProfile* is fitted with an *Inversion*.

We'll use the SLaM pipelines:

'slam/with_lens_light/source/parametric/lens_bulge_disk_sie__source_sersic.py'.
'slam/with_lens_light/source/inversion/from_parametric/lens_light_sie__source_inversion.py'.
'slam/with_lens_light/light/bulge_disk/lens_bulge_disk_sie__source.py'.
'slam/with_lens_light/mass/light_dark/lens_light_mlr_nfw__source.py'.

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
dataset_name = "lens_bulge_disk_mlr_nfw__source_sersic"
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
    grid_inversion_class=al.GridInterpolate,
    pixel_scales_interp=0.1,
    inversion_pixel_limit=1500,
)

"""__PIPELINE SETUP + SETTINGS__

Advanced pipelines still use hyper settings, which customize the hyper-mode features and inclusion of a shear.
"""

hyper = al.slam.Hyper(
    hyper_galaxies=True, hyper_image_sky=False, hyper_background_noise=False
)

source = al.slam.Source(
    pixelization=al.pix.VoronoiBrightnessImage, regularization=al.reg.AdaptiveBrightness
)

light = al.slam.Light(
    align_bulge_disk_centre=True, align_bulge_disk_elliptical_comps=False
)

mass = al.slam.Mass()

slam = al.slam.SLaM(hyper=hyper, source=source, light=light, mass=mass)

"""We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!"""

from autolens_workspace.advanced.slam.pipelines.with_lens_light.source.parametric import (
    lens_bulge_disk_sie__source_sersic,
)
from autolens_workspace.advanced.slam.pipelines.with_lens_light.source.inversion.from_parametric import (
    lens_light_sie__source_inversion,
)

source__parametric = lens_bulge_disk_sie__source_sersic.make_pipeline(
    slam=slam, settings=settings, phase_folders=["slam", dataset_label, dataset_name]
)

source__inversion = lens_light_sie__source_inversion.make_pipeline(
    slam=slam, settings=settings, phase_folders=["slam", dataset_label, dataset_name]
)

from autolens_workspace.advanced.slam.pipelines.with_lens_light.light.bulge_disk import (
    lens_bulge_disk_sie__source,
)

light__bulge_disk = lens_bulge_disk_sie__source.make_pipeline(
    slam=slam, settings=settings, phase_folders=["slam", dataset_label, dataset_name]
)


from autolens_workspace.advanced.slam.pipelines.with_lens_light.mass.light_dark import (
    lens_light_mlr_nfw__source,
)

mass__mlr_nfw = lens_light_mlr_nfw__source.make_pipeline(
    slam=slam, settings=settings, phase_folders=["slam", dataset_label, dataset_name]
)


"""
__PIPELINE COMPOSITION AND RUN__

We finally add the pipelines above together, meaning they will run back-to-back, passing information from earlier 
phases to later phases.
"""

pipeline = source__parametric + source__inversion + light__bulge_disk + mass__mlr_nfw

pipeline.run(dataset=imaging, mask=mask)
