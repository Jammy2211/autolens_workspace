import os
import autofit as af

"""
__Aggregator: Pipeline Runner__

# This script fits the sample of three strong lenses simulated by the script 'autolens_workspace/aggregator/sample.py'
# using an advanced pipeline to illustrate aggregator functionality. If you are only used to using beginner or
# intermediate pipelines, you should still be able to understand the aggregator tutorials.

# we fit each lens with an _EllipticalPowerLaw_ _MassProfile_ and each source using a pixelized _Inversion_. The fit will use 3
# advanced pipelines which are added together to perform a 6 phase analysis, which will allow us to illustrate how the
# results of different pipelines and phases can be loaded using the aggregator.

# This script follows the scripts described in 'autolens_workspace/runners/advanced/' and the pipelines:

# 'autolens_workspace/pipelines/advanced/no_lens_light/source/parametric/mass_sie__source_sersic.py'
# 'autolens_workspace/pipelines/advanced/no_lens_light/source/inversion/from_parametric/lens_sie__source_inversion.py'
# 'autolens_workspace/pipelines/advanced/no_lens_light/mass/sie/lens_sie__source_inversion.py'

# If anything doesn't make sense check those scripts out for details!
"""

""" AUTOFIT + CONFIG SETUP """

# %%
"""Use the WORKSPACE environment variable to determine the path to the autolens workspace."""

# %%
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

"""Use this path to explicitly set the config path and output path."""
conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

""" AUTOLENS + DATA SETUP """

import autolens as al

"""Specify the dataset type, label and name, which we use to determine the path we load the data from."""

pixel_scales = 0.1

for dataset_name in [
    "mass_sie__source_sersic__0",
    "mass_sie__source_sersic__1",
    "mass_sie__source_sersic__2",
]:

    # Create the path where the dataset will be loaded from, which in this case is
    # '/autolens_workspace/aggregator/dataset/imaging/mass_sie__source_sersic'
    dataset_path = af.util.create_path(
        path=workspace_path, folders=["aggregator", "dataset", dataset_name]
    )

    """Using the dataset path, load the data (image, noise-map, PSF) as an _Imaging_ object from .fits files."""
    _Imaging_ = al.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        pixel_scales=pixel_scales,
    )

    """Next, we create the mask we'll fit this data-set with."""
    mask = al.Mask.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    ### PIPELINE SETUP ###

    # Advanced pipelines still use general setup, which customize the hyper-mode features and inclusion of a shear.

    hyper = al.SetupHyper(
        hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False
    )

    source = al.SLaMPipelineSource(
        pixelization=al.pix.VoronoiMagnification, regularization=al.reg.Constant
    )

    mass = al.SLaMPipelineMass(no_shear=False)

    setup = al.SLaM(setup_hyper=hyper, source=source, pipeline_mass=mass)

    # We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

    ### SOURCE ###

    from autolens_workspace.pipelines.advanced.no_lens_light.source.parametric import (
        mass_sie__source_sersic,
    )
    from autolens_workspace.pipelines.advanced.no_lens_light.source.inversion.from_parametric import (
        lens_sie__source_inversion,
    )

    from autolens_workspace.pipelines.advanced.no_lens_light.mass.sie import (
        lens_sie__source,
    )

    pipeline_source__parametric = mass_sie__source_sersic.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    pipeline_source__inversion = lens_sie__source_inversion.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    pipeline_mass__sie = lens_sie__source.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    ### PIPELINE COMPOSITION AND RUN ###

    # We finally add the pipelines above together, which means that they will run back-to-back, as usual passing
    # information throughout the analysis to later phases.

    pipeline = (
        pipeline_source__parametric + pipeline_source__inversion + pipeline_mass__sie
    )

    pipeline.run(dataset=imaging, mask=mask)

for dataset_name in [
    "mass_sie__source_sersic__0",
    "mass_sie__source_sersic__1",
    "mass_sie__source_sersic__2",
]:

    """
Create the path where the dataset will be loaded from, which in this case is
'/autolens_workspace/dataset/imaging/mass_sie__source_sersic'
"""
    dataset_path = af.util.create_path(
        path=workspace_path, folders=["dataset", dataset_name]
    )

    """Using the dataset path, load the data (image, noise-map, PSF) as an _Imaging_ object from .fits files."""
    _Imaging_ = al.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        pixel_scales=pixel_scales,
    )

    """Next, we create the mask we'll fit this data-set with."""
    mask = al.Mask.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    ### PIPELINE SETUP ###

    # Advanced pipelines still use general setup, which customize the hyper-mode features and inclusion of a shear.

    hyper = al.SetupHyper(
        hyper_galaxies=True, hyper_image_sky=False, hyper_background_noise=False
    )

    source = al.SLaMPipelineSource(
        pixelization=al.pix.VoronoiMagnification, regularization=al.reg.Constant
    )

    mass = al.SLaMPipelineMass(no_shear=False)

    setup = al.SLaM(setup_hyper=hyper, source=source, pipeline_mass=mass)

    # We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

    ### SOURCE ###

    from autolens_workspace.pipelines.advanced.no_lens_light.source.parametric import (
        mass_sie__source_sersic,
    )
    from autolens_workspace.pipelines.advanced.no_lens_light.source.inversion.from_parametric import (
        lens_sie__source_inversion,
    )

    from autolens_workspace.pipelines.advanced.no_lens_light.mass.sie import (
        lens_sie__source,
    )

    pipeline_source__parametric = mass_sie__source_sersic.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    pipeline_source__inversion = lens_sie__source_inversion.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    pipeline_mass__sie = lens_sie__source.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    ### PIPELINE COMPOSITION AND RUN ###

    # We finally add the pipelines above together, which means that they will run back-to-back, as usual passing
    # information throughout the analysis to later phases.

    pipeline = (
        pipeline_source__parametric + pipeline_source__inversion + pipeline_mass__sie
    )

    pipeline.run(dataset=imaging, mask=mask)

for dataset_name in [
    "mass_sie__source_sersic__0",
    "mass_sie__source_sersic__1",
    "mass_sie__source_sersic__2",
]:

    """
Create the path where the dataset will be loaded from, which in this case is
'/autolens_workspace/dataset/imaging/mass_sie__source_sersic'
"""
    dataset_path = af.util.create_path(
        path=workspace_path, folders=["dataset", dataset_name]
    )

    """Using the dataset path, load the data (image, noise-map, PSF) as an _Imaging_ object from .fits files."""
    _Imaging_ = al.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        pixel_scales=pixel_scales,
    )

    """Next, we create the mask we'll fit this data-set with."""
    mask = al.Mask.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    ### PIPELINE SETUP ###

    # Advanced pipelines still use general setup, which customize the hyper-mode features and inclusion of a shear.

    hyper = al.SetupHyper(
        hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False
    )

    source = al.SLaMPipelineSource(
        pixelization=al.pix.VoronoiMagnification,
        regularization=al.reg.Constant,
        no_shear=True,
    )

    mass = al.SLaMPipelineMass(no_shear=True)

    setup = al.SLaM(setup_hyper=hyper, source=source, pipeline_mass=mass)

    # We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

    ### SOURCE ###

    from autolens_workspace.pipelines.advanced.no_lens_light.source.parametric import (
        mass_sie__source_sersic,
    )
    from autolens_workspace.pipelines.advanced.no_lens_light.source.inversion.from_parametric import (
        lens_sie__source_inversion,
    )

    from autolens_workspace.pipelines.advanced.no_lens_light.mass.sie import (
        lens_sie__source,
    )

    pipeline_source__parametric = mass_sie__source_sersic.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    pipeline_source__inversion = lens_sie__source_inversion.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    pipeline_mass__sie = lens_sie__source.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    ### PIPELINE COMPOSITION AND RUN ###

    # We finally add the pipelines above together, which means that they will run back-to-back, as usual passing
    # information throughout the analysis to later phases.

    pipeline = (
        pipeline_source__parametric + pipeline_source__inversion + pipeline_mass__sie
    )

    pipeline.run(dataset=imaging, mask=mask)

for dataset_name in [
    "mass_sie__source_sersic__0",
    "mass_sie__source_sersic__1",
    "mass_sie__source_sersic__2",
]:

    """
Create the path where the dataset will be loaded from, which in this case is
'/autolens_workspace/dataset/imaging/mass_sie__source_sersic'
"""
    dataset_path = af.util.create_path(
        path=workspace_path, folders=["dataset", dataset_name]
    )

    """Using the dataset path, load the data (image, noise-map, PSF) as an _Imaging_ object from .fits files."""
    _Imaging_ = al.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        pixel_scales=pixel_scales,
    )

    """Next, we create the mask we'll fit this data-set with."""
    mask = al.Mask.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    ### PIPELINE SETUP ###

    # Advanced pipelines still use general setup, which customize the hyper-mode features and inclusion of a shear.

    hyper = al.SetupHyper(
        hyper_galaxies=True, hyper_image_sky=False, hyper_background_noise=False
    )

    source = al.SLaMPipelineSource(
        pixelization=al.pix.VoronoiMagnification,
        regularization=al.reg.Constant,
        no_shear=True,
    )

    mass = al.SLaMPipelineMass(no_shear=True)

    setup = al.SLaM(setup_hyper=hyper, source=source, pipeline_mass=mass)

    # We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

    ### SOURCE ###

    from autolens_workspace.pipelines.advanced.no_lens_light.source.parametric import (
        mass_sie__source_sersic,
    )
    from autolens_workspace.pipelines.advanced.no_lens_light.source.inversion.from_parametric import (
        lens_sie__source_inversion,
    )

    from autolens_workspace.pipelines.advanced.no_lens_light.mass.sie import (
        lens_sie__source,
    )

    pipeline_source__parametric = mass_sie__source_sersic.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    pipeline_source__inversion = lens_sie__source_inversion.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    pipeline_mass__sie = lens_sie__source.make_pipeline(
        setup=setup, folders=["aggregator", "advanced", dataset_name]
    )

    ### PIPELINE COMPOSITION AND RUN ###

    # We finally add the pipelines above together, which means that they will run back-to-back, as usual passing
    # information throughout the analysis to later phases.

    pipeline = (
        pipeline_source__parametric + pipeline_source__inversion + pipeline_mass__sie
    )

    pipeline.run(dataset=imaging, mask=mask)
