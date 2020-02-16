import os

# This script fits the sample of three strong lenses simulated by the script 'autolens_workspace/aggregator/sample.py'
# using a beginner pipeline to illustrate aggregator functionality.

# We will fit each lens with an SIE mass profile and each source using a pixelized inversion. The fit will use a
# beginner pipelines which performs a 3 phase analysis, which will allow us to illustrate how the results of different
# phases can be loaded using the aggregator.

# This script follows the scripts described in 'autolens_workspace/runners/beginner/' and the pipelines:

# 'autolens_workspace/pipelines/beginner/no_lens_light/lens_sie__source_inversion.py'

# If anything doesn't make sense check those scripts out for details!

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

### AUTOLENS + DATA SETUP ###

import autolens as al

# Specify the dataset label and name, which we use to determine the path we load the data from.
dataset_label = "aggregator_sample"
pixel_scales = 0.1

output_label = "aggregator_sample_beginner"

for dataset_name in [
    "lens_sie__source_sersic__0",
    "lens_sie__source_sersic__1",
    "lens_sie__source_sersic__2",
]:

    # Create the path where the dataset will be loaded from, which in this case is
    # '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
    dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=workspace_path,
        folder_names=["aggregator", "dataset", dataset_label, dataset_name],
    )

    # Using the dataset path, load the data (image, noise-map, PSF) as an imaging object from .fits files.
    imaging = al.imaging.from_fits(
        image_path=dataset_path + "image.fits",
        psf_path=dataset_path + "psf.fits",
        noise_map_path=dataset_path + "noise_map.fits",
        pixel_scales=pixel_scales,
    )

    # Next, we create the mask we'll fit this data-set with.
    mask = al.mask.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    ### PIPELINE SETUP ###

    # The'pipeline_setup' customize a pipeline's behaviour.

    # For pipelines which use an inversion, the pipeline source setup customize:

    # - The Pixelization used by the inversion of this pipeline.
    # - The Regularization scheme used by of this pipeline.

    source_setup = al.setup.Source(
        pixelization=al.pix.VoronoiMagnification, regularization=al.reg.Constant
    )

    # The pipeline mass setup determines whether there is no external shear in the mass model or not.

    mass_setup = al.setup.Mass(no_shear=False)

    setup = al.setup.Setup(source=source_setup, mass=mass_setup)

    ### PIPELINE SETUP + RUN ###

    # To run a pipeline we import it from the pipelines folder, make it and pass the lens data to its run function.

    # The 'phase_folders' below specify the path the pipeliine results are written to. Our output will go to the path
    # 'autolens_workspace/output/beginner/dataset_label/dataset_name/' or equivalently
    # 'autolens_workspace/output/beginner/imaging/lens_sie__source_sersic/'

    from pipelines.beginner.no_lens_light import lens_sie__source_inversion

    pipeline = lens_sie__source_inversion.make_pipeline(
        setup=setup, phase_folders=[output_label, dataset_name]
    )

    pipeline.run(dataset=imaging, mask=mask)
