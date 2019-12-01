import os

# This pipeline runner demonstrates how to use the inner mask pipeline. If you haven't yet, you should read the example
# pipeline 'autolens_workspace/pipelines/features/inner_masking.py' for a description of how inner masking works.

# Most of this runner repeats the command described in the 'runner.'py' file. Therefore, to make it clear where the
# specific positions functionality is used, I have deleted all comments not related to that feature.

### AUTOFIT + CONFIG SETUP ###

import autofit as af

workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

config_path = workspace_path + "config"

af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"
pixel_scales = 0.1

### AUTOLENS + DATA SETUP ###

import autolens as al

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

imaging = al.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
)

al.plot.imaging.subplot(imaging=imaging)

# We simply import the inner masking pipeline and pass the inner_circular_mask_adii as an input parameter to specify
# how large we want the inner circular mask to be (which for the pipeline below, is only used in phase 1).

from pipelines.features import inner_masking

pipeline = inner_masking.make_pipeline(
    phase_folders=[dataset_label, dataset_name], inner_mask_radii=0.2
)

pipeline.run(dataset=imaging)
