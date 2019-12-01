import os

# This pipeline runner demonstrates how to use the sub_gridding pipeline. If you haven't yet, you should read the example
# pipeline 'autolens_workspace/pipelines/features/sub_gridding.py' for a description of how sub-gridding works.

# Most of this runner repeats the command described in the 'runner.'py' file. Therefore, to make it clear where the
# specific sub gridding functionality is used, I have deleted all comments not related to that feature.

### AUTOFIT + CONFIG SETUP ###

import autofit as af

workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
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

# We simply import the sub_gridding pipeline and pass the sub-grid size we want as an input parameter (which
# for the pipeline below, is only used in phase 2).

from pipelines.features import sub_gridding

pipeline = sub_gridding.make_pipeline(
    phase_folders=[dataset_label, dataset_name], sub_size=4
)

pipeline.run(dataset=imaging)
