import os
import json

# This pipeline runner demonstrates how to use the dataset name and metadata in pipelines, so they are accessible in
# the aggregator results. Checkout the preprocess tutorial
# 'autolens_workspace/preprocess/imaging/p6_metadata.ipynb' for a description of metadata.

# I'll assume that you are familiar with how the beginner runners work, so if any code doesn't make sense familiarize
# yourself with those first!

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
import autolens.plot as aplt

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# We load the metadata here, assuming it has been output to the dataset directory as a .json file as is performed in
# preprocess tutorial on metadata.

with open(dataset_path + "metadata.json") as json_file:
    metadata = json.load(json_file)

# We now pass this into the imaging data when we load it from fits, along with the dataset name, so that they are
# stored during the analysis so the aggregator can access them.

imaging = al.Imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
    name=dataset_name,
    metadata=metadata,
)

# That is all we have to do! By passing this imaging to the pipeline.run() function, we're good to go.

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

source_setup = al.setup.Source()
mass_setup = al.setup.Mass()
setup = al.setup.Setup(source=source_setup, mass=mass_setup)

from pipelines.beginner.no_lens_light import lens_sie__source_inversion

pipeline = lens_sie__source_inversion.make_pipeline(
    setup=setup, phase_folders=["features"]
)

pipeline.run(dataset=imaging, mask=mask)
