import os
import json

"""
This pipeline runner demonstrates how to use the dataset name and info in pipelines, so they are accessible in
the aggregator results. Checkout the preprocess tutorial
'autolens_workspace/preprocess/imaging/p6_info.ipynb' for a description of info.

I'll assume that you are familiar with how the beginner runners work, so if any code doesn't make sense familiarize
yourself with those first!
"""

""" AUTOFIT + CONFIG SETUP """

import autofit as af

workspace_path = "{}/../..".format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"
pixel_scales = 0.1

""" AUTOLENS + DATA SETUP """

import autolens as al
import autolens.plot as aplt

dataset_path = af.util.create_path(
    path=workspace_path, folders=["dataset", dataset_label, dataset_name]
)

# We load the info here, assuming it has been output to the dataset directory as a .json file as is performed in
# preprocess tutorial on info.

with open(f"{dataset_path}/info.json") as json_file:
    info = json.load(json_file)

# We now pass this into the imaging data when we load it from fits, along with the dataset name, so that they are
# stored during the analysis so the aggregator can access them.

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
    name=dataset_name,
)

# That is all we have to do! By passing this imaging to the pipeline.run() function, we're good to go.

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

setup = al.PipelineSetup(
    pixelization=al.pix.VoronoiMagnification,
    regularization=al.reg.Constant,
    no_shear=False,
)

from pipelines.imaging.no_lens_light import lens_sie__source_inversion

pipeline = lens_sie__source_inversion.make_pipeline(
    setup=setup, phase_folders=["features"]
)

pipeline.run(dataset=imaging, mask=mask, info=info)
