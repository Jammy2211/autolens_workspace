import os

# This pipeline runner demonstrates how to use the binning up pipeline. If you haven't yet, you should read the example
# pipeline 'workspace/pipelines/features/binning_up.py' for a description of how binning up works.

# Most of this runner repeats the command described in the 'runner.'py' file. Therefore, to make it clear where the
# specific binning up functionality is used, I have deleted all comments not related to that feature.

### AUTOFIT + CONFIG SETUP ###

import autofit as af

workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

config_path = workspace_path + "config"

af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

data_type = "example"
data_name = "lens_sie__source_sersic"
pixel_scale = 0.1

### AUTOLENS + DATA SETUP ###

from autolens.array import mask as msk
from autolens.data.instrument import abstract_data
from autolens.data.instrument import ccd
from autolens.data.plotters import al.ccd_plotters
from autolens.pipeline import pipeline as pl

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

ccd_data = al.load_ccd_data_from_fits(
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    pixel_scale=pixel_scale,
)

al.ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

# We simply import the binning up pipeline and pass the level of binning up we want as an input parameter (which
# for the pipeline below, is only used in phase 1).

from autolens_workspace.pipelines.features import signal_to_noise_limit

pipeline = signal_to_noise_limit.make_pipeline(
    phase_folders=[data_type, data_name], signal_to_noise_limit=20.0
)

pipeline.run(data=ccd_data)
