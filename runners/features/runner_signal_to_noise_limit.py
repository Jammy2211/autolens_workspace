import autofit as af
from autolens.data import ccd
from autolens.data.plotters import ccd_plotters

import os

# This pipeline runner demonstrates how to use the binning up pipeline. If you haven't yet, you should read the example
# pipeline 'workspace/pipelines/features/binning_up.py' for a description of how binning up works.

# Most of this runner repeats the command described in the 'runner.'py' file. Therefore, to make it clear where the
# specific binning up functionality is used, I have deleted all comments not related to that feature.

workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

data_type = "example"
data_name = "lens_mass_and_x1_source"
pixel_scale = 0.1

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

ccd_data = ccd.load_ccd_data_from_fits(
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    pixel_scale=pixel_scale,
)

ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

# We simply import the binning up pipeline and pass the level of binning up we want as an input parameter (which
# for the pipeline below, is only used in phase 1).

from workspace.pipelines.features import signal_to_noise_limit

pipeline = signal_to_noise_limit.make_pipeline(
    phase_folders=[data_type, data_name], signal_to_noise_limit=20.0,
)

pipeline.run(data=ccd_data)
