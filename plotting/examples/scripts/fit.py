import autofit as af
import autolens as al

import os

# In this example, we will load the residual map of a fit from a .fits file and plotters it using the function
# autolens.dataset_label.plotters.array_plotters.plot_array.

# We will use the residuals of a fit to slacs1430+4105, which comes from running the example pipeline
# 'workspacde/pipelines/examples/lens_sie__source_sersic_parametric.py.

# We have included the .fits dataset_label required for this example in the directory
# 'autolens_workspace/output/imaging/slacs1430+4105/pipeline_light_and_x1_source_parametric/phase_3_both/image/fits'.

# However, the complete set of optimizer results for the pipeline are not included, as the large file sizes prohibit
# distribution. Therefore, you may wish to run this pipeline now on slacs1430+4105 to generate your own results.

# We will customize the appearance of this figure to highlight the features of the residual map.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# First, lets setup the path to the .fits file of the residual map.
dataset_label = "imaging"
dataset_name = "slacs1430+4105"

# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/slacs1430+4105/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)
image_path = dataset_path + "image.fits"

pipeline_name = "pipeline_lens_sie__source_sersic_parametric"
phase_name = "phase_3_both"
result_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path,
    folder_names=["dataset", dataset_label, dataset_name, pipeline_name, phase_name],
)
residual_map_path = result_path + "/image/fits/fit_residual_map.fits"
chi_squared_map_path = result_path + "/image/fits/fit_chi_squared_map.fits"

# Now, lets load this arrays as a hyper arrays. A hyper arrays is an ordinary NumPy arrays, but it also includes a pixel
# scale which allows us to convert the axes of the arrays to arc-second coordinates.
image = al.array.from_fits(file_path=image_path, hdu=0, pixel_scales=0.03)

# Now, lets load this image as a hyper arrays. A hyper arrays is an ordinary NumPy arrays, but it also includes a pixel
# scale which allows us to convert the axes of the image to arc-second coordinates.
residual_map = al.array.from_fits(file_path=residual_map_path, hdu=0, pixel_scales=0.03)

# We can now use an arrays plotter to plotters the residual map.
al.plot.array(array=residual_map, title="SLACS1430+4105 Residual Map")

# A useful way to really dig into the residuals is to set upper and lower limits on the normalization of the colorbar.
al.plot.array(
    array=residual_map,
    title="SLACS1430+4105 Residual Map",
    norm_min=-0.02,
    norm_max=0.02,
)

# Or, alternatively, use a symmetric logarithmic colormap
al.plot.array(
    array=residual_map,
    title="SLACS1430+4105 Residual Map",
    norm="symmetric_log",
    linthresh=0.01,
    linscale=0.02,
)

# These tools are equally powerful ways to inspect the chi-squared map of a fit.
chi_squared_map = al.array.from_fits(
    file_path=chi_squared_map_path, hdu=0, pixel_scales=0.04
)
al.plot.array(array=chi_squared_map, title="SLACS1430+4105 Chi-Squared Map")
al.plot.array(
    array=chi_squared_map,
    title="SLACS1430+4105 Chi-Squared Map",
    norm_min=-10.0,
    norm_max=10.0,
)
al.plot.array(
    array=chi_squared_map,
    title="SLACS1430+4105 Chi-Squared Map",
    norm="symmetric_log",
    linthresh=0.01,
    linscale=0.02,
)

# We can also plotters the results of a fit using the fit itself. To do this, we have to make the pipeline and run it
# so as to load up all the results of the pipeline. We can then access the results of every phase.

from pipelines.simple import lens_sersic_sie__source_sersic

image_path = dataset_path + "/image.fits"
psf_path = dataset_path + "/psf.fits"
noise_map_path = dataset_path + "/noise_map.fits"

imaging = al.imaging.from_fits(
    image_path=image_path,
    psf_path=psf_path,
    noise_map_path=noise_map_path,
    pixel_scales=0.03,
)

pipeline = lens_sersic_sie__source_sersic.make_pipeline(
    phase_folders=[dataset_label, dataset_name], include_shear=True
)

# Now we run the pipeline on the dataset to get the result. If a mask was supplied to the pipeline when it was run, it is
# important the same mask is supplied in this run statement.

# The skip_optimizer boolean flag means that the non-linear searches will not run, and visualization will be skipped.
# This ensures the running of the pipeline is fast.

result = pipeline.run(dataset=imaging, skip_optimizer=True)

print(result)
