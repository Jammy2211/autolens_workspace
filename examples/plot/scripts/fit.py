# import autofit as af
# import autolens as al
# import autolens.plot as aplt
#
# # %%
# """
# In this example, we will load the residual map of a fit from a .fits file and plotters it using the function
# autolens.dataset_label.plotters.plotters.plot_array.
#
# We will use the residuals of a fit to slacs1430+4105, which comes from running the example pipeline
# 'workspacde/pipelines/examples/lens_sie__source_sersic_parametric.py.
#
# We have included the .fits dataset_label required for this example in the directory
# 'autolens_workspace/output/imaging/slacs1430+4105/pipeline_light_and_x1_source_parametric/phase_3_both/image/fits'.
#
# However, the complete set of search results for the pipeline are not included, as the large file sizes prohibit
# distribution. Therefore, you may wish to run this pipeline now on slacs1430+4105 to generate your own results.
#
# We will customize the appearance of this figure to highlight the features of the residual map.
#
# Setup the path to the autolens_workspace.
# """
#
# # %%
# workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace"
#
# # %%
# """
# First, lets setup the path to the .fits file of the residual map.
# """
#
# # %%
# dataset_label = "slacs"
# dataset_name = "slacs1430+4105"
#
# # %%
# """
# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/slacs1430+4105/'
# """
#
# # %%
# dataset_path = af.util.create_path(
#     path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
# )
# image_path = f"{dataset_path}/image.fits"
#
# pipeline_name = "pipeline_lens_sie__source_sersic_parametric"
# phase_name = "phase_3_both"
# result_path = af.util.create_path(
#     path=workspace_path,
#     folder_names=["dataset", dataset_label, dataset_name, pipeline_name, phase_name],
# )
# residual_map_path = result_path + "/image/fits/fit_residual_map.fits"
# chi_squared_map_path = result_path + "/image/fits/fit_chi_squared_map.fits"
#
# # %%
# """
# Now, lets load this arrays as a hyper arrays. A hyper arrays is an ordinary NumPy arrays, but it also includes a pixel
# scale which allows us to convert the axes of the arrays to arc-second coordinates.
# """
#
# # %%
# image = al.Array.from_fits(file_path=image_path, hdu=0, pixel_scales=0.03)
#
# # %%
# """
# Now, lets load this image as a hyper arrays. A hyper arrays is an ordinary NumPy arrays, but it also includes a pixel
# scale which allows us to convert the axes of the image to arc-second coordinates.
# """
#
# # %%
# residual_map = al.Array.from_fits(file_path=residual_map_path, hdu=0, pixel_scales=0.03)
#
# # %%
# """
# We can now use an arrays plotter to plotters the residual map.
# """
#
# # %%
# plotter = aplt.Plotter(labels=aplt.Labels(title="SLACS1430+4105 Residual Map"))
#
# aplt.Array(array=residual_map, plotter=plotter)
#
# # %%
# """
# A useful way to really dig into the residuals is to set upper and lower limits on the normalization of the colorbar.
# """
#
# # %%
# plotter = aplt.Plotter(
#     labels=aplt.Labels(title="SLACS1430+4105 Residual Map"),
#     cmap=aplt.ColorMap(norm_min=-0.02, norm_max=0.02),
# )
#
# aplt.Array(array=residual_map, plotter=plotter)
#
# # %%
# """
# Or, alternatively, use a symmetric logarithmic colormap
# """
#
# # %%
# plotter = aplt.Plotter(
#     labels=aplt.Labels(title="SLACS1430+4105 Residual Map"),
#     cmap=aplt.ColorMap(norm="symmetric_log", linthresh=0.02, linscale=0.02),
# )
#
# aplt.Array(array=residual_map, plotter=plotter)
#
# # %%
# """
# These tools are equally powerful ways to inspect the chi-squared map of a fit.
# """
#
# # %%
# chi_squared_map = al.Array.from_fits(
#     file_path=chi_squared_map_path, hdu=0, pixel_scales=0.04
# )
#
# plotter = aplt.Plotter(labels=aplt.Labels(title="SLACS1430+4105 Chi-Squared Map"))
#
# aplt.Array(array=chi_squared_map, plotter=plotter)
#
# plotter = aplt.Plotter(
#     labels=aplt.Labels(title="SLACS1430+4105 Chi-Squared Map"),
#     cmap=aplt.ColorMap(norm_min=-10.0, norm_max=10.0),
# )
# aplt.Array(array=chi_squared_map, plotter=plotter)
#
# plotter = aplt.Plotter(
#     labels=aplt.Labels(title="SLACS1430+4105 Chi-Squared Map"),
#     cmap=aplt.ColorMap(norm="symmetric_log", linthresh=0.01, linscale=0.02),
# )
# aplt.Array(array=chi_squared_map, plotter=plotter)
