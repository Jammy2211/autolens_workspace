# """
# SLaM (Source, Light and Mass): Mass Total + Subhalo NFW + Source Parametric Sensitivity Mapping
# ===============================================================================================
#
# SLaM pipelines break the analysis of 'galaxy-scale' strong lenses down into multiple pipelines which focus on modeling
# a specific aspect of the strong lens, first the Source, then the (lens) Light and finally the Mass. Each of these
# pipelines has it own inputs which customize the model and analysis in that pipeline.
#
# The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
# uses a linear parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS TOTAL PIPELINE.
#
# Using a SOURCE LP PIPELINE, MASS TOTAL PIPELINE and SUBHALO PIPELINE this SLaM script fits `Interferometer` of a strong lens
# system, where in the final model:
#
#  - The lens galaxy's light is omitted from the data and model.
#  - The lens galaxy's total mass distribution is an `Isothermal`.
#  - The source galaxy is an `Sersic`.
#
# It ends by performing sensitivity mapping of the data using the above model, so as to determine where in the data
# subhalos of a given mass could have been detected if present.
#
# This modeling script uses the SLaM pipelines:
#
#  `source_lp`
#  `mass_total`
#  `subhalo/sensitivity_mapping`
#
# Check them out for a detailed description of the analysis!
# """
# # %matplotlib inline
# # from pyprojroot import here
# # workspace_path = str(here())
# # %cd $workspace_path
# # print(f"Working Directory has been set to `{workspace_path}`")
#
# import os
# import sys
# from os import path
# import autofit as af
# import autolens as al
# import autolens.plot as aplt
#
# sys.path.insert(0, os.getcwd())
# import slam
#
# """
# __Dataset + Masking__
#
# Load the `Interferometer` data, define the visibility and real-space masks and plot them.
# """
# real_space_mask = al.Mask2D.circular(
#     shape_native=(151, 151), pixel_scales=0.05, radius=3.0
# )
#
# dataset_name = "dark_matter_subhalo"
# dataset_path = path.join("dataset", "interferometer", dataset_name)
#
# dataset = al.Interferometer.from_fits(
#     data_path=path.join(dataset_path, "data.fits"),
#     noise_map_path=path.join(dataset_path, "noise_map.fits"),
#     uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
#     real_space_mask=real_space_mask,
# )
# dataset = dataset.apply_settings(
#     settings=al.SettingsInterferometer(transformer_class=al.TransformerDFT)
# )
#
# dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
# dataset_plotter.subplot_dataset()
# dataset_plotter.subplot_dirty_images()
#
# """
# __Settings AutoFit__
#
# The settings of autofit, which controls the output paths, parallelization, database use, etc.
# """
# settings_search = af.SettingsSearch(
#     path_prefix=path.join("interferometer", "slam"),
#     unique_tag=dataset_name,
#     info=None,
#     number_of_cores=2,
#     session=None,
# )
#
# """
# __Redshifts__
#
# The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g.
# from arc-seconds to kiloparsecs, masses to solar masses, etc.).
# """
# redshift_lens = 0.5
# redshift_source = 1.0
#
# """
# __SOURCE LP PIPELINE__
#
# The SOURCE LP PIPELINE uses one search to initialize a robust model for the source galaxy's light, which in
# this example:
#
#  - Uses a linear parametric `Sersic` bulge for the source's light (omitting a disk / envelope).
#  - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.
#
#  __Settings__:
#
#  - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
# """
# analysis = al.AnalysisInterferometer(dataset=dataset)
#
# source_lp_result = slam.source_lp.run(
#     settings_search=settings_search,
#     analysis=analysis,
#     lens_bulge=None,
#     lens_disk=None,
#     mass=af.Model(al.mp.Isothermal),
#     shear=af.Model(al.mp.ExternalShear),
#     source_bulge=af.Model(al.lp_linear.SersicCore),
#     mass_centre=(0.0, 0.0),
#     redshift_lens=0.5,
#     redshift_source=1.0,
# )
#
# """
# __MASS TOTAL PIPELINE__
#
# The MASS TOTAL PIPELINE uses one search to fits a complex lens mass model to a high level of accuracy,
# using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors. In this example it:
#
#  - Uses an `PowerLaw` model for the lens's total mass distribution [The centre if unfixed from (0.0, 0.0)].
#  - Uses the `Sersic` model representing a bulge for the source's light.
#  - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS TOTAL PIPELINE.
# """
# analysis = al.AnalysisInterferometer(dataset=dataset)
#
# mass_results = slam.mass_total.run(
#     settings_search=settings_search,
#     analysis=analysis,
#     source_results=source_lp_result,
#     light_results=None,
#     mass=af.Model(al.mp.PowerLaw),
# )
#
# """
# __SUBHALO PIPELINE (sensitivity mapping)__
#
# The SUBHALO PIPELINE (sensitivity mapping) performs sensitivity mapping of the data using the lens model
# fitted above, so as to determine where subhalos of what mass could be detected in the data. A full description of
# Sensitivity mapping if given in the SLaM pipeline script `slam/subhalo/sensitivity_imaging.py`.
#
# Each model-fit performed by sensitivity mapping creates a new instance of an `Analysis` class, which contains the
# data simulated by the `simulate_cls` for that model. This requires us to write a wrapper around the
# PyAutoLens `AnalysisInterferometer` class.
# """
#
#
# class AnalysisInterferometerSensitivity(al.AnalysisInterferometer):
#     def __init__(self, dataset):
#         super().__init__(dataset=dataset)
#
#         self.adapt_galaxy_name_image_dict = (
#             mass_results.last.adapt_galaxy_name_image_dict
#         )
#         self.adapt_model_image = mass_results.last.adapt_model_image
#
#
# subhalo_results = slam.subhalo.sensitivity_mapping_interferometer(
#     settings_search=settings_search,
#     analysis_cls=AnalysisInterferometerSensitivity,
#     uv_wavelengths=dataset.uv_wavelengths,
#     real_space_mask=real_space_mask,
#     mass_result=mass_result,
#     subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
#     grid_dimension_arcsec=3.0,
#     number_of_steps=2,
# )
#
# """
# Finish.
# """
