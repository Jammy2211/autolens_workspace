### MODEL YOUR DATA ###

# Its time to model your own lens data! This is straight forward - just adapt the runner script you ran above to
# load your own CCD imaging data instead of the example data we ran previously.
#
# First, we need to make sure the data conforms to the PyAutoLens inputs. This requires that:
#
# 1) The image is a small (e.g. 501 x 501) cut-out of the strong lens, centred on the lens galaxy.
# 2) The image and noise-map are in electrons per second.
# 3) The PSF is cut-out to an odd-sized kernel with a reasonably small kernel size (e.g. 21x21).

# PyAutoLens has built-in tools to convert your data to these requirements, checkout the scripts in
# 'autolens_workspace/tools/loading_and_preparing_data'.


# Your data probably won't have a custom mask ready in a 'mask.fits' file. You have two options:

# 1) Use a large circular mask by adding the line
#
#     mask = msk.Mask.circular(
#          shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=3.0)

# 2) Creating your own custom mask, using the script 'autolens_workspace/tools/example/mask_maker.py'


# If your data doesn't contain the lens galaxy's light (this is often the case for radio / sub-mm imaging of strong
# lenses, where only the source galaxy is visible) then you should use the 'runner__lens_sie__source_inversion.py' script
# instead.

### POSITIONS ###

# We can also manually specify a set of image-pixels which correspond to the multiple images of the source-galaxy(s).
# During the analysis, PyAutoLens will first check that these pixels trace within a specified arc-second threshold of
# one another (which is controlled by the 'position_threshold' input into a phase). This
# provides two benefits:

# 1) The analysis runs faster, as the non-linear search avoids searching regions of parameter space where the
#    mass-model is clearly not accurate.

# 2) By removing these solutions, a global-maximum solution may be reached instead of a local-maxima. This is because
#    removing the incorrect mass models makes the non-linear parameter space less complex.

# For setting up an image with positions, checkout 'autolens_workspace/tools/data_making/positions_maker.py'. To see
# how positions are used by a runner and pipeline, checkout 'pipelines/features/position_thresholding.py' and
# 'runners/features/runner_positions.py'

### INVERSIONS ###

# So far, all the models we've run have used analytic light profiles to fit for the source galaxy (e.g. Sersic's). However,
# most strongly lensed sources are more complex then these simplistic symmetric light profiles, and our lens modeling
# benefits from reconstructing the source using a more general approach, called an 'inversion'. This basically
# reconstructs the source's light using a pixelized-grid.

# To use an inversion, checkout the pipeline 'pipelines/simple/lens_sersic_sie__source_inversion.py'
# and the runner 'runners/simple/runner__lens_light_mass__source_inversion.py'. By default, these pipelines use
# an adaptive pixelization whereby the inversion's pixels adapt to the surface brightness of the source.

# I would strongly recommend you only use inversions in conjuction with the positions feature described above.
# Inversions run a high risk of going to incorrect solutions if this feature is not used!
#
# Chapter 4 of the howtolens lecture series give a full description of how inversions work.

### FEATURES / ADVANCED / HYPER FUNCTIONALITY ###

# Once you've modeled some of your own lenses and feel more familar with PyAutoLens, You should check out the following
# places in the autolens_workspace:

# 1) 'pipelines/features' and 'runners/features'. These describe pipeline features that customize an analysis of
#    a strong lens, for example by binning up the data to a coarser resolution of performing ray-tracing on a higher
#    resolution 'sub-grid'.

# 2) 'pipelines/advanced' and 'runners/advanced'. These advanced pipelines and runners allow for a much broader range
#    of complex lens models to be fitted, and allow for different pipelines to be hyper_combined together so that the early
#    phases of a pipeline can be reused when fitting more complex models later on.

# 3) 'pipelines/hyper_galaxy' and 'runners/hyper_galaxy'. Hyper functionality is where PyAutoLens adapt the model fit to the data
#     its fitting. This includes fitting for the background sky subtracton in the image and scaling the noise-map to
#     prevent over-fitting small portions of the data. Only once you're confident with PyAutoLens would I
#    recommend that you start experimenting with this functionality!

### FIN. ###

# And with that, we've completed the PyAutoLens quick-start. You should be well equipped to begin modeling your own
# lenses and use a lot of PyAutoLens features. However, there is still a lot we've not had time to cover - so as I've
# said before, if you find yourself unsure *exactly* what PyAutoLens is doing, or don't quite know how to fit the
# lens model you want, you should checkout the howtolens lecture series for a complete description of how to
# use PyAutoLens.
