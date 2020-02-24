### MODEL YOUR DATA ###

# Its time to model your own lens dataset! This is straight forward - just adapt the runner script you ran above to
# load your own imaging dataset instead of the example simulator we ran previously.
#
# First, we need to make sure the dataset conforms to the PyAutoLens inputs. This requires that:
#
# 1) The image is a small (e.g. 501 x 501) cut-out of the strong lens, centred on the lens galaxy.
# 2) The image and noise-map are in electrons per second.
# 3) The PSF is cut-out to an odd-sized kernel with a reasonably small kernel size (e.g. 21x21).

# PyAutoLens has built-in tools to convert your dataset to these requirements, checkout the scripts in
# 'autolens_workspace/tools/loading_and_preparing_data'.


# Your dataset probably won't have a custom mask ready in a 'mask.fits' file. You have two options:

# 1) Use a large circular mask by adding the line
#
#     mask = al.mask.circular(
#          shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0)

# 2) Creating your own custom mask, using the script 'autolens_workspace/tools/imaging/mask_maker.py'


# If your dataset doesn't contain the lens galaxy's light (this is often the case for radio / sub-mm imaging of strong
# lenses, where only the source galaxy is visible) then you should use the 'lens_sie__source_inversion.py'
# instead.

### POSITIONS ###

# We can manually specify a set of image-pixels that correspond to the source galaxy's multiple images. PyAutoLens will
# check that for a lens model these pixels trace within a specified arc-second threshold of one another (which is
# controlled by the 'position_threshold' input into a phase). This provides two benefits:

# 1) The analysis runs faster, as the non-linear search avoids searching regions of parameter space where the
#    mass-model is not accurate.

# 2) By removing these solutions the correct lens model may be found instead of an incorrect solution, because the
#    non-linear search samples fewer incorrect solutions.

# For setting up an image with positions, checkout 'autolens_workspace/tools/data_making/positions_maker.py'. To see
# how positions are used by a runner and pipeline, checkout 'pipelines/features/position_thresholding.py' and
# 'runners/features/position_thresholding.py'

### INVERSIONS ###

# The models run in the quick-start tutorial use analytic light profiles for the source galaxy (e.g. Sersic's). Strong
# lens source morphologies are often more, meaning lens modeling can benefit from reconstructing its light on a pixel
# grid using an 'inversion'.

# To use an inversion, checkout the pipeline 'pipelines/simple/lens_sersic_sie__source_inversion.py' and the runner
# 'runners/simple/lens_light_mass__source_inversion.py'. By default, these pipelines use an adaptive pixelization
# where the pixels adapt to the mass model's magnification.

# I advise that you use inversions in conjuction with the positions feature described above. Inversions run a high risk
# of going to incorrect solutions if this feature is not used!
#
# Chapter 4 of the howtolens lecture series give a full description of how inversions work.

### FEATURES / ADVANCED / HYPER FUNCTIONALITY ###

# Once you've modeled some of your own lenses and are familar with PyAutoLens, You should check out the following folders
# in the autolens_workspace:

# 1) 'pipelines/features' and 'runners/features'. These describe pipeline features that customize an analysis of
#    a strong lens, for example by binning up the dataset to a coarser resolution or performing ray-tracing on a higher
#    resolution 'sub-grid'.

# 2) 'pipelines/advanced' and 'runners/advanced'. Advanced pipelines allow a broader range of complex lens models to be
#    fitted and allow for pipelines to be combined with one another, so the early phases of a pipeline can be reused
#    when fitting different lens model in later phases.

# 3) 'pipelines/hyper' and 'runners/hyper'. Hyper functionality adapts the fit to the dataset. Only once you're confident
#    with PyAutoLens would I recommend that you start experimenting with this functionality!

### FIN. ###

# And with that, we've completed the PyAutoLens quick-start.

# You are nowl equipped to begin modeling your own lenses with PyAutoLens. However, theres lots we've not  covered, so
# if you're unsure *exactly* what PyAutoLens is doing or don't quite know how to fit the lens model you want, you should
# checkout the howtolens lecture series for a complete description of how to use PyAutoLens.
