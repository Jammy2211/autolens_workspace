import autofit as af
import autolens as al
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging

# We've learnt nearly all the tools we need to model strong lenses, so I'm now going to quickly cover how you should
# choose your mask. I'll also show you another neat trick to improve the speed and accuracy of your non-linear search.

# You need to change the path below to the chapter 2 directory.
chapter_path = "/path/to/user/autolens_workspace/howtolens/chapter_2_lens_modeling/"
chapter_path = "/home/jammy/PycharmProjects/PyAutoLens/workspace/howtolens/chapter_2_lens_modeling/"

af.conf.instance = af.conf.Config(
    config_path=chapter_path + "/configs/t8_masking_and_positions",
    output_path=chapter_path + "/output",
)

# Lets simulate the simple image we've used throughout this chapter.
def simulate():

    psf = al.PSF.from_gaussian(shape=(11, 11), sigma=0.1, pixel_scale=0.1)

    grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
        shape=(130, 130), pixel_scale=0.1, sub_grid_size=2
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.6
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.SphericalExponential(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=0.2
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scale=0.1,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_noise=True,
    )


# Simulate the CCD data.
ccd_data = simulate()
al.ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

# When it comes to determining an appropriate mask for this image, the best approach is to set up a mask using the mask
# module and pass it to a ccd plotter. You can then check visually if the mask is an appropriate size or not.
# Below, we choose an inner radius that cuts into our lensed source galaxy - clearly this isn't a good mask.
mask = al.Mask.circular_annular(
    shape=ccd_data.shape,
    pixel_scale=ccd_data.pixel_scale,
    inner_radius_arcsec=1.4,
    outer_radius_arcsec=2.4,
)

al.ccd_plotters.plot_ccd_subplot(
    ccd_data=ccd_data, mask=mask, extract_array_from_mask=True, zoom_around_mask=True
)

# So, lets decrease the inner radius to correct for this.
mask = al.Mask.circular_annular(
    shape=ccd_data.shape,
    pixel_scale=ccd_data.pixel_scale,
    inner_radius_arcsec=0.6,
    outer_radius_arcsec=2.4,
)

al.ccd_plotters.plot_ccd_subplot(
    ccd_data=ccd_data, mask=mask, extract_array_from_mask=True, zoom_around_mask=True
)

# When we run the phase, we don't pass it the mask as an array. Instead, we pass it the mask as a function. The reason
# for this will become clear in the next chapter, but for now I would say you just accept this syntax.
def mask_function():
    return al.Mask.circular_annular(
        shape=ccd_data.shape,
        pixel_scale=ccd_data.pixel_scale,
        inner_radius_arcsec=0.6,
        outer_radius_arcsec=2.4,
    )


phase_with_custom_mask = phase_imaging.PhaseImaging(
    phase_name="phase_t8_with_custom_mask",
    galaxies=dict(
        lens=gm.GalaxyModel(redshift=0.5), source=gm.GalaxyModel(redshift=1.0)
    ),
    mask_function=mask_function,  # <- We input the mask function here
    optimizer_class=af.MultiNest,
)

# So, our mask encompasses the lensed source galaxy. However, is this really the right sized mask? Do we want
# a bigger mask? a smaller mask?

# When it comes to masking, we are essentially balancing run-speed and accuracy. If speed wasn't a consideration,
# bigger masks would *always* be better, for two reasons:

# 1) The lensed source galaxy may have very faint emission that when you look at the plot above you don't notice.
#    Overly aggressive masking risks you masking out some of that light - data which would better constrain your
#    lens model!

# 2) When you fit an image with a model image the fit is performed only within the masked region. Outside of the
#    masked region it is possible that the model image produces some source-galaxy light in a region of the image
#    where it isn't actually observed. If this region is masked, the poor fit in this region won't reduce the model's
#    likelihood.

# As you use PyAutoLens more you will get a feel for how fast an analysis will run given a certain image resolution,
# lens model complexity, non-linear search priors / settings, etc. As you develop this intuition, I would recommend you
# always aim to use masks as big as possible which still give a reasonable run-speed. Aggresive masking
# will get your code running fast - but it could lead you to infer an incorrect lens model!

# If you are fitting the foreground lens galaxy's light you pretty much have no choice but to use a large circular
# mask anyway, as you'll need to capture the lens's extended emission. Chances are this will encompass the entire
# source galaxy.


# We can also manually specify a set of image-pixels which correspond to the multiple images of the source-galaxy(s).
# During the analysis, PyAutoLens will first check that these pixels trace within a specified arc-second threshold of
# one another (which is controlled by the 'position_threshold' parameter input into a phase). This
# provides two benefits:

# 1) The analysis runs faster as the non-linear search avoids searching regions of parameter space where the
#    mass-model is clearly not accurate.

# 2) By removing these solutions, a global-maximum solution may be reached instead of a local-maxima. This is because
#    removing the incorrect mass models makes the non-linear parameter space less complex.

# We can easily check the image-positions are accurate by plotting them using our ccd plotter (they are the magenta
# dots on the image).
al.ccd_plotters.plot_ccd_subplot(
    ccd_data=ccd_data, positions=[[[1.6, 0.0], [0.0, 1.6], [-1.6, 0.0], [0.0, -1.6]]]
)

# We can then tell our phase to use these positions in the analysis.
phase_with_positions = phase_imaging.PhaseImaging(
    phase_name="phase_t8_with_positions",
    galaxies=dict(
        lens=gm.GalaxyModel(redshift=0.5, mass=al.mass_profiles.SphericalIsothermal),
        source=gm.GalaxyModel(
            redshift=1.0, light=al.light_profiles.SphericalExponential
        ),
    ),
    positions_threshold=0.5,  # <- We input a positions threshold here, to signify how far pixels must trace within one another.
    optimizer_class=af.MultiNest,
)

# The positions are passed to the phase when we run it,as shown below.
print(
    "MultiNest has begun running - checkout the workspace/howtolens/chapter_2_lens_modeling/output/t7_multinest_black_magic"
    " folder for live output of the results, images and lens model."
    " This Jupyter notebook cell with progress once MultiNest has completed - this could take some time!"
)

phase_with_positions.run(
    data=ccd_data, positions=[[[1.6, 0.0], [0.0, 1.6], [-1.6, 0.0], [0.0, -1.6]]]
)

print("MultiNest has finished run - you may now continue the notebook.")
# You may observe multiple source-galaxies each with their own set of multiple-images. If you have a means by
# which to pair different positions to the same source galaxies (for example, spectroscopic data) you can set up
# multiple sets of positions which each have to trace to within the position threshold of one another for the lens
# model to be accepted.


def simulate_two_galaxies():

    psf = al.PSF.from_gaussian(shape=(11, 11), sigma=0.1, pixel_scale=0.1)

    grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
        shape=(130, 130), pixel_scale=0.1, sub_grid_size=2
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.6
        ),
    )

    source_galaxy_0 = al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.SphericalExponential(
            centre=(1.0, 0.0), intensity=0.2, effective_radius=0.2
        ),
    )

    source_galaxy_1 = al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.SphericalExponential(
            centre=(-1.0, 0.0), intensity=0.2, effective_radius=0.2
        ),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1]
    )

    return al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scale=0.1,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_noise=True,
    )


# Simulate the new data.

ccd_data = simulate_two_galaxies()

al.ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

# To specify the positions we break the positions list into two cells. They will be plotted in different colours to
# represent the fact they trace from different source galaxies.
al.ccd_plotters.plot_ccd_subplot(
    ccd_data=ccd_data,
    positions=[[[2.65, 0.0], [-0.55, 0.0]], [[-2.65, 0.0], [0.55, 0.0]]],
)

# Again, we tell our phase to use the positions and pass this list of pixels to our phase when we run it.
phase_with_x2_positions = phase_imaging.PhaseImaging(
    phase_name="phase_t8_with_x2_positions",
    galaxies=dict(
        lens=gm.GalaxyModel(redshift=0.5, mass=al.mass_profiles.SphericalIsothermal),
        source_0=gm.GalaxyModel(
            redshift=1.0, light=al.light_profiles.SphericalExponential
        ),
        source_1=gm.GalaxyModel(
            redshift=1.0, light=al.light_profiles.SphericalExponential
        ),
    ),
    positions_threshold=0.5,  # <- We input a positions threshold here, to signify how far pixels must trace within one another.
    optimizer_class=af.MultiNest,
)

phase_with_x2_positions.run(
    data=ccd_data, positions=[[[2.65, 0.0], [-0.55, 0.0]], [[-2.65, 0.0], [0.55, 0.0]]]
)

# And that completes our final tutorial in this chapter! At this point, I recommend that you checkout the
# 'autolens_workspace/tools/mask_maker.py' and 'autolens_workspace/tools/positions_maker.py'
# scripts. These tools allow you create custom masks and positions for a specific strong lens and output them so
# they can be loaded before an analysis.

# When we cover pipelines next, you'll see that pipelines allow us to use a custom mask and set of positions for each lens
# we model. So, although we have to draw the masks and positions for each lens in a sample, once we've done that
# we can fit all lenses with one standardized pipeline!

#  There are two things you should bare in mind in terms of masking and positions:
#
# 1) Customizing the mask and positions for the analysis of one strong lens gets the analysis running fast and can
#    provide accurate non-linear sampling. However, for a large sample of lenses, customizing the mask and positions
#    will begin to take a lot of time. If you're willing to put that time and effort in, great, but these solutions
#    *do not* scale-up to large samples of lenses.

# 2) A word of warning - be *extremely* careful when using positions, especially if it is unclear if the lensed source
#    galaxy has one or multiple source's of light. If your position threshold is small and the positions you give the
#    analysis correspond to different parts of the source, you may remove the *correct lens model*. In my experience,
#    as long as you keep the threshold above ~0.5" you'll be fine.

# And with that, we've completed the chapter.
