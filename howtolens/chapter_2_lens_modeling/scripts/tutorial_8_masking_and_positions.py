import autofit as af
import autolens as al
import autolens.plot as aplt

# We've learnt nearly all the tools we need to model strong lenses, so I'm now going to quickly cover how you should
# choose your mask. I'll also show you another neat trick to improve the speed and accuracy of your non-linear search.

# You need to change the path below to the chapter 2 directory.
chapter_path = "/path/to/user/autolens_workspace/howtolens/chapter_2_lens_modeling/"
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace/howtolens/chapter_2_lens_modeling/"

af.conf.instance = af.conf.Config(
    config_path=chapter_path + "/configs/t8_masking_and_positions",
    output_path=chapter_path + "/output",
)

# Lets simulate the simple image we've used throughout this chapter.
def simulate():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=0.1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalExponential(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=0.2
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(130, 130),
        pixel_scales=0.1,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


# Simulate the imaging dataset.
imaging = simulate()
aplt.imaging.subplot_imaging(imaging=imaging)

# When it comes to determining an appropriate mask for this image, the best approach is to set up a mask using the mask
# module and pass it to a imaging plotter. You can then check visually if the mask is an appropriate size or not.
# Below, we choose an inner radius that cuts into our lensed source galaxy - clearly this isn't a good mask.
mask = al.mask.circular_annular(
    shape_2d=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    inner_radius=1.4,
    outer_radius=2.4,
)

aplt.imaging.subplot_imaging(imaging=imaging, mask=mask)

# So, lets decrease the inner radius to correct for this.
mask = al.mask.circular_annular(
    shape_2d=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    inner_radius=0.6,
    outer_radius=2.4,
)

aplt.imaging.subplot_imaging(imaging=imaging, mask=mask)


phase_with_custom_mask = al.PhaseImaging(
    phase_name="phase_t8_with_custom_mask",
    galaxies=dict(
        lens=al.GalaxyModel(redshift=0.5), source=al.GalaxyModel(redshift=1.0)
    ),
    optimizer_class=af.MultiNest,
)

# So, our mask encompasses the lensed source galaxy. However, is this really the right sized mask? Do we want
# a bigger mask? a smaller mask?

# When it comes to masking, we are essentially balancing run-speed and accuracy. If speed wasn't a consideration,
# bigger masks would *always* be better, for two reasons:

# 1) The lensed source galaxy may have very faint emission that when you look at the plotters above you don't notice.
#    Overly aggressive masking risks you masking out some of that light -data which would better constrain your
#    lens model!

# 2) When you fit an image with a model image the fit is performed only within the masked region. Outside of the
#    masked region it is possible that the model image produces some source-galaxy light in a region of the image
#    where it isn't actually observed. If this region is masked, the poor fit in this region won't reduce the model's
#    likelihood.

# As you use PyAutoLens more you will get a feel for how fast an analysis will run given a certain image resolution,
# lens model complexity, non-linear search priors / setup, etc. As you develop this intuition, I would recommend you
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

# We can easily check the image-positions are accurate by plotting them using our imaging plotter (they are the magenta
# dots on the image).
aplt.imaging.subplot_imaging(
    imaging=imaging, positions=[(1.6, 0.0), (0.0, 1.6), (-1.6, 0.0), (0.0, -1.6)]
)

# We can then tell our phase to use these positions in the analysis.
phase_with_positions = al.PhaseImaging(
    phase_name="phase_t8_with_positions",
    galaxies=dict(
        lens=al.GalaxyModel(redshift=0.5, mass=al.mp.SphericalIsothermal),
        source=al.GalaxyModel(redshift=1.0, light=al.lp.SphericalExponential),
    ),
    positions_threshold=0.5,  # <- We input a positions threshold here, to signify how far pixels must trace within one another.
    optimizer_class=af.MultiNest,
)

# The positions are passed to the phase when we run it,as shown below.
print(
    "MultiNest has begun running - checkout the autolens_workspace/howtolens/chapter_2_lens_modeling/output/t7_multinest_black_magic"
    " folder for live output of the results, images and lens model."
    " This Jupyter notebook cell with progress once MultiNest has completed - this could take some time!"
)

phase_with_positions.run(
    dataset=imaging,
    mask=mask,
    positions=[(1.6, 0.0), (0.0, 1.6), (-1.6, 0.0), (0.0, -1.6)],
)

print("MultiNest has finished run - you may now continue the notebook.")
# You may observe multiple source-galaxies each with their own set of multiple-images. If you have a means by
# which to pair different positions to the same source galaxies (for example, spectroscopicdata) you can set up
# multiple sets of positions which each have to trace to within the position threshold of one another for the lens
# model to be accepted.


def simulate_two_galaxies():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=0.1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6),
    )

    source_galaxy_0 = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalExponential(
            centre=(1.0, 0.0), intensity=0.2, effective_radius=0.2
        ),
    )

    source_galaxy_1 = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalExponential(
            centre=(-1.0, 0.0), intensity=0.2, effective_radius=0.2
        ),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1]
    )

    simulator = al.simulator.imaging(
        shape_2d=(130, 130),
        pixel_scales=0.1,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


# Simulate the newdataset.

imaging = simulate_two_galaxies()

aplt.imaging.subplot_imaging(imaging=imaging)

# To specify the positions we break the positions list into two cells. They will be plotted in different colours to
# represent the fact they trace from different source galaxies.
aplt.imaging.subplot_imaging(
    imaging=imaging,
    positions=[[(2.65, 0.0), (-0.55, 0.0)], [(-2.65, 0.0), (0.55, 0.0)]],
)

# Again, we tell our phase to use the positions and pass this list of pixels to our phase when we run it.
phase_with_x2_positions = al.PhaseImaging(
    phase_name="phase_t8_with_x2_positions",
    galaxies=dict(
        lens=al.GalaxyModel(redshift=0.5, mass=al.mp.SphericalIsothermal),
        source_0=al.GalaxyModel(redshift=1.0, light=al.lp.SphericalExponential),
        source_1=al.GalaxyModel(redshift=1.0, light=al.lp.SphericalExponential),
    ),
    positions_threshold=0.5,  # <- We input a positions threshold here, to signify how far pixels must trace within one another.
    optimizer_class=af.MultiNest,
)

phase_with_x2_positions.run(
    dataset=imaging,
    mask=mask,
    positions=[[(2.65, 0.0), (-0.55, 0.0)], [(-2.65, 0.0), (0.55, 0.0)]],
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
