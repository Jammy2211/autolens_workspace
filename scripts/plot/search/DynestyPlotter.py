"""
Plots: DynestyPlotter
=====================

This example illustrates how to plot visualization summarizing the results of a dynesty non-linear search using
a `ZeusPlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
First, lets create a result via dynesty by repeating the simple model-fit that is performed in 
the `modeling/mass_total__source_parametric.py` example.
"""
dataset_name = "mass_sie__source_sersic"

search = af.DynestyStatic(
    path_prefix=path.join("plot"),
    name="DynestyPlotter",
    unique_tag=dataset_name,
    nlive=50,
)

dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisImaging(dataset=imaging)

result = search.fit(model=model, analysis=analysis)

"""
We now pass the samples to a `DynestyPlotter` which will allow us to use dynesty's in-built plotting libraries to 
make figures.

The dynesty readthedocs describes fully all of the methods used below 

 - https://dynesty.readthedocs.io/en/latest/quickstart.html
 - https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.plotting
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)

"""
The `cornerplot` method produces a triangle of 1D and 2D PDF's of every parameter in the model fit.
"""
dynesty_plotter.cornerplot(
    dims=None,
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    color="black",
    smooth=0.02,
    quantiles_2d=None,
    hist_kwargs=None,
    hist2d_kwargs=None,
    label_kwargs=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    verbose=False,
)

"""
The `runplot` method shows how the estimates of the log evidence and other quantities progress as a function of
iteration number during the dynesty model-fit.
"""
dynesty_plotter.runplot(
    span=None,
    logplot=False,
    kde=True,
    nkde=1000,
    color="blue",
    plot_kwargs=None,
    label_kwargs=None,
    lnz_error=True,
    lnz_truth=None,
    truth_color="red",
    truth_kwargs=None,
    max_x_ticks=8,
    max_y_ticks=3,
    use_math_text=True,
    mark_final_live=True,
    fig=None,
)

"""
The `traceplot` method shows how the live points of each parameter converged alongside their PDF.
"""
dynesty_plotter.traceplot(
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    smooth=0.02,
    thin=1,
    dims=None,
    post_color="blue",
    post_kwargs=None,
    kde=True,
    nkde=1000,
    trace_cmap="plasma",
    trace_color=None,
    trace_kwargs=None,
    connect=False,
    connect_highlight=10,
    connect_color="red",
    connect_kwargs=None,
    max_n_ticks=5,
    use_math_text=False,
    label_kwargs=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    verbose=False,
    fig=None,
)


"""
The `cornerpoints` method produces a triangle of 1D and 2D plots of the weight points of every parameter in the model 
fit.
"""
dynesty_plotter.cornerpoints(
    dims=None,
    thin=1,
    span=None,
    cmap="plasma",
    color=None,
    kde=True,
    nkde=1000,
    plot_kwargs=None,
    label_kwargs=None,
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    max_n_ticks=5,
    use_math_text=False,
    fig=None,
)

"""
The `boundplot` method produces a plot of the bounding distribution used to draw a live point at a given iteration `it`
of the sample or of a dead point `idx`.
"""
dynesty_plotter.boundplot(
    dims=(2, 2),
    it=100,
    idx=None,
    prior_transform=None,
    periodic=None,
    reflective=None,
    ndraws=5000,
    color="gray",
    plot_kwargs=None,
    label_kwargs=None,
    max_n_ticks=5,
    use_math_text=False,
    show_live=False,
    live_color="darkviolet",
    live_kwargs=None,
    span=None,
    fig=None,
)

"""
The `cornerbound` method produces the bounding distribution used to draw points at an input iteration `it` or used to
specify a dead point via `idx`.
"""
dynesty_plotter.cornerbound(
    it=100,
    idx=None,
    dims=None,
    prior_transform=None,
    periodic=None,
    reflective=None,
    ndraws=5000,
    color="gray",
    plot_kwargs=None,
    label_kwargs=None,
    max_n_ticks=5,
    use_math_text=False,
    show_live=False,
    live_color="darkviolet",
    live_kwargs=None,
    span=None,
    fig=None,
)

"""
Finish.
"""
