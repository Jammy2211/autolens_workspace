"""
Tutorial 8: Model-Fit
=====================

To illustrate lens modeling using an inversion this tutorial revists revisit the complex source model-fit that we
performed in tutorial 6 of chapter 3. This time, as you have probably guessed, we will fit the complex source using
an inversion.

We will use search chaining to do this, first fitting the source with a light profile, thereby initialize the mass
model priors and avoiding the unphysical solutions discussed in tutorial 6. In the later searches we will switch to
an `Inversion`.
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
__Initial Setup__

we'll use strong lensing data, where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is four `Sersic`.
"""
dataset_name = "source_complex"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.6
)


dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()


"""
__Model + Search + Analysis + Model-Fit (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a parametric `SersicCore` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
model_1 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            mass=al.mp.Isothermal,
            shear=al.mp.ExternalShear,
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
    ),
)

search_1 = af.Nautilus(
    path_prefix=path.join("howtolens", "chapter_4"),
    name="search[1]_mass[sie]_source[lp]",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Position Likelihood (Search 2)__

We add a penalty term ot the likelihood function, which penalizes models where the brightest multiple images of
the lensed source galaxy do not trace close to one another in the source plane. This removes "demagnified source
solutions" from the source pixelization, which one is likely to infer without this penalty.

A comprehensive description of why we do this is given at the following readthedocs page. 

You were directed to this page in tutorial 6, however I suggest you reread the section "Auto Position Updates" as
we will be using this functionality below.

 https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html

__Model + Search + Analysis + Model-Fit (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [Parameters fixed to 
 results of search 1].
 
 - The source galaxy's pixelization uses an `Overlay` image-mesh [2 parameters]

 - The source-galaxy's pixelization uses a `Delaunay` mesh [0 parameters].

 - This pixelization is regularized using a `ConstantSplit` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

This search allows us to very efficiently set up the resolution of the mesh and regularization coefficient 
of the regularization scheme, before using these models to refit the lens mass model.

Also, note how we can pass the `al.SettingsInversion` object to an analysis class to customize if the border relocation
is used.
"""
model_2 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            mass=result_1.instance.galaxies.lens.mass,
            shear=result_1.instance.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            pixelization=af.Model(
                al.Pixelization,
                image_mesh=al.image_mesh.Overlay,
                mesh=al.mesh.Delaunay,
                regularization=al.reg.Constant,
            ),
        ),
    ),
)

search_2 = af.Nautilus(
    path_prefix=path.join("howtolens", "chapter_4"),
    name="search[2]_mass[sie]_source[pix_init]",
    unique_tag=dataset_name,
    n_live=50,
)

analysis_2 = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood=result_1.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
    settings_inversion=al.SettingsInversion(use_border_relocator=True),
)

"""
__Run Time__

The run time of a pixelization is longer than many other features, with the estimate below coming out at around ~0.5 
seconds per likelihood evaluation. This is because the fit has a lot of linear algebra to perform in order to
reconstruct the source on the pixel-grid.

Nevertheless, this is still fast enough for most use-cases. If run-time is an issue, the following factors determine
the run-time of a a pixelization and can be changed to speed it up (at the expense of accuracy):

 - The number of unmasked pixels in the image data. By making the mask smaller (e.g. using an annular mask), the 
   run-time will decrease.

 - The number of source pixels in the pixelization. By reducing the `shape` from (30, 30) the run-time will decrease.

This also serves to highlight why the positions threshold likelihood is so powerful. The likelihood evaluation time
of this step is below 0.001 seconds, meaning that the initial parameter space sampling is extremely efficient even
for a pixelization (this is not accounted for in the run-time estimate below)!
"""
run_time_dict, info_dict = analysis_2.profile_log_likelihood_function(
    instance=model_2.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model_2.total_free_parameters * 10000)
    / search_2.number_of_cores,
)

"""
Run the search.
"""
result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Model + Search (Search 3)__

We use the results of searches 1 and 2 to create the lens model fitted in search 3, where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters: priors 
 initialized from search 1].
 
 - The source galaxy's pixelization uses an `Overlay` image-mesh [parameters fixed to results of search 2].

 - The source-galaxy's pixelization uses a `Delaunay` mesh [parameters fixed to results of search 2].

 - This pixelization is regularized using a `ConstantSplit` scheme [parameters fixed to results of search 2]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.

This search therefore refits the lens mass model using the pixelized source.
"""
model_3 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            mass=result_1.model.galaxies.lens.mass,
            shear=result_1.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            pixelization=result_2.instance.galaxies.source.pixelization,
        ),
    ),
)

search_3 = af.Nautilus(
    path_prefix=path.join("howtolens", "chapter_4"),
    name="search[3]_mass[sie]_source[pix]",
    unique_tag=dataset_name,
    n_live=100,
)

"""
__Positions + Analysis + Model-Fit (Search 3)__

The unphysical solutions that can occur in an `Inversion` can be mitigated by using a positions threshold to resample
mass models where the source's brightest lensed pixels do not trace close to one another. With search chaining, we can
in fact use the model-fit of a previous search (in this example, search 1) to compute the positions that we use in a 
later search.

Below, we use the results of the first search to compute the lensed source positions that are input into search 2. The
code below uses the  maximum log likelihood model mass model and source galaxy centre, to determine where the source
positions are located in the image-plane. 

We also use this result to set the `position_threshold`, whereby the threshold value is based on how close these 
positions trace to one another in the source-plane (using the best-fit mass model again). This threshold is multiplied 
by a `factor` to ensure it is not too small (and thus does not remove plausible mass  models). If, after this 
multiplication, the threshold is below the `minimum_threshold`, it is rounded up to this minimum value.
"""
analysis_3 = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood=result_2.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Wrap Up__

And with that, we now have a pipeline to model strong lenses using an inversion! 

Checkout the example pipelines in the `autolens_workspace/*/chaining` package for inversion pipelines that 
includes the lens light component.
"""
