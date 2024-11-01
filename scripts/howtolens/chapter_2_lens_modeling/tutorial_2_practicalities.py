"""
Tutorial 2: Practicalities
==========================

In the last tutorial, we introduced foundational statistical concepts essential for model-fitting, such as parameter
spaces, likelihoods, priors, and non-linear searches. Understanding these statistical concepts is crucial for
performing model fits effectively.

However, achieving successful model-fitting also requires practical skills, including how to manage outputs,
review results, interpret model quality, and ensure run-times are efficient enough for your scientific needs.

This tutorial will focus on these practical aspects of model-fitting, including:

- How to save results to your hard disk.
- How to navigate the output folder and examine model-fit results to assess quality.
- How to estimate the run-time of a model-fit before initiating it, and change settings to make it faster or run the analysis in parallel.

__Contents__

This tutorial is split into the following sections:

 **PyAutoFit:** The parent package of PyAutoGalaxy, which handles practicalities of model-fitting.
 **Initial Setup:** Load the dataset we'll fit a model to using a non-linear search.
 **Mask:** Apply a mask to the dataset.
 **Model:** Introduce the model we will fit to the data.
 **Search:** Setup the non-linear search, Nautilus, used to fit the model to the data.
 **Search Settings:** Discuss the settings of the non-linear search, including the number of live points.
 **Number Of Cores:** Discuss how to use multiple cores to fit models faster in parallel.
 **Parallel Script:** Running the model-fit in parallel if a bug occurs in a Jupiter notebook.
 **Iterations Per Update:** How often the non-linear search outputs the current results to hard-disk.
 **Analysis:** Create the Analysis object which contains the `log_likelihood_function` that the non-linear search calls.
 **Model-Fit:** Fit the model to the data.
 **Result:** Print the results of the model-fit to the terminal.
 **Output Folder:** Inspect the output folder where results are stored.
 **Unique Identifier:** Discussion of the unique identifier of the model-fit which names the folder in the output directory.
 **Output Folder Contents:** What is output to the output folder (model results, visualization, etc.).
 **Result:** Plot the best-fit model to the data.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__PyAutoFit__

Modeling uses the probabilistic programming language
[PyAutoFit](https://github.com/rhayes777/PyAutoFit), an open-source project that allows complex model
fitting techniques to be straightforwardly integrated into scientific modeling software. 

The majority of tools that make model-fitting practical are provided by PyAutoFit, for example it handles
all output of the non-linear search to hard-disk, the visualization of results and the estimation of run-times.
"""
import autofit as af

"""
__Initial Setup__

Lets first load the `Imaging` dataset we'll fit a model with using a non-linear search. 

This is the same dataset we fitted in the previous tutorial, and we'll repeat the same fit, as we simply want
to illustrate the practicalities of model-fitting in this tutorial.
"""
dataset_name = "simple__no_lens_light__mass_sis"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

The non-linear fit also needs a `Mask2D`, lets use a 3.0" circle.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

We compose the model using the same API as the previous tutorial.

This model is the same as the previous tutorial, an `Isothermal` sphereical mass profile representing the lens
galaxy and a `ExponentialCoreSph` light profile representing the source galaxy.
"""
# Lens:

mass = af.Model(al.mp.IsothermalSph)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

bulge = af.Model(al.lp_linear.ExponentialCoreSph)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)


"""
__Search__

To fit the model, we now create the non-linear search object, selecting the nested sampling algorithm, Nautilus, 
as recommended in the previous tutorial.

We set up the `Nautilus` object with these parameters:

- **`path_prefix`**: specifies the output directory, here set to `autolens_workspace/output/howtogalaxy/chapter_2`.
  
- **`name`**: gives the search a descriptive name, which creates the full output path 
as `autolens_workspace/output/howtogalaxy/chapter_2/tutorial_2_practicalities`.

- **`n_live`**: controls the number of live points Nautilus uses to sample parameter space.

__Search Settings__

Nautilus samples parameter space by placing "live points" representing different galaxy models. Each point has an 
associated `log_likelihood` that reflects how well it fits the data. By mapping where high-likelihood solutions are 
located, it can focus on searching those regions.

The main setting to balance is the **number of live points**. More live points allow Nautilus to map parameter space 
more thoroughly, increasing accuracy but also runtime. Fewer live points reduce run-time but may make the search less 
reliable, possibly getting stuck in local maxima.

The ideal number of live points depends on model complexity. More parameters generally require more live points, but 
the default of 200 is sufficient for most lens models. Lower values can still yield reliable results, particularly 
for simpler models. For this example (7 parameters), we reduce the live points to 100 to speed up runtime without 
compromising accuracy.

Tuning non-linear search settings (e.g., the number of live points) to match model complexity is essential. We aim 
for enough live points to ensure accurate results (i.e., finding a global maximum) but not so many that runtime 
is excessive.

In practice, the optimal number of live points is often found through trial and error, guided by summary statistics 
on how well the search is performing, which we’ll cover below. For this single Sersic model with a linear light 
profile, 80 live points is sufficient to achieve reliable results.

__Number Of Cores__

We may include an input `number_of_cores`, which when above 1 means that Nautilus uses parallel processing to sample 
multiple models at once on your CPU. When `number_of_cores=2` the search will run roughly two times as
fast, for `number_of_cores=3` three times as fast, and so on. The downside is more cores on your CPU will be in-use
which may hurt the general performance of your computer.

You should experiment to figure out the highest value which does not give a noticeable loss in performance of your 
computer. If you know that your processor is a quad-core processor you should be able to use `number_of_cores=4`. 

Above `number_of_cores=4` the speed-up from parallelization diminishes greatly. We therefore recommend you do not
use a value above this.

For users on a Windows Operating system, using `number_of_cores>1` may lead to an error, in which case it should be 
reduced back to 1 to fix it.

__Parallel Script__

Depending on the operating system (e.g. Linux, Mac, Windows), Python version, if you are running a Jupyter notebook 
and other factors, this script may not run a successful parallel fit (e.g. running the script 
with `number_of_cores` > 1 will produce an error). It is also common for Jupyter notebooks to not run in parallel 
correctly, requiring a Python script to be run, often from a command line terminal.

To fix these issues, the Python script needs to be adapted to use an `if __name__ == "__main__":` API, as this allows
the Python `multiprocessing` module to allocate threads and jobs correctly. An adaptation of this example script 
is provided at `autolens_workspace/scripts/modeling/imaging/customize/parallel.py`, which will hopefully run 
successfully in parallel on your computer!

Therefore if paralellization for this script doesn't work, check out the `parallel.py` example. You will need to update
all scripts you run to use the this format and API. 

__Iterations Per Update__

Every N iterations, the non-linear search outputs the current results to the folder `autolens_workspace/output`,
which includes producing visualization. 

Depending on how long it takes for the model to be fitted to the data (see discussion about run times below), 
this can take up a large fraction of the run-time of the non-linear search.

For this fit, the fit is very fast, thus we set a high value of `iterations_per_update=10000` to ensure these updates
so not slow down the overall speed of the model-fit. 

**If the iteration per update is too low, the model-fit may be significantly slowed down by the time it takes to
output results and visualization frequently to hard-disk. If your fit is consistent displaying a log saying that it
is outputting results, try increasing this value to ensure the model-fit runs efficiently.**
"""
search = af.Nautilus(
    path_prefix=path.join("howtolens", "chapter_2"),
    name="tutorial_2_practicalities",
    unique_tag=dataset_name,
    n_live=100,
    iterations_per_update=2500,
    # number_of_cores=1, # Try uncommenting this line to run in parallel but see "Parallel Script" above.
)

"""
__Analysis__

We again create the `AnalysisImaging` object which contains the `log_likelihood_function` that the non-linear search
calls to fit the model to the data.
"""
analysis = al.AnalysisImaging(dataset=dataset)

"""
__Run Times__

Lens modeling can be a computationally expensive process. When fitting complex models to high resolution datasets 
run times can be of order hours, days, weeks or even months.

Run times are dictated by two factors:

 - **The log likelihood evaluation time:** the time it takes for a single `instance` of the model to be fitted to 
   the dataset such that a log likelihood is returned.

 - **The number of iterations (e.g. log likelihood evaluations) performed by the non-linear search:** more complex lens
   models require more iterations to converge to a solution (and as discussed above, settings like the number of live
   points also control this).

The log likelihood evaluation time can be estimated before a fit using the `profile_log_likelihood_function` method,
which returns two dictionaries containing the run-times and information about the fit.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

"""
The overall log likelihood evaluation time is given by the `fit_time` key.

For this example, it is ~0.05 seconds, which is extremely fast for lens modeling. 

The more advanced fitting techniques discussed at the end of chapter 1 (e.g. shapelets, multi Gaussian expansions, 
pixelizations) have longer log likelihood evaluation times (1-3 seconds) and therefore may require more efficient 
search settings to keep the overall run-time feasible.
"""
print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")

"""
To estimate the expected overall run time of the model-fit we multiply the log likelihood evaluation time by an 
estimate of the number of iterations the non-linear search will perform. 

Estimating this is tricky, as it depends on the lens model complexity (e.g. number of parameters)
and the properties of the dataset and model being fitted. With 7 free parameters, this gives an estimate 
of ~7000 iterations, which at ~0.05 seconds per iteration gives a total run-time of ~180 seconds (or ~3 minutes).

For this example, we conservatively estimate that the non-linear search will perform ~1000 iterations per free 
parameter in the model. This is an upper limit, with models typically converging in fewer iterations.

If you perform the fit over multiple CPUs, you can divide the run time by the number of cores to get an estimate of
the time it will take to fit the model. Parallelization with Nautilus scales well, it speeds up the model-fit by the 
`number_of_cores` for N < 8 CPUs and roughly `0.5*number_of_cores` for N > 8 CPUs. This scaling continues 
for N> 50 CPUs, meaning that with super computing facilities you can always achieve fast run times!
"""
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 1000)
    / search.number_of_cores,
)

"""
__Model-Fit__

To begin the model-fit, we pass the model and analysis objects to the search, which performs a non-linear search to 
identify models that best fit the data.

Running model fits via non-linear search can take significant time. While the fit in this tutorial should 
complete in a few minutes, more complex models may require longer run times. In Jupyter notebooks, this can be 
limiting, as the notebook cell will only complete once the fit finishes, preventing you from advancing through the 
tutorial or running additional code.

To work around this, we recommend running tutorial scripts as standalone Python scripts, found 
in `autolens_workspace/scripts/howtogalaxy`. These scripts mirror the notebook tutorials but run independently of 
Jupyter notebooks. For example, you can start a script with:

`python3 scripts/howtogalaxy/chapter_2_modeling/tutorial_2_practicalities.py`

Using scripts allows results to be saved to the hard drive in the `output` folder, enabling you to inspect results 
immediately once the script completes. When rerun, the script loads results directly from disk, so any Jupyter 
notebook cells will quickly load and display the complete model-fit results if they’re already saved.

This approach not only avoids the slowdowns associated with notebook cells during lengthy runs but is also essential 
for using super-computers for fitting tasks, as they require separate Python scripts.

For tasks like loading results, inspecting data, plotting, and interpreting results, Jupyter notebooks remain ideal.
"""
print(
    "The non-linear search has begun running - checkout the autolens_workspace/output/"
    " folder for live output of the results, images and lens model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result = search.fit(model=model, analysis=analysis)

print("Search has finished run - you may now continue the notebook.")

"""
__Result Info__

A concise readable summary of the results is given by printing its `info` attribute.

[Above, we discussed that the `info_whitespace_length` parameter in the config files could b changed to make 
the `model.info` attribute display optimally on your computer. This attribute also controls the whitespace of the
`result.info` attribute.]
"""
print(result.info)

"""
__Output Folder__

Now checkout the `autolens_workspace/output` folder.

This is where the results of the search are written to hard-disk (in the `tutorial_2_practicalities` folder). 

Once completed images, results and information about the fit appear in this folder, meaning that you don't need 
to keep running Python code to see the result.

__Unique Identifier__

In the output folder, you will note that results are in a folder which is a collection of random characters. This acts 
as a `unique_identifier` of the model-fit, where this identifier is generated based on the model, search and dataset 
that are used in the fit.
 
An identical combination of model, search and dataset generates the same identifier, meaning that rerunning the
script will use the existing results to resume the model-fit. In contrast, if you change the model, search or dataset,
a new unique identifier will be generated, ensuring that the model-fit results are output into a separate folder. 

We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different datasets
with the same model and search results are output to a different folder. We achieved this for the fit above by passing 
the `dataset_name` to the search's `unique_tag`.

__Output Folder Contents__

Now this is running you should checkout the `autolens_workspace/output` folder. This is where the results of the 
search are written to hard-disk (in the `start_here` folder), where all outputs are human readable (e.g. as .json,
.csv or text files).

As the fit progresses, results are written to the `output` folder on the fly using the highest likelihood model found
by the non-linear search so far. This means you can inspect the results of the model-fit as it runs, without having to
wait for the non-linear search to terminate.
 
The `output` folder includes:

 - `model.info`: Summarizes the model, its parameters and their priors discussed in the next tutorial.
 
 - `model.results`: Summarizes the highest likelihood model inferred so far including errors.
 
 - `images`: Visualization of the highest likelihood model-fit to the dataset, (e.g. a fit subplot showing the 
 galaxies, model data and residuals).
 
 - `files`: A folder containing .fits files of the dataset, the model as a human-readable .json file, 
 a `.csv` table of every non-linear search sample and other files containing information about the model-fit.
 
 - `search.summary`: A file providing summary statistics on the performance of the non-linear search.
 
 - `search_internal`: Internal files of the non-linear search (in this case Nautilus) used for resuming the fit and
  visualizing the search.

__Result__

The `search.fit` method produces a `result` object, packed with useful information about the model fit that we’ll 
explore in detail in a later tutorial.

One component of the `result` object we’ll use now is the `FitImaging` object, which corresponds to the set of model 
parameters that yielded the maximum log-likelihood solution. Plotting this object lets us visually inspect how well 
the model fits the data.

In this example, the fit to the data is excellent, with residuals near zero, as expected since the same model was 
used both to simulate and fit the data.
"""
fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
The Probability Density Functions (PDF's) of the results can be plotted using an in-built visualization 
library, which is wrapped via the `NestPlotter` object.

The PDF shows the 1D and 2D probabilities estimated for every parameter after the model-fit. The two dimensional 
figures can show the degeneracies between different parameters, for example how increasing the intensity $I$ of the
source galaxy and decreasing its effective radius $R_{Eff}$ lead to similar likelihoods and probabilities.

This PDF will be discussed more in the next tutorial.

The plot is labeled with short hand parameter names (e.g. `sersic_index` is mapped to the short hand 
parameter `n`). These mappings ate specified in the `config/notation.yaml` file and can be customized by users.

The superscripts of labels correspond to the name each component was given in the model (e.g. for the `Isothermal`
mass its name `mass` defined when making the `Model` above is used).
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
__Other Practicalities__

The following are examples of other practicalities which I will document fully in this example script in the future,
but so far have no found the time:

- `config`: The files in `autogalaxy_workspace/config` which control many aspects of how PyAutoGalaxy runs,
 including visualization, the non-linear search settings.

- `config/priors`: Folder containing the default priors on all model components.

- `results`: How to load the results of a model-fit from the output folder to a Python script or Jupyter notebook.

- `output.yaml`: What files are output to control file size.

__Wrap Up__

This tutorial has illustrated how to handle a number of practicalities that are key to performing model-fitting
effectively. These include:

- How to save results to your hard disk.

- How to navigate the output folder and examine model-fit results to assess quality.

- How to estimate the run-time of a model-fit before initiating it, and change settings to make it faster or run the
  analysis in parallel.
"""
