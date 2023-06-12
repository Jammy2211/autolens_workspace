"""
Tutorial 0: Visualization
=========================

In this tutorial, we quickly cover visualization in **PyAutoLens** and make sure images display clealry in your
Jupyter notebook and on your computer screen.
"""
# %matplotlib inline

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Directories__

**PyAutoLens assumes** the working directory is `autolens_workspace` on your hard-disk. This is so that it can:
 
 - Load configuration settings from config files in the `autolens_workspace/config` folder.
 - Load example data from the `autolens_workspace/dataset` folder.
 - Output the results of models fits to your hard-disk to the `autolens/output` folder. 
    
At the top of every tutorial notebook, you'll see the following cell. This cell uses the project `pyprojroot` to
locate the path to the workspace on your computer and use it to set the working directory of the notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

"""
If the printed working directory does not match the workspace path on your computer, you can manually set it
as follows (the example below shows the path I would use on my laptop. The code is commented out so you do not
use this path in this tutorial!
"""
# workspace_path = "/Users/Jammy/Code/PyAuto/autolens_workspace"
# #%cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

"""
__Dataset__

The `dataset_path` specifies where the dataset is located, which is the 
directory `autolens_workspace/dataset/imaging/simple__no_lens_light`.

There are many example simulated images of strong lenses in this directory that will be used throughout the 
**HowToLens** lectures.
"""
dataset_path = path.join("dataset", "imaging", "simple__no_lens_light")

"""
We now load this dataset from .fits files and create an instance of an `imaging` object.
"""
dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

"""
We can plot an image as follows:
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(data=True)

"""
__Plot Customization__

Does the figure display correctly on your computer screen? 

If not, you can customize a number of matplotlib setup options using a `MatPlot2D` object, which 
wraps the `matplotlib` methods used to display the image.

(For example, the `Figure` class wraps the `matplotlib` method `plt.figure()`, whereas the `YTicks` class wraps
`plt.yticks`).
"""
mat_plot = aplt.MatPlot2D(
    figure=aplt.Figure(figsize=(7, 7)),
    yticks=aplt.YTicks(fontsize=8),
    xticks=aplt.XTicks(fontsize=8),
    title=aplt.Title(fontsize=12),
    ylabel=aplt.YLabel(fontsize=6),
    xlabel=aplt.XLabel(fontsize=6),
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.figures_2d(data=True)

"""
Many matplotlib options can be customized, but for now we're only concerned with making sure figures display clear in 
your Jupyter Notebooks. Nevertheless, a comprehensive API reference guide of all `matplotlib` wrappers and methods can 
be found in the `autolens_workspace/plot` package. You should check this out once you are more familiar with 
**PyAutoLens**.

Ideally, we would not specify a new `MatPlot2D` object every time we plot an image, especially as you would be 
changing the same option to the same values every time (e.g. the `figsize`) to make the figure display correctly over 
and over again. Fortunately, the default values can be fully customized via the config files.

Checkout the `mat_wrap.yaml`, `mat_wrap_1d.yaml` and `mat_wrap_2d.yaml` files 
in `autolens_workspace/config/visualize/mat_wrap`.

All default matplotlib values are here. There are a lot of entries, so lets focus on whats important for displaying 
figures:

 - mat_wrap.yaml -> Figure -> figure: -> figsize
 - mat_wrap.yaml -> YLabel -> figure: -> fontsize
 - mat_wrap.yaml -> XLabel -> figure: -> fontsize
 - mat_wrap.yaml -> TickParams -> figure: -> labelsize
 - mat_wrap.yaml -> YTicks -> figure: -> labelsize
 - mat_wrap.yaml -> XTicks -> figure: -> labelsize

Don't worry about all the other files or options listed for now, as they`ll make a lot more sense once you are familiar 
with **PyAutoLens**.

If you had to change any of the above settings to get the figures to display clearly, you should update their values
in the corresponding config files above (you will need to reset your Jupyter notebook server for these changes to 
take effect, so make sure you have the right values using the `mat_plot_2d` object in the cell above beforehand!).

__Subplots__

In addition to plotting individual `figures`, **PyAutoLens** can also plot `subplots` which are again customized via
the `mat_plot` objects.

Lets plot a subplot of our `Imaging` data:
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
Again, we can customize this using a `MatPlot2D`.
"""
mat_plot = aplt.MatPlot2D(
    figure=aplt.Figure(figsize=(7, 7)),
    yticks=aplt.YTicks(fontsize=8),
    xticks=aplt.XTicks(fontsize=8),
    title=aplt.Title(fontsize=12),
    ylabel=aplt.YLabel(fontsize=6),
    xlabel=aplt.XLabel(fontsize=6),
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()

"""
Again, you can customize the default appearance of subplots by editing the config files above, but now editing the
corresponding entries under the subplot: headers.

We can also customize what is included in a plot. for example whether the origin of the coordinate system appears on 
the image:
"""
include = aplt.Include2D(origin=True)
dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, mat_plot_2d=mat_plot, include_2d=include
)
dataset_plotter.figures_2d(data=True)

include = aplt.Include2D(origin=False)
dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, mat_plot_2d=mat_plot, include_2d=include
)
dataset_plotter.figures_2d(data=True)

"""
__Wrap Up__

Throughout the **HowToLens** lectures you'll see lots more objects that can be included on figures and subplots.

Just like the matplotlib setup, you can customize what does and does not appear on figures by default using the 
config file `autolens_workspace/config/visualize/include.yaml`

Great! Hopefully, visualization in **PyAutoLens** is displaying nicely for us to get on with the **HowToLens** 
lecture series.
"""
