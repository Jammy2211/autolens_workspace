from os import path
import autolens as al
import autolens.plot as aplt

"""
In this example, we demonstrate how the appearance of figures in PyAutoLens can be customized, using an image of the 
strong lens slacs1430+4105.

The customization functions demonstrated in this example are generic to any 2D arrays of dataset_type, and can 
therefore be applied to the plotting of noise-maps, PSF`s, residual-maps, chi-squared-maps, etc. Many of the options
can also be applied to the plotting of other data structures, for example `Grid`'s and `Mappers``..
"""

"""
We have included the .fits dataset_type required for this example in the directory
`autolens_workspace/output/dataset/imaging/slacs1430+4105/`.
"""

dataset_path = path.join("dataset", "slacs" "slacs1430+4105")
image_path = path.join(dataset_path, "image.fits")

"""
Now, lets load this arrays as an `Array` object. which is an ordinary NumPy ndarray but includes additional 
functionality and attributes which are used during plotter. For example, it includes a pixel scale which converts the 
axes of the arrays from pixels to arc-second coordinates (the vast majority of image-like objects you encourter in 
PyAutoLens, residual-maps, images, noise-maps, etc, are `Array``.!).
"""

image = al.Array.from_fits(file_path=image_path, hdu=0, pixel_scales=0.03)

"""
**Plot**

We can use an `Array` plotter to plot the `Array`. We customize the plotters as follows:
"""

aplt.Array(array=image)

"""
PyAutoLens`s visualization tools have a wide range of tools that enable one to customize the image that is plotted. 
we'll cover each one by one ini this example script, noting that they follow the Matplotlib API as closely as possible.

To customize a figure, note below how we create a `Plotter` object and pass that to the method `aplt.Array`.
"""

"""
**Figure**

The `Figure` object customizes the size of the figure the `Array` is plotted using. Below, we:

1) Plot the `Array` using a figure whose size has increased to (12, 12) from the default (7,7).
2) Adjust the aspect ratio to 0.5, making the plot appear rectangular. This overwrites the default aspect input, 
   `square`, which plots the figure as a square with aspect ratio of 1.0.

Note how carefully we have chosen the **PyAutoLens** default values of all Matplotlib objects to ensure the misaligned
colorbar and weird figure shape don't impact most default visualization!
"""

plotter = aplt.Plotter(figure=aplt.Figure(figsize=(12, 12), aspect=0.5))

aplt.Array(array=image, plotter=plotter)

"""
**Units**

The `Units` object customizes the units of the y and x axes the `Array` is plotted using. Below, we:

1) Use scaled units to plot the y and x axis labels of the `Array`. Its scaled coordinates are its coordinates in 
   arc-seconds, converted from pixels using its *pixel_scales* attribute. Switching this to `False` will plot the axes
    in pixel units.
2) Input a conversion factor of 10.0, which multiplies the y and x coordinates (compared to the figure above) by 10.0.

This method is used to plot figures in units of kiloparsec converted from arcseconds, as shown by the `in_kpc`
input below. 

An `Array` does not know its `cosmology` and thus this conversion is not possible, however when plotting objects
such as a `Galaxy` or `Tracer` which have a redshift and cosmology this option automatically changes the units to kpc.
"""

plotter = aplt.Plotter(
    units=aplt.Units(use_scaled=True, conversion_factor=10.0, in_kpc=False)
)

aplt.Array(array=image, plotter=plotter)

"""
**ColorMap**

The `ColorMap` object customizes the colormap of the image and scales of the normalization of the plot. Below we:

1) Change the colormap color scheme to `coolwar` from the default `jet`.
2) Specify a symmetric logarithmic color map (default is `linear`, `log` can also be used) with manually input values 
   for the minimum and maximum values of this color map.
3) Specify the linthresh and linscale parameters of symmetric log colormap (see 
https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.colors.SymLogNorm.html)
"""

plotter = aplt.Plotter(
    cmap=aplt.ColorMap(
        cmap="coolwarm",
        norm="symmetric_log",
        norm_min=0.1,
        norm_max=0.8,
        linthresh=0.05,
        linscale=0.1,
    )
)

aplt.Array(array=image, plotter=plotter)

"""
**ColorBar**

The `ColorBar` object customizes the colorbar. Below we:

1) Increase the ticksize of the colorbar to 20 from the default of 10 so the tick fontsize is larger.
2) Change fraction / pad values of the colorbar (from defaults of 0.047 / 0.01) which change the size and shape of the
   colorbar.
3) Manually override the colorbar labels with new values (tick_labels), with their location on the colorbar running 
   from 0 -> 1 (tick_values).
"""

plotter = aplt.Plotter(
    cb=aplt.ColorBar(
        ticksize=20,
        fraction=0.1,
        pad=0.2,
        tick_labels=[1.0, 2.0, 3.0],
        tick_values=[0.2, 0.4, 0.6],
    )
)

aplt.Array(array=image, plotter=plotter)

"""
**Ticks**

The `Ticks` object customizes the figure ticks. Below we:

1) Increase the size of the y and x ticks from 16 to 24.
2) Manually override the tick labels with new values.
"""

plotter = aplt.Plotter(
    ticks=aplt.Ticks(
        ysize=24, xsize=24, y_manual=[1.0, 2.0, 3.0, 4.0], x_manual=[4.0, 5.0, 6.0, 7.0]
    )
)

aplt.Array(array=image, plotter=plotter)

"""
**Labels**

The `Labels` object customizes the figure labels. Below we:

1) Manually set the figure title, y and x labels.
2) Manually set the title, y and x label font sizes.
"""

plotter = aplt.Plotter(
    labels=aplt.Labels(
        title="SLACS1430+4105 Image",
        yunits="Hi",
        xunits="Hello",
        titlesize=15,
        ysize=10,
        xsize=20,
    )
)

aplt.Array(array=image, plotter=plotter)

"""
**Output**

The `Output` object allows us to output a figure to hard-disc.

1) Output the figure to the folder `autolens_workspace/examples/plot/plots/array.png
"""

plotter = aplt.Plotter(
    output=aplt.Output(
        path=path.join("examples", "plot", "plots"), filename="array", format="png"
    )
)

aplt.Array(array=image, plotter=plotter)
