"""
Customize: Modeling Positions
=============================

Before fitting a strong lens, we can manually specify a grid of image-plane coordinates corresponding to the multiple
images of the lensed source-galaxy(s). During the model-fit, **PyAutoLens** will check that these coordinates trace
within a specified arc-second threshold of one another in the source-plane. If they do not meet this threshold, the
mass model is discarded and a new sample is generated by the non-linear search.

__Advantages__

The model-fit is faster, as the non-linear search avoids regions of parameter space where the mass-model
is clearly not accurate. Removing these unphysical solutions may also mean that the global-maximum solution is inferred
instead of a local-maxima, given that removing unphysical mass models makes non-linear parameter space less complex.

__Disadvantages__

If the positions are inaccurate or threshold is set too low, one may inadvertantly remove the correct
mass model!

The positions are associated with the `Imaging` dataset and they are loaded from a `positions.json` file which is in the
same folder as the dataset itself. To create this file, we used a GUI to `draw on` the positions with our mouse. This
GUI can be found in the script:

 `autolens_workspace/*/data_preparation/imaging/gui/positions.py`

If you wish to use positions for modeling your own lens data, you should use this script to draw on the positions of
every lens in you dataset.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
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
__Dataset + Masking__

Load and plot the strong lens dataset `simple__no_lens_light` via .fits files
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
__Positions__

The positions are loaded from a `positions.json` file which is in the same folder as the dataset itself. 

To create this file, we used a GUI to `draw on` the positions with our mouse. This GUI can be found in the 
script `autolens_workspace/*/data_preparation/imaging/gui/positions.py`
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)

visuals = aplt.Visuals2D(mask=mask, positions=positions)
dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.subplot_dataset()

"""
Alternatively, the positions can be specified manually in the modeling script script using the `Grid2DIrregular`object.
 
Below, we specify a list of (y,x) coordinates (that are not on a uniform or regular grid) which correspond to the 
arc-second (y,x) coordinates ot he lensed source's brightest pixels.
"""
positions = al.Grid2DIrregular(
    [(0.4, 1.6), (1.58, -0.35), (-0.43, -1.59), (-1.45, 0.2)]
)

visuals = aplt.Visuals2D(mask=mask, positions=positions)
dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.subplot_dataset()

"""
__Model + Search__ 

The code below performs the normal steps to set up a model-fit. We omit comments of this code as you should be 
familiar with it and it is not specific to this example!
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


search = af.Nautilus(
    path_prefix=path.join("imaging", "customize"),
    name="positions",
    unique_tag=dataset_name,
)

"""
__Position Thresholding__

Unlike other example scripts, we also pass the `AnalysisImaging` object below a `PositionsLHPenalty` object, which
includes the positions we loaded above, alongside a `threshold`.

This is because `Inversion`'s suffer a bias whereby they fit unphysical lens models where the source galaxy is 
reconstructed as a demagnified version of the lensed source. 

To prevent these solutions biasing the model-fit we specify a `position_threshold` of 0.5", which requires that a 
mass model traces the four (y,x) coordinates specified by our positions (that correspond to the brightest regions of the 
lensed source) within 0.5" of one another in the source-plane. If this criteria is not met, a large penalty term is
added to likelihood that massive reduces the overall likelihood. 

This ensures the unphysical solutions that bias a pixelization have a lower likelihood that the physical solutions
we desire. Furthermore, the penalty term reduces as the positions trace closer in the source-plane, ensuring Nautilus
will converges towards an accurate mass model. It does this very fast, as ray-tracing positions is computationally 
cheap. 

The threshold of 0.5" is large. For an accurate lens model we would anticipate the positions trace within < 0.01" of
one another. The high threshold ensures only the initial mass models at the start of the fit are resampled.
"""
positions_likelihood = al.PositionsLHPenalty(positions=positions, threshold=0.3)

"""
__Analysis__
"""
analysis = al.AnalysisImaging(
    dataset=dataset, positions_likelihood=positions_likelihood
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Because the `AnalysisImaging` was passed positions, many unphysical mass models will be discarded, speeding up the
model-fit.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

In this example, we used positional information about the lensed source galaxy's multiple images to speed up our
model-fit and make it more robust. Advanced **PyAutoLens** use will introduce a technique called non-linear search
chaining, which performs a model-fit as a sequence of many non-linear searches. This includes a feature called 
'automatic positions' which automatically computes the positions and updates the positions threshold, based on a 
previous mass model. This is detailed in the script:

 `autolens_workspace/imaging/notebooks/chaining/examples/parametric_to_pixelization.py`.
"""
