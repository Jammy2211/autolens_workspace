"""
SLaM: Multi Wavelength Simultaneous
===================================

This example shows how to use the SLaM pipeline to fit a lens dataset at multiple wavelengths simultaneously.

Simultaneous multi-dataset fits are currently built into the SLaM pipeline without user input or customization.
Therefore, as long as lists of `Analysis` objects are created, summed and passed to the SLaM pipelines, the analysis
will fit every dataset simultaneously and it will adapt the model as follows:

- Sub-pixel offsets between the datasets are fully modeled as free parameters in each stage of the pipeline, assuming
  broad uniform priors for every step. This is because the precision of a lens model can often be less than the
  requirements on astrometry.

- The regularization parameters are free for every dataset in the `source_pix[1]` and `source_pix[2]` stages. This is because
  the source morphology can be different between datasets, and the regularization scheme adapts to this.

- From the `light_lp` stage onwards, the regularization scheme for each dataset is different fixed to that inferred
  for the `source_pix[2]` stage.

Simultaneous fitting SLaM pipelines are not designed for customization, for example changing the model from the
set up above. This is because we are still figuring out the best way to perform multi-wavelength modeling, but have
so far figured the above settings are important.

If you need customization of the model or pipeline, you should pick apart the SLaM pipeline and customize
them as you see fit.

__Preqrequisites__

Before reading this script, you should have familiarity with the following key concepts:

- **Multi**: The `autolens_workspace/*/advanced/multi` package describes many different ways that multiple datasets
  can be modeled in a single analysis.

__This Script__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this SLaM modeling
script  fits `Imaging` dataset  of a strong lens system where in the final model:

 - The lens galaxy's light is a bulge with Multiple Gaussian Expansion (MGE) light profile.
 - The lens galaxy's total mass distribution is an `PowerLaw` plus an `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.

This modeling script uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`

__Start Here Notebook__

If any code in this script is unclear, refer to the `chaining/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import os
import sys
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())
import slam

"""
__Dataset__ 

Load, plot and mask the `Imaging` data.
"""
dataset_waveband_list = ["g", "r"]
pixel_scale_list = [0.12, 0.08]

dataset_name = "lens_sersic"
dataset_main_path = path.join("dataset", "multi", "imaging", dataset_name)
dataset_path = path.join(dataset_main_path, dataset_name)


dataset_list = []

for dataset_waveband, pixel_scale in zip(dataset_waveband_list, pixel_scale_list):
    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_main_path, f"{dataset_waveband}_data.fits"),
        noise_map_path=path.join(
            dataset_main_path, f"{dataset_waveband}_noise_map.fits"
        ),
        psf_path=path.join(dataset_main_path, f"{dataset_waveband}_psf.fits"),
        pixel_scales=pixel_scale,
    )

    mask_radius = 3.0

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    dataset_list.append(dataset)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=path.join("slam", "multi", "simultaneous"),
    unique_tag=dataset_name,
    info=None,
    number_of_cores=4,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__SOURCE LP PIPELINE__

The SOURCE LP PIPELINE fits an identical to the `start_here.ipynb` example, except:

 - The model includes the (y,x) offset of each dataset relative to the first dataset, which is added to every
  `AnalysisImaging` object such that there are 2 extra parameters fitted for each dataset.
"""
analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

analysis = sum(analysis_list)

# Lens Light

centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

total_gaussians = 30
gaussian_per_basis = 2

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    bulge_gaussian_list += gaussian_list

lens_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

# Source Light

centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

total_gaussians = 30
gaussian_per_basis = 1

log10_sigma_list = np.linspace(-3, np.log10(1.0), total_gaussians)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    bulge_gaussian_list += gaussian_list

source_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

source_lp_result = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=lens_bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
    dataset_model=af.Model(al.DatasetModel),
)

"""
__SOURCE PIX PIPELINE__

The SOURCE PIX PIPELINE uses two searches to initialize a robust model for the `Pixelization` that
reconstructs the source galaxy's light. 

This pixelization adapts its source pixels to the morphology of the source, placing more pixels in its 
brightest regions. To do this, an "adapt image" is required, which is the lens light subtracted image meaning
only the lensed source emission is present.

The SOURCE LP Pipeline result is not good enough quality to set up this adapt image (e.g. the source
may be more complex than a simple light profile). The first step of the SOURCE PIX PIPELINE therefore fits a new
model using a pixelization to create this adapt image.

The first search, which is an initialization search, fits an `Overlay` image-mesh, `Delaunay` mesh 
and `AdaptiveBrightnessSplit` regularization.

__Adapt Images / Image Mesh Settings__

If you are unclear what the `adapt_images` and `SettingsInversion` inputs are doing below, refer to the 
`autolens_workspace/*/imaging/advanced/chaining/pix_adapt/start_here.py` example script.

__Settings__:

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
positions_likelihood = source_lp_result[0].positions_likelihood_from(
    factor=3.0, minimum_threshold=0.2
)

analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_image_maker=al.AdaptImageMaker(result=result),
        positions_likelihood=positions_likelihood,
    )
    for result in source_lp_result
]

analysis = sum(analysis_list)

source_pix_result_1 = slam.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.Delaunay,
    dataset_model=af.Model(al.DatasetModel),
)

"""
__SOURCE PIX PIPELINE 2 (with lens light)__

The second search, which uses the mesh and regularization used throughout the remainder of the SLaM pipelines,
fits the following model:

- Uses a `Hilbert` image-mesh. 

- Uses a `Delaunay` mesh.

 - Uses an `AdaptiveBrightnessSplit` regularization.

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
 SOURCE PIX PIPELINE.

The `Hilbert` image-mesh and `AdaptiveBrightness` regularization adapt the source pixels and regularization weights
to the source's morphology.

Below, we therefore set up the adapt image using this result.
"""
analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_image_maker=al.AdaptImageMaker(result=result),
        settings_inversion=al.SettingsInversion(
            image_mesh_min_mesh_pixels_per_pixel=3,
            image_mesh_min_mesh_number=5,
            image_mesh_adapt_background_percent_threshold=0.1,
            image_mesh_adapt_background_percent_check=0.8,
        ),
    )
    for result in source_pix_result_1
]

analysis = sum(analysis_list)

source_pix_result_2 = slam.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    image_mesh=al.image_mesh.Hilbert,
    mesh=al.mesh.Delaunay,
    regularization=al.reg.AdaptiveBrightnessSplit,
    dataset_model=af.Model(al.DatasetModel),
)

"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE LP PIPELINE.
In this example it:

 - Uses a multi Gaussian expansion with 2 sets of 30 Gaussians for the lens galaxy's light. [6 Free Parameters].

 - Uses an `Isothermal` mass model with `ExternalShear` for the lens's total mass distribution [fixed from SOURCE PIX PIPELINE].

 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift and source redshift of the SOURCE PIPELINE through to the MASS PIPELINE [fixed values].   
"""
analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_image_maker=al.AdaptImageMaker(result=result),
        raise_inversion_positions_likelihood_exception=False,
    )
    for result in source_pix_result_1
]

analysis = sum(analysis_list)
total_gaussians = 30
gaussian_per_basis = 2

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = gaussian_list[0].centre.centre_0
        gaussian.centre.centre_1 = gaussian_list[0].centre.centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    bulge_gaussian_list += gaussian_list

lens_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

light_result = slam.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
    lens_disk=None,
    dataset_model=af.Model(al.DatasetModel),
)

"""
__MASS TOTAL PIPELINE__

The MASS TOTAL PIPELINE uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIX PIPELINE to initialize the model priors and the lens 
light model of the LIGHT LP PIPELINE. 

In this example it:

 - Uses a linear Multi Gaussian Expansion bulge [fixed from LIGHT LP PIPELINE].

 - Uses an `PowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
 PARAMETRIC PIPELINE + centre unfixed from (0.0, 0.0)].

 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift and source redshift of the SOURCE PIPELINE through to the MASS TOTAL PIPELINE.

__Settings__:

 - adapt: We may be using adapt features and therefore pass the result of the SOURCE PIX PIPELINE to use as the
 hyper dataset if required.

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
positions_likelihood = source_lp_result[0].positions_likelihood_from(
    factor=3.0, minimum_threshold=0.2
)

analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_image_maker=al.AdaptImageMaker(result=result),
        positions_likelihood=positions_likelihood,
    )
    for result in source_lp_result
]

analysis = sum(analysis_list)

mass_result = slam.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
    dataset_model=af.Model(al.DatasetModel),
)

"""
__Output__

The SLaM pipeline above outputs the model-fitting results to the `output` folder of the workspace, which includes
the usual model results, visualization, and .json files.

As described in the `autolens_workspace/*/results` package there is an API for loading these results from hard disk
to Python, for example so they can be manipulated in a Juypter notebook.

However, it is also often useful to output the results to the dataset folder of each lens in standard formats, for
example images of the lens and lensed source in .fits or visualization outputs like .png files. This makes transferring
the results more portable, especially if they are to be used by other people.

The `slam_util` module provides convenience methods for outputting many results to the dataset folder, we
use it below to output the following results:

 - Images of the model lens light, lensed source light and source reconstruction to .fits files.
 - A text `model.results` file containing the lens model parameter estimates.
 - A subplot containing the fit in one row, which is output to .png.
 - A subplot of the source reconstruction in the source plane in one row, which is output to .png.

"""
slam.slam_util.output_model_to_fits(
    output_path=path.join(dataset_path, "model"),
    result=mass_result,
    model_lens_light=True,
    model_source_light=True,
    source_reconstruction=True,
)

slam.slam_util.output_model_results(
    output_path=path.join(dataset_path, "model"),
    result=mass_result,
    filename="model.results",
)

slam.slam_util.output_fit_multi_png(
    output_path=dataset_path,
    result_list=[mass_result],
    filename="sie_fit",
)

slam.slam_util.output_source_multi_png(
    output_path=dataset_path,
    result_list=[mass_result],
    filename="source_reconstruction",
)

"""
Finish.
"""
