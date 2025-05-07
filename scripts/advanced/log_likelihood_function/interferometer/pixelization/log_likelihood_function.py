"""
__Log Likelihood Function: Pixelization__

This script provides a step-by-step guide of the **PyAutoLens** `log_likelihood_function` which is used to fit
`Interferometer` data with an inversion (specifically a `Rectangular` mesh and `Constant` regularization scheme`).

This script has the following aims:

 - To provide a resource that authors can include in papers using **PyAutoLens**, so that readers can understand the
 likelihood function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

 - To make inversions in **PyAutoLens** less of a "black-box" to users.

Accompanying this script is the `contributor_guide.py` which provides URL's to every part of the source-code that
is illustrated in this guide. This gives contributors a sequential run through of what source-code functions, modules and
packages are called when the likelihood is evaluated.

__Prerequisites__

The likelihood function of pixelizations is the most complicated likelihood function.

It is advised you read through the following two simpler likelihood functions first, which break down a number of the
concepts used in this script:

 - `interferometer/light_profile/log_likelihood_function.py` the likelihood function for a parametric light profile.
 - `imaging/linear_light_profile/log_likelihood_function.py` the likelihood function for a linear light profile, which
 introduces the linear algebra used for a pixelization but with a simpler use case.

This script repeats all text and code examples in the above likelihood function examples. It therefore can be used to
learn about the linear light profile likelihood function without reading other likelihood scripts.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autolens as al
import autolens.plot as aplt

"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the galaxy is evaluated using.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(80, 80), pixel_scales=0.05, radius=4.0
)

"""
__Dataset__

Load and plot the galaxy `Interferometer` dataset `simple` from .fits files, which we will fit 
with the model.

This includes the method used to Fourier transform the real-space image of the galaxy to the uv-plane and compare 
directly to the visibilities. We use a non-uniform fast Fourier transform, which is the most efficient method for 
interferometer datasets containing ~1-10 million visibilities. We will discuss how the calculation of the likelihood
function changes for different methods of Fourier transforming in this guide.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

"""
This guide uses in-built visualization tools for plotting. 

For example, using the `InterferometerPlotter` the dataset we perform a likelihood evaluation on is plotted.

The `subplot_dataset` displays the visibilities in the uv-plane, which are the raw data of the interferometer
dataset. These are what will ultimately be directly fitted in the Fourier space.

The `subplot_dirty_images` displays the dirty images of the dataset, which are the reconstructed images of visibilities
using an inverse Fourier transform to convert these to real-space. These dirty images are not the images we fit, but
visualization of the dirty images are often used in radio interferometry to show the data in a way that is more
interpretable to the human eye.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Over Sampling__

Over sampling evaluates a light profile using multiple samples of its intensity per image-pixel.

For simplicity, in previous likelihood function examples we disabled over sampling by setting `sub_size=1`. 

A full description of over sampling and how to use it is given in `autogalaxy_workspace/*/guides/over_sampling.py`.

Over sampling is used for the same purpose in a pixelization, whereby it uses multiple samples of a pixel to
perform the reconstruction via the pixelization. It uses an independent over sampling factor to the light profile
over sampling factor, called `over_sample_size_pixelization`.

However, for interferometer datasets, over sampling is not used in the pixelization (or for light profiles)
therefore it is implicitly set to 1 and can be ignored hereafter.

The notebook `log_likelihood_function/imaging/pixelization/with_over_sampling.ipynb` describes how the likelihood
function of a pixelization changes when over sampling is used.

__Masked Image Grid__

To perform galaxy calculations we define a 2D image-plane grid of (y,x) coordinates.

For light profiles these are given by `dataset.lp`, which is a uniform grid of (y,x) Cartesian coordinates
which have had the 3.0" circular mask applied.

A pixelization uses a separate grid of (y,x) coordinates, called `dataset.grids.pixelization`, which is
identical to the light profile grid but may of had a different over-sampling scale applied (but in this example
does not).

Each (y,x) coordinate coordinates to the centre of each image-pixel in the dataset, meaning that when this grid is
used to construct a pixelization there is a straight forward mapping between the image data and pixelization pixels.
"""
grid_plotter = aplt.Grid2DPlotter(grid=dataset.grids.pixelization)
grid_plotter.figure_2d()

"""
__Likelihood Setup: Galaxy__

We combine the pixelization into a single `Galaxy` object.

The galaxy includes the rectangular mesh and constant regularization scheme, which will ultimately be used
to reconstruct its star forming clumps.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.Rectangular(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=1.0),
)

galaxy = al.Galaxy(redshift=0.5, pixelization=pixelization)

"""
__Rectangular Mesh__

The pixelization is used to create the rectangular mesh which is used to reconstruct the galaxy.

The function below does this by overlaying the rectangular mesh over the masked image grid, such that the edges of
the rectangular mesh touch the ask grid's edges.
"""
grid_rectangular = al.Mesh2DRectangular.overlay_grid(
    shape_native=galaxy.pixelization.mesh.shape, grid=dataset.grids.pixelization
)

"""
The rectangular mesh will now be referred to interchangeably as the `source-plane`, to represent that it is a 
pixelization which will reconstruct a source of light,

Plotting the rectangular mesh shows that the source-plane has been discretized into a grid of rectangular pixels.

(To plot the rectangular mesh, we have to convert it to a `Mapper` object, which is described in the next likelihood 
step).

Below, we plot the rectangular mesh without the image-grid pixels (for clarity) and with them as black dots in order
to show how each set of image-pixels fall within a rectangular pixel.
"""
mapper_grids = al.MapperGrids(
    mask=real_space_mask,
    source_plane_data_grid=dataset.grids.pixelization,
    source_plane_mesh_grid=grid_rectangular,
)

mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

include = aplt.Include2D(mapper_source_plane_data_grid=False)
mapper_plotter = aplt.MapperPlotter(mapper=mapper, include_2d=include)
mapper_plotter.figure_2d()

include = aplt.Include2D(mapper_source_plane_data_grid=True)
mapper_plotter = aplt.MapperPlotter(mapper=mapper, include_2d=include)
mapper_plotter.figure_2d()

"""
__Image-Source Mapping__

We now combine grids computed above to create a `Mapper`, which describes how every masked image grid pixel maps to
every rectangular pixelization pixel. 

There are two steps in this calculation, which we show individually below.
"""
mapper_grids = al.MapperGrids(
    mask=real_space_mask,
    source_plane_data_grid=dataset.grids.pixelization,
    source_plane_mesh_grid=grid_rectangular,
)

mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

"""
The `Mapper` contains:

 1) `source_plane_data_grid`: the grid of masked (y,x) image-pixel coordinate centres (`dataset.grids.pixelization`).
 2) `source_plane_mesh_grid`: The rectangular mesh of (y,x) pixelization pixel coordinates (`grid_rectangular`).

We have therefore discretized the source-plane into a rectangular mesh, and can pair every image-pixel coordinate
with the corresponding rectangular pixel it lands in.

This pairing is contained in the ndarray `pix_indexes_for_sub_slim_index` which maps every image-pixel index to 
every pixelization pixel index.

In the API, the `pix_indexes` refers to the pixelization pixel indexes (e.g. rectangular pixel 0, 1, 2 etc.) 
and `sub_slim_index`  refers to the index of an image pixel (e.g. image-pixel 0, 1, 2 etc.). 

For example, printing the first ten entries of `pix_indexes_for_sub_slim_index` shows the first ten rectanfgular 
pixelization pixel indexes these image sub-pixels map too.
"""
pix_indexes_for_sub_slim_index = mapper.pix_indexes_for_sub_slim_index

print(pix_indexes_for_sub_slim_index[0:9])

"""
This array can be used to visualize how an input list of image-pixel indexes map to the rectangular pixelization.

It also shows that image-pixel indexing begins from the top-left and goes rightwards and downwards, accounting for 
all image-pixels which are not masked.
"""
include = aplt.Include2D(mapper_source_plane_data_grid=False)

visuals = aplt.Visuals2D(indexes=[list(range(2050, 2090))])

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper, visuals_2d=visuals, include_2d=include
)
mapper_plotter.subplot_image_and_mapper(
    image=dataset.dirty_image, interpolate_to_uniform=False
)

"""
The reverse mappings of pixelization pixels to image-pixels can also be used.
"""
visuals = aplt.Visuals2D(pix_indexes=[[200]])
mapper_plotter = aplt.MapperPlotter(
    mapper=mapper, visuals_2d=visuals, include_2d=include
)

mapper_plotter.subplot_image_and_mapper(
    image=dataset.dirty_image, interpolate_to_uniform=False
)

"""
__Mapping Matrix__

The `mapping_matrix` represents the image-pixel to pixelization-pixel mappings above in a 2D matrix. 

It has dimensions `(total_image_pixels, total_rectangular_pixels)`.

(A number of inputs are not used for the `Rectangular` mesh and are expanded upon in the `with_interpolation.ipynb`
log likelihood guide notebook).
"""
mapping_matrix = al.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for rectangular
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for rectangular
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=np.array(mapper.over_sampler.sub_fraction),
)

"""
A 2D plot of the `mapping_matrix` shows of all image-pixelization pixel mappings.

No row of pixels has more than one non-zero entry. It is not possible for two image pixels to map to the same 
pixelization pixel (meaning that there are no correlated pixels in the mapping matrix).
"""
plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
plt.show()
plt.close()

"""
Each column of the `mapping_matrix` can therefore be used to show all image-pixels it maps too. 

For example, above, we plotted all image-pixels of pixelization pixel 200 (as well as 202 and 204). We can extract all
image-pixel indexes of pixelization pixels 200 using the `mapping_matrix` and use them to plot the image of this
pixelization pixel (which corresponds to only values of zeros or ones).
"""
indexes_pix_200 = np.nonzero(mapping_matrix[:, 200])

print(indexes_pix_200[0])

array_2d = al.Array2D(values=mapping_matrix[:, 200], mask=dataset.mask)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
grid_plotter.figure_2d()

"""
__Transformed Mapping Matrix ($f$)__

Each pixelization pixel can therefore be thought of as an image (where all entries of this image are zeros and ones). 

However, for interferometer datasets we want to fit the visibilities in the uv-plane, not the image-plane. Therefore,
each image in the `mapping_matrix` must be transformed to the uv-plane via a Fourier transform, such that each
column in the `transformed_mapping_matrix` represents the visibilities in the uv-plane of each pixelization pixel.

This operation changes the dimensions of the mapping matrix, meaning the `transformed_mapping_matrix` has
dimensions `(total_image_pixels, total_visibilities)`. 

If the number of visibilities is large (e.g. 10^6) this matrix becomes extremely large and computationally expensive to 
store memory, meaning the `w_tilde` likelihood function, described in 
the `/log_likelihood_function/interferometer/`w_tilde.ipynb` notebook must be used instead.

The `transformed_mapping_matrix` is also complex, storing all entries of the visibilities after the NUFFT as real
and complex values.
"""
transformed_mapping_matrix = dataset.transformer.transform_mapping_matrix(
    mapping_matrix=mapping_matrix
)

"""
A 2D plot of the `transformed_mapping_matrix` shows all visibility-source pixel mappings.

Note how, unlike for the `mapping_matrix`, every row of image-pixels fully consists of non-zero entries. This
means the matrix is fully dense, making it even more difficult to store in memory for large datasets.

Below, we plot the real and imaginary components of the `transformed_mapping_matrix` separately.
"""
plt.imshow(
    transformed_mapping_matrix.real,
    aspect=(transformed_mapping_matrix.shape[1] / transformed_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.show()
plt.close()

plt.imshow(
    transformed_mapping_matrix.imag,
    aspect=(transformed_mapping_matrix.shape[1] / transformed_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.show()
plt.close()

"""
Each column of the `transformed_mapping_matrix` shows all visibilities it maps to after the NUFFT.
"""
indexes_pix_200 = np.nonzero(transformed_mapping_matrix[:, 200])

print(indexes_pix_200[0])

visibilities = al.Visibilities(visibilities=transformed_mapping_matrix[:, 200])

grid_plotter = aplt.Grid2DPlotter(grid=visibilities.in_grid)
grid_plotter.figure_2d()

"""
In Warren & Dye 2003 (https://arxiv.org/abs/astro-ph/0302587) the `transformed_mapping_matrix` is denoted $f_{ij}$
where $i$ maps over all $I$ source pixels and $j$ maps over all $J$ visibilities. 

For example: 

 - $f_{0, 2} = 0.3$ indicates that visibility number $2$ maps to pixelization pixel $0$ with a weight of $0.3$ after the NUFFT.

The indexing of the `mapping_matrix` is reversed compared to the notation of WD03 (e.g. visibilities
are the first entry of `mapping_matrix` whereas for $f$ they are the second index).
"""
print(f"Mapping between visibility 0 and rectangular pixel 2 = {mapping_matrix[0, 2]}")

"""
__Data Vector (D)__

To solve for the rectangular pixel fluxes we now pose the problem as a linear inversion.

This requires us to convert the `transformed_mapping_matrix` and our `data` and `noise map` into matrices of certain dimensions. 

The `data_vector`, $D$, is the first matrix and it has dimensions `(total_rectangular_pixels,)`.

In WD03 (https://arxiv.org/abs/astro-ph/0302587) and N15 (https://arxiv.org/abs/1412.7436) the data vector 
is give by: 

 $\vec{D}_{i} = \sum_{\rm  j=1}^{J}f_{ij}(d_{j})/\sigma_{j}^2 \, \, .$

Where:

 - $d_{\rm j}$ are the image-pixel data flux values.
 - $\sigma{\rm _j}^2$ are the statistical uncertainties of each image-pixel value.

$i$ maps over all $I$ source pixels and $j$ maps over all $J$ image pixels. 
"""
data_vector = (
    al.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=np.array(dataset.data),
        noise_map=np.array(dataset.noise_map),
    )
)

"""
$D$ describes which rectangular pixels trace to which visibilities, with associated weights, after the NUFFT. This 
ensures the reconstruction fully accounts for the NUFFT when fitting the data.

We can plot $D$ as a column vector:
"""
plt.imshow(
    data_vector.reshape(data_vector.shape[0], 1), aspect=10.0 / data_vector.shape[0]
)
plt.colorbar()
plt.show()
plt.close()

"""
The dimensions of $D$ are the number of source pixels.
"""
print("Data Vector:")
print(data_vector)
print(data_vector.shape)

"""
__Curvature Matrix (F)__

The `curvature_matrix` $F$ is the second matrix and it has 
dimensions `(total_rectangular_pixels, total_rectangular_pixels)`.

In WD03 / N15 (https://arxiv.org/abs/astro-ph/0302587) the curvature matrix is a 2D matrix given by:

 ${F}_{ik} = \sum_{\rm  j=1}^{J}f_{ij}f_{kj}/\sigma_{j}^2 \, \, .$

NOTE: this notation implicitly assumes a summation over $K$, where $k$ runs over all pixelization pixel indexes $K$.

Note how summation over $J$ runs over $f$ twice, such that every entry of $F$ is the sum of the multiplication
between all values in every two columns of $f$.

For example, $F_{0,1}$ is the sum of all visibility values in $f$ of source pixel 0 multiplied by
all visibility values of source pixel 1.

Visibilities are both real and complex values, and the `curvature_matrix` is computed separately for the real and
imaginary components of the visibilities and then summed together.
"""
real_curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=transformed_mapping_matrix.real,
    noise_map=dataset.noise_map.real,
)

imag_curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=transformed_mapping_matrix.imag,
    noise_map=dataset.noise_map.imag,
)

curvature_matrix = np.add(real_curvature_matrix, imag_curvature_matrix)

plt.imshow(curvature_matrix)
plt.colorbar()
plt.show()
plt.close()

"""
For $F_{ik}$ to be non-zero, this requires that the images of rectangular pixels $i$ and $k$ share at least one
image-pixel, which for visibilities after the NUFFT is always true for all $i$ and $k$.

For example, we can see a non-zero entry for $F_{100,101}$ and plotting their images
show overlap.
"""
source_pixel_0 = 0
source_pixel_1 = 1

print(curvature_matrix[source_pixel_0, source_pixel_1])

visibilities = al.Visibilities(
    visibilities=transformed_mapping_matrix[:, source_pixel_0],
)

grid_plotter = aplt.Grid2DPlotter(grid=visibilities.in_grid)
grid_plotter.figure_2d()

visibilities = al.Visibilities(
    visibilities=transformed_mapping_matrix[:, source_pixel_1],
)

grid_plotter = aplt.Grid2DPlotter(grid=visibilities.in_grid)
grid_plotter.figure_2d()

"""
The following chi-squared is minimized when we perform the inversion and reconstruct the galaxy:

$\chi^2 = \sum_{\rm  j=1}^{J} \bigg[ \frac{(\sum_{\rm  i=1}^{I} s_{i} f_{ij}) - d_{j}}{\sigma_{j}} \bigg]$

Where $s$ is the reconstructed pixel fluxes in all $I$ rectangular pixels.

The solution for $s$ is therefore given by (equation 5 WD03):

 $s = F^{-1} D$

We can compute this using NumPy linear algebra:
"""

# Because we are no using regularizartion (see below) it is common for the curvature matrix to be singular and lead
# to a LinAlgException. The loop below mitigates this -- you can ignore it as it is not important for understanding
# the PyAutoLens likelihood function.

for i in range(curvature_matrix.shape[0]):
    curvature_matrix[i, i] += 1e-8

reconstruction = np.linalg.solve(curvature_matrix, data_vector)

"""
We can plot this reconstruction -- it looks like a mess.

The pixelization pixels have noisy and unsmooth values, and it is hard to make out if a galaxy is even being 
reconstructed. 

In fact, the linear inversion is (over-)fitting noise in the image data, meaning this system of equations is 
ill-posed. We need to apply some form of smoothing on the reconstruction to avoid over fitting noise.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)

mapper_plotter.figure_2d(solution_vector=reconstruction, interpolate_to_uniform=False)

"""
__Regularization Matrix (H)__

Regularization adds a linear regularization term $G_{\rm L}$ to the $\chi^2$ we solve for giving us a new merit 
function $G$ (equation 11 WD03):

 $G = \chi^2 + \lambda \, G_{\rm L}$

where $\lambda$ is the `regularization_coefficient` which describes the magnitude of smoothness that is applied. A 
higher $\lambda$ will regularize the source more, leading to a smoother galaxy reconstruction.

Different forms for $G_{\rm L}$ can be defined which regularize the reconstruction in different ways. The 
`Constant` regularization scheme used in this example applies gradient regularization (equation 14 WD03):

 $G_{\rm L} = \sum_{\rm  i}^{I} \sum_{\rm  n=1}^{N}  [s_{i} - s_{i, v}]$

This regularization scheme is easier to express in words -- the summation goes to each rectangular pixelization pixel,
determines all rectangular pixels with which it shares a direct vertex (e.g. its neighbors) and penalizes solutions 
where the difference in reconstructed flux of these two neighboring pixels is large.

The summation does this for all rectangular pixels, thus it favours solutions where neighboring rectangular 
pixels reconstruct similar values to one another (e.g. it favours a smooth galaxy reconstruction).

We now define the `regularization matrix`, $H$, which allows us to include this smoothing when we solve for $s$. $H$
has dimensions `(total_rectangular_pixels, total_rectangular_pixels)`.

This relates to $G_{\rm L}$ as (equation 13 WD03):

 $H_{ik} = \frac{1}{2} \frac{\partial G_{\rm L}}{\partial s_{i} \partial s_{k}}$

$H$ has the `regularization_coefficient` $\lambda$ folded into it such $\lambda$'s control on the degree of smoothing
is accounted for.
"""
regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
    coefficient=galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.source_plane_mesh_grid.neighbors,
    neighbors_sizes=mapper.source_plane_mesh_grid.neighbors.sizes,
)

"""
We can plot the regularization matrix and note that:

 - non-zero entries indicate that two rectangular pixelization pixels are neighbors and therefore are regularized 
 with one another.

 - Zeros indicate the two rectangular pixels do not neighbor one another.

The majority of entries are zero, because the majority of rectangular pixels are not neighbors with one another.
"""
plt.imshow(regularization_matrix)
plt.colorbar()
plt.show()
plt.close()

"""
__F + Lamdba H__

$H$ enters the linear algebra system we solve for as follows (WD03 equation (12)):

 $s = [F + H]^{-1} D$
"""
curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

"""
__Galaxy Reconstruction (s)__

We can now solve the linear system above using NumPy linear algebra. 

Note that the for loop used above to prevent a LinAlgException is no longer required.
"""
reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

"""
By plotting this galaxy reconstruction we can see that regularization has lead us to reconstruct a smoother galaxy,
which actually looks like the star forming clumps in the galaxy imaging data! 

This also implies we are not over-fitting the noise.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)

mapper_plotter.figure_2d(solution_vector=reconstruction, interpolate_to_uniform=False)

"""
__Visibilities Reconstruction__

Using the reconstructed pixel fluxes we can map the reconstruction back to the image plane (via 
the `blurred mapping_matrix`) and produce a reconstruction of the image data.
"""
mapped_reconstructed_visibilities = (
    al.util.inversion_interferometer.mapped_reconstructed_visibilities_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        reconstruction=reconstruction,
    )
)

mapped_reconstructed_visibilities = al.Visibilities(
    visibilities=mapped_reconstructed_visibilities
)

grid_plotter = aplt.Grid2DPlotter(grid=mapped_reconstructed_visibilities.in_grid)
grid_plotter.figure_2d()


"""
__Likelihood Function__

We now quantify the goodness-of-fit of our pixelization galaxy reconstruction. 

We compute the `log_likelihood` of the fit, which is the value returned by the `log_likelihood_function`.

The likelihood function for galaxy modeling consists of five terms:

 $-2 \mathrm{ln} \, \epsilon = \chi^2 + s^{T} H s + \mathrm{ln} \, \left[ \mathrm{det} (F + H) \right] - { \mathrm{ln}} \, \left[ \mathrm{det} (H) \right] + \sum_{\rm  j=1}^{J} { \mathrm{ln}} \left [2 \pi (\sigma_j)^2 \right]  \, .$

This expression was first derived by Suyu 2006 (https://arxiv.org/abs/astro-ph/0601493) and is given by equation (19).
It was derived into **PyAutoLens** notation in Dye 2008 (https://arxiv.org/abs/0804.4002) equation (5).

We now explain what each of these terms mean.

__Chi Squared__

The first term is a $\chi^2$ statistic, which is defined above in our merit function as and is computed as follows:

 - `model_data` = `mapped_reconstructed_visibilities`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residuals`) ** 2.0 = ((`data` - `model_data`)**2.0)/(`variances`)
 - `chi_squared` = sum(`chi_squared_map`)

The chi-squared therefore quantifies if our fit to the data is accurate or not. 

High values of chi-squared indicate that there are many image pixels our model did not produce a good fit to the image 
for, corresponding to a fit with a lower likelihood.
"""
model_visibilities = mapped_reconstructed_visibilities

residual_map = dataset.data - model_visibilities


normalized_residual_map_real = (residual_map.real / dataset.noise_map.real).astype(
    "complex128"
)
normalized_residual_map_imag = (residual_map.imag / dataset.noise_map.imag).astype(
    "complex128"
)
normalized_residual_map = (
    normalized_residual_map_real + 1j * normalized_residual_map_imag
)


chi_squared_map_real = (residual_map.real / dataset.noise_map.real) ** 2
chi_squared_map_imag = (residual_map.imag / dataset.noise_map.imag) ** 2
chi_squared_map = chi_squared_map_real + 1j * chi_squared_map_imag


chi_squared_real = np.sum(chi_squared_map.real)
chi_squared_imag = np.sum(chi_squared_map.imag)
chi_squared = chi_squared_real + chi_squared_imag

print(chi_squared)

"""
The `chi_squared_map` indicates which regions of the image we did and did not fit accurately.
"""
chi_squared_map = al.Visibilities(visibilities=chi_squared_map)

grid_plotter = aplt.Grid2DPlotter(grid=chi_squared_map.in_grid)
grid_plotter.figure_2d()

"""
__Regularization Term__

The second term, $s^{T} H s$, corresponds to the $\lambda $G_{\rm L}$ regularization term we added to our merit 
function above.

This is the term which sums up the difference in flux of all reconstructed rectangular pixels, and reduces the 
likelihood of solutions where there are large differences in flux (e.g. the galaxy is less smooth and more likely to be 
overfitting noise).

We compute it below via matrix multiplication, noting that the `regularization_coefficient`, $\lambda$, is built into 
the `regularization_matrix` already.
"""
regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

print(regularization_term)

"""
__Complexity Terms__

Up to this point, it is unclear why we chose a value of `regularization_coefficient=1.0`. 

We cannot rely on the `chi_squared` and `regularization_term` above to optimally choose its value, because increasing 
the `regularization_coefficient` smooths the solution more and therefore:

 - Decreases `chi_squared` by fitting the data worse, producing a lower `log_likelihood`.

 - Increases the `regularization_term` by penalizing the differences between source pixel fluxes more, again reducing
 the inferred `log_likelihood`.

If we set the regularization coefficient based purely on these two terms, we would set a value of 0.0 and be back where
we started over-fitting noise!

The terms $\left[ \mathrm{det} (F + H) \right]$ and $ - { \mathrm{ln}} \, \left[ \mathrm{det} (H) \right]$ address 
this problem. 

They quantify how complex the reconstruction is, and penalize solutions where *it is more complex*. Reducing 
the `regularization_coefficient` makes the galaxy reconstruction more complex (because a galaxy that is 
smoothed less uses more flexibility to fit the data better).

These two terms therefore counteract the `chi_squared` and `regularization_term`, so as to attribute a higher
`log_likelihood` to solutions which fit the data with a more smoothed and less complex source (e.g. one with a higher 
`regularization_coefficient`).

In **HowToGalaxy** -> `chapter 4` -> `tutorial_4_bayesian_regularization` we expand on this further and give a more
detailed description of how these different terms impact the `log_likelihood_function`. 
"""
log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

print(log_curvature_reg_matrix_term)
print(log_regularization_matrix_term)

"""
__Noise Normalization Term__

Our likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel.

The final term ins the likelihood function is therefore a `noise_normalization` term, which consists of the sum
of the log of every noise-map value squared. 

Given the `noise_map` is fixed, this term does not change during the galaxy modeling process and has no impact on the 
model we infer.
"""
noise_normalization_real = np.sum(np.log(2 * np.pi * dataset.noise_map.real**2.0))
noise_normalization_imag = np.sum(np.log(2 * np.pi * dataset.noise_map.imag**2.0))
noise_normalization = noise_normalization_real + noise_normalization_imag

"""
__Calculate The Log Likelihood__

We can now, finally, compute the `log_likelihood` of the galaxy model, by combining the five terms computed above using
the likelihood function defined above.
"""
log_evidence = float(
    -0.5
    * (
        chi_squared
        + regularization_term
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )
)

print(log_evidence)

"""
__Fit__

This process to perform a likelihood function evaluation performed via the `FitInterferometer` object.
"""
galaxies = al.Galaxies(galaxies=[galaxy])

fit = al.FitInterferometer(
    dataset=dataset,
    galaxies=galaxies,
    settings_inversion=al.SettingsInversion(
        use_w_tilde=False, use_border_relocator=True
    ),
)
fit_log_evidence = fit.log_evidence
print(fit_log_evidence)

"""
__Galaxy Modeling__

To fit a galaxy model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `nautilus` (https://github.com/joshspeagle/nautilus)
but **PyAutoLens** supports multiple MCMC and optimization algorithms. 

__Wrap Up__

We have presented a visual step-by-step guide to the pixelization likelihood function.

There are a number of other inputs features which slightly change the behaviour of this likelihood function, which
are described in additional notebooks found in this package. In brief, these describe:

 - **Over Sampling**: Oversampling the image grid into a finer grid of sub-pixels, which are all individually 
 paired fractionally with each rectangular pixel.

 - **Source-plane Interpolation**: Using bilinear interpolation on the rectangular pixelization to pair each 
 image (sub-)pixel to multiple rectangular pixels with interpolation weights.

 - **Luminosity Weighted Regularization**: Using an adaptive regularization coefficient which adapts the level of 
 regularization applied to the galaxy based on its luminosity.
"""
