import autofit as af
import autolens as al
import autolens.plot as aplt
from tools.preprocessing.loading_and_preparing_data import simulate_data

# To modeldata with PyAutoLens you first need to ensure it is in a format suitable for lens modeling. This tutorial
# takes you throughdata preparation, introducing PyAutoLens's built in tools that convertdata to a suitable format.

# First, lets setup the path to our current working directory. I recommend you use the 'autolens_workspace' directory
# and place simulator in the 'autolens_workspace/simulator' directory.

# for this tutorial, we'll use the 'autolens_workspace/tools/loading_and_preparing_data/dataset' directory. The folder
# 'simulator' contains example simulator we'll use in this tutorial.

path = (
    "path/to/AutoLens/autolens_workspace/tools/loading_and_preparing_data/"
)  # <----- You must include this slash on the end

path = "/autolens_workspace/tools/preprocessing/loading_and_preparing_data/"

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=path, folder_names=["dataset"]
)

# This populates the 'simulator' path with example simulated imaging datasets.

simulate_data.simulate_all_imaging(dataset_path=dataset_path)

# First, lets load a dataset using the 'load_imaging_from_fits' function of the imaging module. This
# dataset represents a good simulator-reduction that conforms to the formatting standards I describe in this tutorial!
imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=dataset_path, folder_names=["imaging"]
)

imaging = al.imaging.from_fits(
    image_path=imaging_path + "image.fits",
    noise_map_path=imaging_path + "noise_map.fits",
    psf_path=imaging_path + "psf.fits",
    pixel_scales=0.1,
)

aplt.imaging.subplot_imaging(imaging=imaging)

# If your dataset comes in one .fits file spread across multiple hdus you can specify the hdus of each image instead.
imaging = al.imaging.from_fits(
    image_path=imaging_path + "multiple_hdus.fits",
    image_hdu=0,
    noise_map_path=imaging_path + "multiple_hdus.fits",
    noise_map_hdu=1,
    psf_path=imaging_path + "multiple_hdus.fits",
    psf_hdu=2,
    pixel_scales=0.1,
)

aplt.imaging.subplot_imaging(imaging=imaging)

# Lets think about the format of our dataset. There are numerous reasons why the image we just looked at is a good dataset
# for lens modeling. I strongly recommend you reduce your dataset to conform to the standards discussed below - it'll make
# your time using PyAutoLens a lot simpler.

# However, you may not have access to the dataset-reduction tools that made the dataset, so we've included a number of
# in-built functions in PyAutoLens to convert the dataset to a good format for you. However, your life will be much easier
# if you can just reduce it this way in the first place!

# 1) Brightness unit_label - the image's flux and noise-map values are in unit_label of electrons per second (not electrons,
#    counts, ADU's etc.). Although PyAutoLens can technically perform an analysis using other unit_label, the default
#    setup assume the image is in electrons per second (e.g. the priors on light profile image and
#    regularization coefficient). Thus, images not in electrons per second should be converted!

# Lets look at an image that is in unit_label of counts - its easy to tell because the peak values are in the 1000's or
# 10000's.
imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=dataset_path, folder_names=["imaging_in_counts"]
)

imaging_in_counts = al.imaging.from_fits(
    image_path=imaging_path + "image.fits",
    pixel_scales=0.1,
    noise_map_path=imaging_path + "noise_map.fits",
    psf_path=imaging_path + "psf.fits",
)

aplt.imaging.subplot_imaging(imaging=imaging_in_counts)

# If your dataset is in counts you can convert it to electrons per second by supplying the function above with an
# exposure time and using the 'convert_arrays_from_counts' boolean flag.
imaging_converted_to_eps = al.imaging.from_fits(
    image_path=imaging_path + "image.fits",
    pixel_scales=0.1,
    noise_map_path=imaging_path + "noise_map.fits",
    psf_path=imaging_path + "psf.fits",
    exposure_time_map_from_single_value=1000.0,
    convert_from_electrons=True,
)

aplt.imaging.subplot_imaging(imaging=imaging_converted_to_eps)

# The effective exposure time in each pixel may vary. This occurs when simulator is reduced using 'dithering' and 'drizzling'.
# If you have access to an effective exposure-time map, you can use this to convert the image to electrons per second
# instead.
imaging_converted_to_eps = al.imaging.from_fits(
    image_path=imaging_path + "image.fits",
    pixel_scales=0.1,
    noise_map_path=imaging_path + "noise_map.fits",
    psf_path=imaging_path + "psf.fits",
    exposure_time_map_path=imaging_path + "exposure_time_map.fits",
    convert_from_electrons=True,
)
aplt.imaging.subplot_imaging(imaging=imaging_converted_to_eps)

# 2) Postage stamp size - The bigger the postage stamp cut-out of the image the more memory it requires to store it.
#    Why keep the edges surrounding the lens if they are masked out anyway?

#    Lets look at an example of a very large postage stamp - we can barely even see the lens and source galaxies!

imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=dataset_path, folder_names=["imaging_with_large_stamp"]
)

imaging_large_stamp = al.imaging.from_fits(
    image_path=imaging_path + "image.fits",
    pixel_scales=0.1,
    noise_map_path=imaging_path + "noise_map.fits",
    psf_path=imaging_path + "psf.fits",
)

aplt.imaging.subplot_imaging(imaging=imaging_large_stamp)

#  If you have a large postage stamp you can trim it when you load the dataset by specifying a new image size in pixels.
#  This will also trim the noise-map, exposoure time map and other structures which are the same dimensions as the image.
#  This trimming is centred on the image.
imaging_large_stamp_trimmed = al.imaging.from_fits(
    image_path=imaging_path + "image.fits",
    pixel_scales=0.1,
    noise_map_path=imaging_path + "noise_map.fits",
    psf_path=imaging_path + "psf.fits",
    resized_imaging_shape=(101, 101),
)

aplt.imaging.subplot_imaging(imaging=imaging_large_stamp_trimmed)

# 3) Postage stamp size (again). The stamp may also be too small - for example it must have enough padding in the border
#    that our mask includes all pixels with signal. In fact, this padding must also include the a 'blurring region',
#    corresponding to all unmasked image pixels where light blurs into the masks after PSF convolution. Thus, we may
#    need to pad an image to include this region.

imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=dataset_path, folder_names=["imaging_with_small_stamp"]
)

# This image is an example of a stamp which is big enough to contain the lens and source galaxies, but when we
# apply a sensible masks we get an error, because the masks's blurring region goes into the edge of the image.
imaging_small_stamp = al.imaging.from_fits(
    image_path=imaging_path + "image.fits",
    pixel_scales=0.1,
    noise_map_path=imaging_path + "noise_map.fits",
    psf_path=imaging_path + "psf.fits",
)

aplt.imaging.subplot_imaging(imaging=imaging_small_stamp)

# If we apply a masks to this image we get an error when we try to use it to set up a lens image because its
# blurring region hits the image edge.

mask = al.mask.circular(
    shape_2d=imaging_small_stamp.shape,
    pixel_scales=imaging_small_stamp.pixel_scales,
    radius=2.0,
)

# This gives an error because the mask's blurring region hits an edge.
# masked_imaging = al.masked.imaging(imaging=imaging_small_stamp, mask=mask)

# We overcome this using the same input as before. However, now, the resized image shape is bigger than the image,
# thus a padding of zeros is introduced to the edges.
imaging_small_stamp_padded = al.imaging.from_fits(
    image_path=imaging_path + "image.fits",
    pixel_scales=0.1,
    noise_map_path=imaging_path + "noise_map.fits",
    psf_path=imaging_path + "psf.fits",
    resized_imaging_shape=(140, 140),
)

mask = al.mask.circular(
    shape_2d=imaging_small_stamp_padded.shape,
    pixel_scales=imaging_small_stamp_padded.pixel_scales,
    radius=2.0,
)

aplt.imaging.subplot_imaging(imaging=imaging_small_stamp_padded, mask=mask)

# This no longer gives an error!
masked_imaging = al.masked.imaging(imaging=imaging_small_stamp_padded, mask=mask)

########## IVE INCLUDED THE TEXT FOR 5 BELOW SO YOU CAN BE AWARE OF CENTERING, BUT THE BUILT IN FUNCTIONALITY FOR #####
########## RECENTERING CURRENTLY DOES NOT WORK :( ###########

# 5) Lens Galaxy Centering - The lens galaxy should be in the centre of the image as opposed to a corner. This ensures
#    the origin of the lens galaxy's light and mass profiles are near the origin (0.0", 0.0") of the grid used to perform
#    ray-tracing. The defaults priors on light and mass profiles assume a origin of (0.0", 0.0").

# Lets look at an off-center image - clearly both the lens galaxy and Einstein ring are offset in the positive y and x d
# directions.

# imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(path=dataset_path,
#                                                                           folder_names=['imaging_offset_centre'])

# imaging_offset_centre = al.imaging.from_fits(image_path=path+'image.fits', pixel_scales=0.1,
#                                   noise_map_path=path+'noise_map.fits',
#                                   psf_path=path+'psf.fits')
# aplt.imaging.subplot(imaging=imaging_offset_centre)

# We can address this by using supplying a new centre for the image, in pixels. We also supply the resized shape, to
# instruct the code whether it should trim the image or pad the edges that now arise due to recentering.

# imaging_recentred_pixels = al.imaging.from_fits(image_path=path+'image.fits', pixel_scales=0.1,
#                                             noise_map_path=path+'noise_map.fits',
#                                             psf_path=path+'psf.fits',
#                                             resized_imaging_shape=(100, 100),
#                                             resized_imaging_centre_pixels=(0, 0))
# #                                            resized_imaging_centre_arc_seconds=(1.0, 1.0))
# print(imaging_recentred_pixels.shape)
# aplt.imaging.subplot(imaging=imaging_recentred_pixels)

# 6) The noise-map values are the RMS standard deviation in every pixel (and not the variances, HST WHT-map values,
#    etc.). You MUST be 100% certain that the noise_map map is the RMS standard deviations or else your analysis will
#    be incorrect.

# There are many different ways the noise-map can be reduced. We are aiming to include conversion functions for all
# common simulator-reductions. Currently, we have a function to convert an image from a HST WHT map, where
# RMS SD = 1.0/ sqrt(WHT). This can be called using the 'convert_noise_map_from_weight_map' flag.

imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=dataset_path, folder_names=["imaging_with_large_stamp"]
)

imaging_noise_from_wht = al.imaging.from_fits(
    image_path=imaging_path + "image.fits",
    pixel_scales=0.1,
    noise_map_path=imaging_path + "noise_map.fits",
    psf_path=imaging_path + "psf.fits",
    convert_noise_map_from_weight_map=True,
)

aplt.imaging.subplot_imaging(imaging=imaging_noise_from_wht)

# (I don't currently have an example image in WHT for this tutorial, but the function above will work. Above, it
# actually converts an accurate noise-map to an inverse WHT map!

# 7) The PSF zooms around its central core, which is the most important region for strong lens modeling. By
#    default, the size of the PSF image is used to perform convolution. The larger this stamp, the longer this
#    convolution will take to run. In general, we would recommend the PSF size is 21 x 21.

#    Lets look at an image where a large PSF kernel is loaded.

imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=dataset_path, folder_names=["imaging_with_large_psf"]
)

imaging_with_large_psf = al.imaging.from_fits(
    image_path=imaging_path + "image.fits",
    pixel_scales=0.1,
    noise_map_path=imaging_path + "noise_map.fits",
    psf_path=imaging_path + "psf.fits",
)

aplt.imaging.subplot_imaging(imaging=imaging_with_large_psf)

# We can resize a psf the same way that we resize an image.
imaging_with_trimmed_psf = al.imaging.from_fits(
    image_path=imaging_path + "image.fits",
    pixel_scales=0.1,
    noise_map_path=imaging_path + "noise_map.fits",
    psf_path=imaging_path + "psf.fits",
    resized_psf_shape=(21, 21),
)

aplt.imaging.subplot_imaging(imaging=imaging_with_trimmed_psf)

# 8) The PSF dimensions are odd x odd (21 x 21). It is important that the PSF dimensions are odd, because even-sized
#    PSF kernels introduce a half-pixel offset in the convolution routine which can lead to systematics in the lens
#    analysis.

# We do not currently have built-in functionality to address this issue. Therefore, if your PSF has an even
# dimension you must manually trim and recentre it. If you need help on doing this, contact me on the PyAutoLens
# SLACK channel, as I'll have already written the routine to do this by the time you read this tutorial!
