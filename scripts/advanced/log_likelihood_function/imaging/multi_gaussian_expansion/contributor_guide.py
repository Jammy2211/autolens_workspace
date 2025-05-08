"""
__Log Likelihood Function: Linear Light Profile__

The `light_profile` script accompanying this one provides a step-by-step guide of the log_likelihood_function used to
fit `Imaging` data with a parametric light profile.

This script provides a contributor guide, that gives links to every part of the source-code that performs each step
of the likelihood evaluation.

This gives contributors a sequential run through of what functions, modules and packages in the source code are called
when the likelihood is evaluated, and should help them navigate the source code.


__Source Code__

The likelihood evaluation is spread over the following two GitHub repositories:

**PyAutoArray**: https://github.com/Jammy2211/PyAutoArray
**PyAutoGalaxy**: https://github.com/Jammy2211/PyAutoGalaxy


__LH Setup: Light Profiles (Setup)__

To see examples of all light profiles checkout the `light_profiles` package:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/profiles/light

Each light profile has an `image_2d_from` method that returns the image of the profile, which is used in the
likelihood function example.

Each function uses a `@aa.grid_dec.transform` decorator, which performs the coordinate transformation of the
grid to elliptical coordinates via the light profile's geometry.

These coordinate transforms are performed in the following modules:

https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/geometry_profiles.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/geometry/geometry_util.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/decorators/transform.py


Every one of these light profiles has a corresponding linear light profile class which inherits from it,
which can be found in the `light_profiles.linear` package:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/profiles/light/linear

This is the module which is imported as `lp_linear` and allows us to create light profiles via the
`ag.lp.Linear` API.

__LightProfileLinearObjFuncList__

This step passed the linear light profiles into the `LightProfileLinearObjFuncList` object, which acts as an
interface between the linear light profiles and the linear algebra used to compute their intensity via the inversion.

This object can be found at the following URL:

https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/light/linear/abstract.py

__LH Step 2: Mapping Matrix__

`Mapper.__init__`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/abstract.py
`mapping_matrix_from`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/mapper_util.py


__LH Step 3: Blurred Mapping Matrix (f)__

This uses the methods in `Convolver.__init__` and `Convolver.convolve_mapping_matrix`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/operators/convolver.py


__LH Step 4: Data Vector (D)__

The calculation is performed in the `data_vector` attribute of an `Inversion` object:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/imaging/mapping.py

The function called is `inversion_imaging_util.data_vector_via_blurred_mapping_matrix_from`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/imaging/inversion_imaging_util.py


__LH Step 5: Curvature Matrix (F)__

The calculation is performed in the `curvature_matrix` attribute of an `Inversion` object:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/imaging/mapping.py

The function called is `inversion_imaging_util.curvature_matrix_via_blurred_mapping_matrix_from`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/imaging/inversion_imaging_util.py


__LH Step 6: Source Reconstruction (S)__

 An `Inversion` object has a property `reconstruction` to perform this calculation:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/matrices.py


__LH Step 7: Image Reconstruction__

 The calculation is performed by the method `mapped_reconstructed_data_via_mapping_matrix_from` at:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/leq/leq_util.py

 This function is called by `AbstractInversion.mapped_reconstructed_data`:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/abstract.py


__LH Step 8-10: Likelihood Function__

The `model_image` was subtracted from the observed image, and the residuals, chi-squared
and log likelihood computed.

This is performed in the `FitImaging` object:

https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/imaging/fit_imaging.py

The following methods are relevant in this module:

`blurred_image`: Computes the blurred image via convolution described above.
`model_data`: This is the blurred image, but the variable is renamed as for more advanced fits it is extended.

The steps of subtracting the model image from the observed image and computing the residuals, chi-squared and
log likelihood are performed in the following `FitDataset` and `FitImaging` objects::

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_dataset.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_imaging.py

Many calculations occur in `fit_util.py`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_util.py

These modules also ihnclude the calcualtion of the `noise_normalization`.

__GalaxiesToInversion__

The end of the `log_likelihood_function` script uses the `GalaxiesToInversion` object to create the `Inversion` object.

This is how an inversion is created inside of the `FitImaging` object.

This object can be found here:

https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/galaxy/to_inversion.py
"""
