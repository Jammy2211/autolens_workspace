"""
__Log Likelihood Function: Light Profile__

The `light_profile` script accompanying this one provides a step-by-step guide of the log_likelihood_function used to
fit `Imaging` data with a linear parametric light profile.

This script provides a contributor guide, that gives links to every part of the source-code that performs each step
of the likelihood evaluation.

This gives contributors a sequential run through of what functions, modules and packages in the source code are called
when the likelihood is evaluated, and should help them navigate the source code.


__Source Code__

The likelihood evaluation is spread over the following two GitHub repositories:

**PyAutoArray**: https://github.com/Jammy2211/PyAutoArray
**PyAutoGalaxy**: https://github.com/Jammy2211/PyAutoGalaxy


__LH Setup: Light Profiles (Setup)__

To see examples of all light profiles checkout  the `light_profiles` package:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/profiles/light

Each light profile has an `image_2d_from` method that returns the image of the profile, which is used in the
likelihood function example.

Each function uses a `@aa.grid_dec.transform` decorator, which performs the coordinate transformation of the
grid to elliptical coordinates via the light profile's geometry.

These coordinate transforms are performed in the following modules:

https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/geometry_profiles.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/geometry/geometry_util.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/decorators/transform.py


__LH Setup: Galaxy__

The galaxy package and module contains the `Galaxy` object:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/galaxy

The `Galaxy` object also has an `image_2d_from` method that returns the image of the galaxy, which calls the
`image_2d_from` functiuon of each light profile and sums them.


__LH Step 1: Galaxy Image__

This step calls the `image_2d_from` method of the `Galaxy` object, as described above.

The input to this function is the masked dataset's `Grid2D` object, which is computed here:

- See the method `grid`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/abstract/dataset.py

The calculation also used a `blurring_grid` to evaluate light which the PSF convolution blurred into the mask,
which is computed here:

- See the method `blurring_grid`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/imaging/dataset.py


__Fourier Transform__

The Non-Uniform Fast Fourier Transform (NUFFT) uses one of the `Transformer` objects and its method `visibilities_from`

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/operators/transformer.py


__LH Step 3: Likelihood Function__

The `model_data` was subtracted from the observed visibilities_from, and the residuals, chi-squared
and log likelihood computed.

This is performed in the `FitInterferometer` object:

https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/interferometer/fit_interferometer.py

The following methods are relevant in this module:

`model_data`: This is the model visibilities.

The steps of subtracting the model data from the observed data and computing the residuals, chi-squared and
log likelihood are performed in the following `FitDataset` and `FitInterferometer` objects::

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_dataset.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_interferometer.py

Many calculations occur in `fit_util.py`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_util.py

These modules also include the calculation of the `noise_normalization`.
"""
