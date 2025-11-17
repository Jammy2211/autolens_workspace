"""
Tutorial 8: Need For Speed
==========================

In this chapter, we have learnt how to model strong lenses and how to balance complexity and realism to ensure that we
infer a good lens model.

For fitting more complex lens models, the final challenge that we face is keeping the run-time low. One can easily end
up in a situation where a model-fit takes days, or longer, to fit just one image. For fitting complex models and high
resolution datasets this is somewhat unavoidable. However, it is worth us discussing what drives the long run-times of
the lens modeling process and how we might speed it up.

We have carefully tracked the log likelihood evaluation times and therefore expected overall run-time of every
fit performed in this chapter. Lets quickly remind ourselves of the main factors that drive the run-time, and
how we might reduce it.

__Searching Non-linear Parameter Space__

The time it takes for the non-linear search to sample parameter space and find the high likelihood models is driven by:

 - Dimensionality: A more complex parameter space (e.g. more parameters) takes longer to search.
 - Priors: The broader the priors on each parameter the longer the search.
 - Settings: Non-linear search settings which sample parameter space more thoroughly lead to longer run-times.

When we use only one search to fit a lens model, we are somewhat restricted in how we can try to achieve faster run
times by changing these 3 aspects of the search.

In the next chapter, we introduce 'non-linear search chaining', which fits a lens model using multiple searches that
are performed back-to-back. A key motivation for this is that it gives us a lot more flexibility in juggling the
dimensionality, priors and settings so as to perform faster and more efficient lens modeling.

In the optional **HowToGalaxy** tutorial `chapter_optional/tutorial_searches.ipynb` we discuss other non-linear
searches supported by **HowToGalaxy** which use a different approach to sample parameter sample than `Nautilus`. For
those familiar with statistical inference, this includes maximum likelihood estimators and MCMC algorithms.

For lens modeling, we have not found another non-linear search that provides as robust and efficient results as
Nautilus. We therefore recommend uses stick to `Nautilus`.

__Algorithmic Optimization__

Every operation **PyAutoLens** performs to fit strong lens data with a model takes time, for example:

 - Computing the intensity values from a light profile.
 - Convolving the image that comes from a tracer with the PSF to compare it to the data.

One can therefore in principle make **PyAutoLens** run faster by using more efficient algorithms. However, I am
confident that for many tasks and operations we have written code that is already very fast!

I often get asked, given that **PyAutoLens** is written in Python (a synonymously slow programming language), is it not
really slow? **PyAutoLens** uses a library called `numba` to ensure that it runs fast, which recompiles Python
functions into C functions before **PyAutoLens** runs. This gives us C-like speed, but in Python code. If you`ve got
your own code that needs speeding up, I strongly recommend that you look up Numba:

http://numba.pydata.org/

Therefore, **PyAutoLens** is pretty well optimized and there are no 'low hanging fruit' speed ups available by writing
the code in a different language.

__Data Quantity__

The final factor driving run-speed is the quantity of data that is fitted. For every image-pixel that we fit,
we have to compute the light profile intensities, mass profile deflection angles and convolve it with the telescope's
PSF. The larger that PSF is, the more convolution operations we have to perform too.

In the previous exercises, we used images with a pixel scale of 0.1". This value is relatively low resolution: most
Hubble Space Telescope images have a pixel scale of 0.05", which is four times the number of pixels! Some telescopes
observe at scales of 0.03" or, dare I say it, 0.01". At these resolutions things can *really* slow down, if we
do not think carefully about run speed beforehand.

There are ways that we can reduce the number of image-pixels we fit, via masking. If we mask out more of the image,
we will fit fewer pixels and **PyAutoLens** will run faster. If you want the best, most perfect lens model possible,
aggressive masking and cutting the data in this way is a bad idea, as discussed in tutorial 5.

__Preloading__

When certain components of a model are fixed its associated quantities do not change during a model-fit. For
example, for a lens model where all light profiles are fixed, the PSF blurred model-image of those light profiles
is also fixed.

In the next chapter we will introduce search chaining, whereby pipelines that fit a chain of different lens models are
used. In these pipelines, it is common for different components of the model to be fixed, and thus preloading
can be used to speed up the analysis.

**PyAutoLens** uses _implicit preloading_ to inspect the model and determine what quantities are fixed. It then stores
these in memory before the non-linear search begins such that they are not recomputed for every likelihood evaluation.

__Wrap Up__

This tutorial simply wanted to get you thinking about *why* a lens model takes as long to fit as it does.
"""
