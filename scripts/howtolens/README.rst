The ``howtolens`` folder contains **HowToLens** lectures, which teach a new user what strong lensing is and how to model
a strong lens.

Folders
-------

- ``chapter_1_introduction``: An introduction to strong gravitational lensing and **PyAutolens**.
- ``chapter_2_lens_modeling``: How to model strong lenses, including a primer on Bayesian non-linear analysis.
- ``chapter_3_search_chaining``: How to fit complex lens models using non-linear search chaining.
- ``chapter_4_pixelizations``: How to perform pixelized reconstructions of the source-galaxy.
- ``chapter_5_hyper_mode``: How to use **PyAutoLens** advanced modeling features that adapt the model to the strong lens being analysed.
- ``chapter_optional``: Optional tutorials.

Full Explanation
----------------

Welcome to **HowToLens** - The **PyAutoLens** tutorial!

JUYPTER NOTEBOOKS
-----------------

All tutorials are supplied as Jupyter Notebooks, which come with a '.ipynb' suffix. For those new to Python, Jupyter 
Notebooks are a different way to write, view and use Python code. Compared to the traditional Python scripts, 
they allow:

- Small blocks of code to be viewed and run at a time.
- Images and visualization from a code to be displayed directly underneath it.
- Text script to appear between the blocks of code.

This makes them an ideal way for us to present the **HowToLens** lecture series, therefore I recommend you get yourself
a Jupyter notebook viewer (https://jupyter.org/) if you have not done so already.

If you *really* want to use Python scripts, all tutorials are supplied a ``.py`` python files in the 'scripts' folder of
each chapter.

For actual **PyAutoLens** use, I recommend you use Python scripts. Therefore, as you go through the lecture series 
you will notice that we will transition you to Python scripts in the third chapter.

LENSING THEORY
--------------

HowToLens assumes minimal previous knowledge of gravitational lensing and astronomy. However, it is beneficial to give
yourself a basic theoretical grounding as you go through the lectures. I heartily recommend you have open the
lecture course on gravitational lensing by Massimo Meneghetti below as you go through the tutorials, and refer to it
for anything that isn't clear in HowToLens.

http://www.ita.uni-heidelberg.de/~massimo/sub/Lectures/gl_all.pdf

VISUALIZATION
-------------

Before beginning the **HowToLens** lecture series, in chapter 1 you should do 'tutorial_0_visualization'. This will
take you through how **PyAutoLens** interfaces with matplotlib to perform visualization and will get you setup such that
images and figures display correctly in your Jupyter notebooks.

CODE STYLE AND FORMATTING
-------------------------

When you begin the notebooks, you may notice the style and formatting of our Python code looks different to what you
are used to. For example, it is common for brackets to be placed on their own line at the end of function calls,
the inputs of a function or class may be listed over many separate lines and the code in general takes up a lot more
space then you are used to.

This is intentional, because we believe it makes the cleanest, most readable code possible. In fact - lots of people do,
which is why we use an auto-formatter to produce the code in a standardized format. If you're interested in the style
and would like to adapt it to your own code, check out the Python auto-code formatter 'black'.

https://github.com/python/black

HOW TO TACKLE HowToLens
-----------------------

The **HowToLens** lecture series current sits at 5 chapters, and each will take a day or so to go through
properly. You probably want to be modeling lenses faster than that! Furthermore, the concepts in the
later chapters are pretty challenging, and familiarity and lens modeling is desirable before you
tackle them.
 
Therefore, we recommend that you complete chapters 1 & 2 and then apply what you've learnt to the modeling of simulated
and real strong lens data, using the scripts found in the 'autolens_workspace'. Once you're happy
with the results and confident with your use of **PyAutoLens**, you can then begin to cover the advanced functionality
covered in chapters 3, 4 & 5.

OVERVIEW OF CHAPTER 1 (Beginner)
--------------------------------

**Strong Lensing with PyAutoLens**

In chapter 1, we'll learn about strong gravitational lensing and **PyAutoLens**. At the end, you'll
be able to:

1) Create uniform grid's of (x,y) Cartesian coordinates.
2) Combine these grid's with light and mass profiles to make images, convergence maps, gravitational potentials and deflection angle-maps.
3) Combine these light and mass profiles to make galaxies.
4) Perform ray-tracing with these galaxy's whereby a grid is ray-traced through an image-plane / source-plane strong lensing configuration.
5) Simulate telescope CCD imaging data of a strong gravitational lens.
6) Fit strong lensing data with model images generated via ray-tracing.

OVERVIEW OF CHAPTER 2 (Beginner)
--------------------------------

**Bayesian Inference and Non-linear Searches**

In chapter 2, we'll cover Bayesian inference and model-fitting via a non-linear search. We will use these tools to
fit CCD imaging data of a strong gravitational lens with a lens model. At the end, you'll understand:

1) The concept of a non-linear search and non-linear parameter space.
2) How to fit a lens model to strong lens CCD imaging via a non-linear search.
3) The trade-off between realism and complexity when choosing a lens model.
4) Why an incorrect lens model may be inferred and how to prevent this from happening.
5) The challenges that are involved in inferred a robust lens model in a computationally reasonable run-time.

**Once completed, you'll be ready to model your own strong gravitational lenses with PyAutoLens!**

OVERVIEW OF CHAPTER 3 (Intermediate)
------------------------------------

**Automated Modeling with non-linear search chaining**

In chapter 3, we'll learn how to chain multiple non-linear searches together to build automated lens modeling pipelines
which can:

1) Break-down the fitting of a model using multiple non-linear searches and prior passing.
2) Fit CCD imaging of a strong lens where the lens light and source light are fitted separately.
3) Use a custom pipeline to fit a strong lens with multiple lens galaxies or source galaxies.
4) Know how to use advanced pipelines called the Source, Light and Mass (SLaM) pipelines.

OVERVIEW OF CHAPTER 4 (Intermediate)
------------------------------------

**Using an inverison to perform a pixelized source reconstructions**

In chapter 4, we'll learn how to reconstruct the lensed source galaxy using a pixel-grid, ensuring that we can fit an
accurate lens model to sources with complex and irregular morphologies. You'll learn how to:

1) Pixelize a source-plane into a set of source-plane pixels defined by mappings to image pixels.
2) Perform a linear inversion on this source-plane pixelization to reconstruct the source's light.
3) Apply a smoothness prior on the source reconstruction, called regularization.
4) Apply smoothing within a Bayesian framework to objectively quantify the source reconstruction's complexity.
5) Define a border in the source-plane to prevent pixels tracing outside the source reconstruction.
6) Use alternative pixelizations, for example a Voronoi mesh whose pixels adapt to the lens's mass model.
7) Use these features to fit a lens model via non-linear searches.

OVERVIEW OF CHAPTER 5 (Advanced)
--------------------------------

**Hyper-Mode**

In hyper-mode, we introduced advanced functionality that adapts various parts of the lens modeling procedure to the
data that we are fitting.

NOTE: Hyper-mode is conceptually quite challenging, and I advise that you make sure you are very familiar with
PyAutoLens before covering chapter 5!

1) Adapt an inversions's `Pixelization` to the morphology of the reconstructed source galaxy.
2) Adapt the `Regularization` scheme applied to this source to its surface brightness profile.
3) Use hyper-galaxies to scale the image's noise-map during fitting, to prevent over-fitting regions of the image.
4) include aspects of the data reduction in the model fitting, for example the background sky subtraction.
5) Use these features in PyAutoLens's search chaining framework.