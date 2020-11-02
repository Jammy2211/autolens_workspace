input(
    "\n"
    "############################################\n"
    "### AUTOLENS WORKSPACE WORKING DIRECTORY ###\n"
    "############################################\n\n"
    """
    PyAutoLens scripts assume that the `autolens_workspace` directory is the Python working directory. 
    This means that, when you run an example script, you should run it from the `autolens_workspace` 
    as follows:
    
    
    cd path/to/autolens_workspace (if you are not already in the autolens_workspace).
    python3 examples/model/beginner/mass_total__source_parametric.py


    The reasons for this are so that PyAutoLens can:
     
    - Load configuration settings from config files in the `autolens_workspace/config` folder.
    - Load example data from the `autolens_workspace/dataset` folder.
    - Output the results of models fits to your hard-disk to the `autolens/output` folder. 

    If you have any errors relating to importing modules, loading data or outputting results it is likely because you
    are not running the script with the `autolens_workspace` as the working directory!
    
    [Press Enter to continue]"""
)

input(
    "\n"
    "###############################\n"
    "##### MATPLOTLIB BACKEND ######\n"
    "###############################\n\n"
    """
    We`re now going to plot an image in PyAutoLens using Matplotlib, using the backend specified in the following
    config file (the backend tells Matplotlib where to render the plot)"


    autolens_workspace/config/visualize/general.ini -> [general] -> `backend`


    The default entry for this is `default` (check the config file now). This uses the default Matplotlib backend
    on your computer. For most users, pushing Enter now will show the figure without error.

    However, we have had reports that if the backend is set up incorrectly on your system this plot can either
    raise an error or cause the `welcome.py` script to crash without a message. If this occurs after you
    push Enter, the error is because the Matplotlib backend on your computer is set up incorrectly.

    To fix this in PyAutoLens, try changing the backend entry in the config file to one of the following values:"


    backend=TKAgg
    backend=Qt5Agg
    backeknd=Qt4Agg


    [Press Enter to continue]
    """
)

import autolens as al
import autolens.plot as aplt

grid = al.Grid.uniform(
    shape_2d=(50, 50),
    pixel_scales=0.1,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

sersic_light_profile = al.lp.EllipticalExponential(
    centre=(0.3, 0.2), elliptical_comps=(0.2, 0.0), intensity=0.05, effective_radius=1.0
)

aplt.LightProfile.image(light_profile=sersic_light_profile, grid=grid)

input(
    "\n"
    "##############################\n"
    "## LIGHT AND MASS PROFILES ###\n"
    "##############################\n\n"
    """
    The image displayed on your screen shows a `LightProfile`, the object PyAutoLens uses to represent the 
    luminous emission of galaxies. This emission is unlensed, which is why it looks like a fairly ordinary and 
    boring galaxy.

    To perform ray-tracing, we need a `MassProfile`, which will be shown after you push [Enter]. The figures will 
    show the convergence, gravitational potential and deflection angle map of the `MassProfile`, vital quantities 
    for performing lensing calculations.

    [Press Enter to continue]
    """
)

isothermal_mass_profile = al.mp.EllipticalIsothermal(
    centre=(0.0, 0.0), elliptical_comps=(0.1, 0.0), einstein_radius=1.6
)

aplt.MassProfile.convergence(mass_profile=isothermal_mass_profile, grid=grid)
aplt.MassProfile.potential(mass_profile=isothermal_mass_profile, grid=grid)
aplt.MassProfile.deflections_y(mass_profile=isothermal_mass_profile, grid=grid)
aplt.MassProfile.deflections_x(mass_profile=isothermal_mass_profile, grid=grid)

input(
    "\n"
    "########################\n"
    "##### RAY TRACING ######\n"
    "########################\n\n"
    """
    By combining `LightProfile`'s and `MassProfile`'s PyAutoLens can perform ray-tracing and calculate how the 
    path of the source `Galaxy`'s light-rays are bent by the lens galaxy as they travel to the Earth!
    
    Pushing [Enter] will show the `LightProfile` displayed previously gravitationally lensed by the `MassProfile`.
    
    [Press Enter to continue]
    """
)

from astropy import cosmology as cosmo

lens_galaxy = al.Galaxy(redshift=0.5, mass=isothermal_mass_profile)
source_galaxy = al.Galaxy(redshift=1.0, light=sersic_light_profile)
tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy], cosmology=cosmo.Planck15
)

aplt.Tracer.image(tracer=tracer, grid=grid)

input(
    "\n"
    "###########################\n"
    "##### WORKSPACE TOUR ######\n"
    "###########################\n\n"
    """
    PyAutoLens is now set up and you can begin exploring the workspace. We recommend new users begin in the 
    following folders:
    
    - examples: Example Python scripts showing how to perform lensing calculations, fit data and model strong 
      lenses.
      
    - howtolens: Jupyter notebook tutorials introducing beginners to strong gravitational lensing, describing how to
     perform scientific analysis of lens data and detailing the PyAutoLens API.
     
    - simulators: Example scripts for simulating images of strong lenses for CCD imaging devices (e.g. the Hubble
     Space Telescope) and interferometers (e.g. ALMA).
     
    - preprocess: Tutorials describing the input formats of data for PyAutoLens and tools for preprocessing the
     data before fitting it with PyAutoLens.
    
    The reaming folders in the workpace (e.g. `transdimensional`, `slam`, `aggregator`, etc.) are for experienced
    users. The example scripts and HowToLens lectures will guide new users to these modules when they have sufficient
    experience and familiarity with PyAutoLens.
    
    [Press Enter to continue]
    """
)
