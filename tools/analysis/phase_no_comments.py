import autofit as af
from autolens.pipeline.phase import phase_imaging, phase_extensions
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.data import ccd
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.lens.plotters import lens_fit_plotters

import os

path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

lens_name = 'example_lens'

ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/' + lens_name + '/image.fits',
                                       psf_path=path+'/data/'+lens_name+'/psf.fits',
                                       noise_map_path=path+'/data/'+lens_name+'/noise_map.fits',
                                       pixel_scale=0.1)

# Create a mask for the data, which we setup as a 3.0" circle.
mask = msk.Mask.circular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=3.0)

# We model our lens galaxy using a mass profile (a singular isothermal ellipsoid) and our source galaxy
# a light profile (an elliptical Sersic). We load these profiles from the 'light_profiles (lp)' and
# 'mass_profiles (mp)' modules.

# To setup our model galaxies, we use the GalaxyModel class, which represents a galaxy whose parameters
# are variable and fitted for by the analysis. The galaxies are also assigned redshifts.

lens_galaxy_model = gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal)
source_galaxy_model = gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic)

# To perform the analysis, we set up a phase using the 'phase' module (imported as 'ph').
# A phase takes our galaxy models and fits their parameters using a non-linear search
# (in this case, MultiNest).
phase = phase_imaging.LensSourcePlanePhase(lens_galaxies=dict(lens=lens_galaxy_model),
                                source_galaxies=dict(source=source_galaxy_model),
                                phase_name='example/phase_example', optimizer_class=af.MultiNest)

# We run the phase on the ccd data, print the results and plot the fit.
result = phase.run(data=ccd_data)
lens_fit_plotters.plot_fit_subplot(fit=result.most_likely_fit)

