"""
SLaM (Source, Light and Mass): Subhalo Source Parametric Sensitivity Mapping
============================================================================

This example illustrates how to perform DM subhalo sensitivity mapping using a SLaM pipeline for a source modeling
using light profiles.

__Model__

Using a SOURCE LP PIPELINE, LIGHT LP PIPELINE, MASS TOTAL PIPELINE and SUBHALO PIPELINE this SLaM script
fits `Imaging` of a strong lens system, where in the final model:

 - The lens galaxy's light is a bulge with a linear parametric `Sersic` light profile.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - A dark matter subhalo near The lens galaxy mass is included as a`NFWMCRLudlowSph`.
 - The source galaxy is an `Inversion`.

It ends by performing sensitivity mapping of the data using the above model, so as to determine where in the data
subhalos of a given mass could have been detected if present.

This uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`
 `subhalo/detection`

Check them out for a full description of the analysis!

__Start Here Notebook__

If any code in this script is unclear, refer to the `subhalo/detect/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import sys
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())
import slam

"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "dark_matter_subhalo"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=path.join("imaging", "slam"),
    unique_tag=dataset_name,
    info=None,
    number_of_cores=2,
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

This is the standard SOURCE LP PIPELINE described in the `slam/start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(dataset=dataset)

bulge = af.Model(al.lp_linear.Sersic)

source_lp_result = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp_linear.SersicCore),
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__LIGHT LP PIPELINE__

This is the standard LIGHT LP PIPELINE described in the `slam/start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
)

bulge = af.Model(al.lp_linear.Sersic)

light_results = slam.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_lp_result,
    source_result_for_source=source_lp_result,
    lens_bulge=bulge,
    lens_disk=None,
)

"""
__MASS TOTAL PIPELINE__

This is the standard MASS TOTAL PIPELINE described in the `slam/start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(dataset=dataset)

mass_result = slam.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_lp_result,
    source_result_for_source=source_lp_result,
    light_result=light_results,
    mass=af.Model(al.mp.PowerLaw),
)

"""
__SUBHALO PIPELINE (sensitivity mapping)__

The SUBHALO PIPELINE (sensitivity mapping) performs sensitivity mapping of the data using the lens model
fitted above, so as to determine where subhalos of what mass could be detected in the data. A full description of
Sensitivity mapping if given in the SLaM pipeline script `slam/subhalo/sensitivity_imaging.py`.
"""
subhalo_results = slam.subhalo.sensitivity_imaging_lp.run(
    settings_search=settings_search,
    mask=mask,
    psf=dataset.psf,
    mass_result=mass_result,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
)

"""
Finish.
"""
