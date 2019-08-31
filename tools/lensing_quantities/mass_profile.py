from astropy import cosmology as cosmo

import autolens as al

sie = al.mass_profiles.EllipticalIsothermal(
    centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.0
)

galaxy = al.Galaxy(redshift=0.5, mass=sie)

summary = galaxy.summarize_in_units(
    unit_mass="solMass", radii=[1.0], redshift_source=1.0, cosmology=cosmo.Planck15
)

print("\n".join(summary))
