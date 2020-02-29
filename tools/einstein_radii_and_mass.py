import autolens as al

# This is a simple script for computing the Einstein Radii and Mass of a Lens Galaxy given known input parameters.

# For errors, you'll need to use the aggregator (autolens_workspace -> aggregator).

# Lets set up an SIE mass profile.

sie = al.mp.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.8)

# We can compute its Einstein Radius and Mass using the "einstein_radius_in_units' function.

# The Einstein Radius can then be printed in arc-seconds (for an elliptical Isothermal the model Einstein Radius
# differs from the one output below, due to a difference in definition). The definition below is that used by
# SLACS, which is the area of the critical curves in the image-plane.

print(sie.einstein_radius_in_units(unit_length="arcsec", redshift_object=0.5))

# The Einstein Radius requires the redshift of the profile (e.g. the lens galaxy) to be converted to kpc.

print(sie.einstein_radius_in_units(unit_length="kpc", redshift_object=0.5))

# The Einstein Mass requires the redshifts of the profile (the lens) and the source to be converted to solMass.
einstein_mass = sie.einstein_mass_in_units(
    unit_mass="solMass", redshift_object=0.5, redshift_source=1.0
)
print(einstein_mass)
print("{:.4e}".format(einstein_mass))

# We can also use the above methods on Galaxy objects.

galaxy = al.Galaxy(redshift=0.5, mass=sie)

# There is currently a bug which means these methods do not use the galaxy's redsshift (doh) and that it needs to be
# input manually again.

print()
print(galaxy.einstein_radius_in_units(unit_length="arcsec"))
print(galaxy.einstein_radius_in_units(unit_length="kpc", redshift_object=0.5))
print(
    "{:.4e}".format(
        (
            galaxy.einstein_mass_in_units(
                unit_mass="solMass", redshift_object=0.5, redshift_source=1.0
            )
        )
    )
)

# Finally, tracers work too (again, use of their redshifts is currently bugged :( ).

# Infact, these seem pretty buggy in general... lets ignore them for now.

# lens_galaxy = al.Galaxy(redshift=0.5, mass=sie)
# source_galaxy = al.Galaxy(redshift=1.0)
#
# tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
# print()
# print(tracer.einstein_radius_in_units(unit_length="arcsec", redshift_object=0.5))
# print(tracer.einstein_radius_in_units(unit_length="kpc", redshift_object=0.5))
# print("{:.4e}".format(tracer.einstein_mass_in_units(unit_mass="SolMass", redshift_object=0.5, redshift_source=1.0)))
