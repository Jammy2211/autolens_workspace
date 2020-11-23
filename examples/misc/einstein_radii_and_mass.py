import autolens as al
from astropy import cosmology as cosmo

"""
__Einstein Radii and Mass__

This is a simple script for computing the Einstein Radii and Mass given known input parameters.

For errors, you`ll need to use the aggregator (autolens_workspace -> aggregator).

Lets set up an `EllipticalIsothermal` `MassProfile`.
"""

sie = al.mp.EllipticalIsothermal(einstein_radius=2.0, elliptical_comps=(0.0, 0.333333))

"""
We can compute its Einstein Radius and Mass, which are defined as the area within the tangential critical curve. These
are stored in the properties `einstein_radius_via_tangential_critical_curve` and 
`einstein_mass_via_tangential_critical_curve`.

Lets print the Einstein Radius, which is returned in the default internal **PyAutoLens** units of arc-seconds.
"""

einstein_radius = sie.einstein_radius_via_tangential_critical_curve
print("Einstein Radius (arcsec) = ", einstein_radius)

"""If we know the redshift of the `MassProfile` and assume an **AstroPy** cosmology we can convert this to 
kilo-parsecs."""

kpc_per_arcsec = al.util.cosmology.kpc_per_arcsec_from(
    redshift=0.5, cosmology=cosmo.Planck15
)
einstein_radius_kpc = einstein_radius * kpc_per_arcsec
print("Einstein Radius (kpc) = ", einstein_radius_kpc)


"""
A `MassProfile` does not know its redshift, nor does it know its source redshift. Thus, the Einstein mass cannot be
provided in units of solar masses. Instead, its mass is computed in angular units and is given by:

 pi * einstein_radius (arcsec) ** 2.0
"""

einstein_mass = sie.einstein_mass_angular_via_tangential_critical_curve
print("Einstein Mass (angular) = ", einstein_mass)

"""
To convert this mass to solar masses, we need the critical surface mass density of the `MassProfile`, which relies on 
it being a strong lens with not only a lens redshift (e.g. the redshift of the profile) but also a source redshift.

If we assume this mass profile is at redshift 0.5 and it is lensing a source at redshift 1.0 we can compute its mass
in solar masses.
"""

critical_surface_density = al.util.cosmology.critical_surface_density_between_redshifts_from(
    redshift_0=0.5, redshift_1=1.0, cosmology=cosmo.Planck15
)
einstein_mass_kpc = einstein_mass * critical_surface_density
print("Einstein Mass (kpc) = ", einstein_mass_kpc)
print("Einstein Mass (kpc) = ", "{:.4e}".format(einstein_mass_kpc))

"""We can also use the above methods on Galaxy objects, which may contain multiple `MassProfile`'s."""
galaxy = al.Galaxy(redshift=0.5, mass_0=sie, mass_1=sie)
print(
    "Einstein Radius (arcsec) via Galaxy = ",
    galaxy.einstein_radius_via_tangential_critical_curve,
)
print(
    "Einstein Mass (angular) via Galaxy = ",
    galaxy.einstein_mass_angular_via_tangential_critical_curve,
)

"""
In principle, the Einstein Mass of a `Tracer` should be readily accessible in a `Tracer` object, given this contains
all of the galaxies in a strong lens system (and thus has their redshifts) as well as an input Cosmology.

However, we do not provide methods with this quantity and require that you, the user, compute the Einstein mass 
(in angular or solar masses) using examples above. This is because for systems with multiple galaxies or planes, the 
definition of an Einstein Radius / Mass become less clear. We feel it is better that a user explicitly computes these 
quantities from a `Tracer` so if it has multiple galaxies or planes you are aware of this.

The example below shows how for a single lens + source galaxy system `Tracer` can be used to compute the
Einstein Radii and Masses above.
"""

source_galaxy = al.Galaxy(redshift=1.0)

tracer = al.Tracer.from_galaxies(galaxies=[galaxy, source_galaxy])

image_plane_galaxy = tracer.planes[0].galaxies[0]
source_plane_galaxy = tracer.planes[1].galaxies[0]

einstein_radius = image_plane_galaxy.einstein_radius_via_tangential_critical_curve
print("Einstein Radius via Tracer (arcsec) = ", einstein_radius)

kpc_per_arcsec = al.util.cosmology.kpc_per_arcsec_from(
    redshift=image_plane_galaxy.redshift, cosmology=cosmo.Planck15
)
einstein_radius_kpc = einstein_radius * kpc_per_arcsec
print("Einstein Radius via Tracer (kpc) = ", einstein_radius_kpc)

einstein_mass = image_plane_galaxy.einstein_mass_angular_via_tangential_critical_curve
print("Einstein Mass via Tracer (angular) = ", einstein_mass)

critical_surface_density = al.util.cosmology.critical_surface_density_between_redshifts_from(
    redshift_0=image_plane_galaxy.redshift,
    redshift_1=source_plane_galaxy.redshift,
    cosmology=cosmo.Planck15,
)
einstein_mass_kpc = einstein_mass * critical_surface_density
print("Einstein Mass via Tracer (kpc) = ", einstein_mass_kpc)
print("Einstein Mass via Tracer (kpc) = ", "{:.4e}".format(einstein_mass_kpc))
