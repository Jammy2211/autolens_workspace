import numpy as np
from os import path
import json
import autofit as af
import autolens as al
import autolens.plot as aplt


def ref3_to_deg(ra, dec, d_ra, d_dec):
    """
    Transform a realtiv sky position RA and DEC positions in arcsencs in absolute coordinate using a reference position

    Parameters
    ----------
    ra : float
             Right ascension position of the reference point (in degrees)
    dec : float
              Declinaison position of the reference point (in degrees)
    d_ra : float
             Right ascension of the relativ position (in arcsec)
    d_dec : float
            Declination of the relativ position (in arcsec)
    Returns
    ----------
    RA,DEC : float,float
             absolute position in RA and DEC (in degrees)
    """
    DEC = dec + d_dec / 3600.0
    RA = ra - d_ra / 3600.0 / np.cos(dec / 180.0 * np.pi)

    return RA, DEC


def deg_to_ref3(ra_ref, dec_ref, ra, dec):
    """
    Transform an aboslute sky position RA and DEC positions in degrees in a realtive coordinate using a reference position

    Parameters
    ----------
    ra_ref : float
             Right ascension position of the reference point (in degrees)
    dec_ref : float
              Declinaison position of the reference point (in degrees)
    ra : float
             Right ascension of the object position (in degrees)
    dec : float
            Declination of the object position (in  degrees)
    Returns
    ----------
    RA,DEC : float,float
             relativ poisiotn in RA and DEC (in acresecnonds)
    """
    d_dec = (dec - dec_ref) * 3600.0
    d_ra = (ra_ref - ra) * np.cos(dec_ref / 180.0 * np.pi) * 3600.0
    # print RA_ref,DEC_ref,RA,DEC,d_ra,d_dec
    return d_ra, d_dec


def deg_to_ref3(RA_ref, DEC_ref, RA, DEC):
    """
    Transform an aboslute sky position RA and DEC positions in degrees in a realtive coordinate using a reference position

    Parameters
    ----------
    RA_ref : float
             Right ascension position of the reference point (in degrees)
    DEC_ref : float
              Declinaison position of the reference point (in degrees)
    RA : float
             Right ascension of the object position (in degrees)
    DEC : float
            Declination of the object position (in  degrees)
    Returns
    ----------
    RA,DEC : float,float
             relativ poisiotn in RA and DEC (in acresecnonds)
    """
    d_dec = (DEC - DEC_ref) * 3600.0
    d_ra = (RA_ref - RA) * np.cos(DEC_ref / 180.0 * np.pi) * 3600.0
    # print RA_ref,DEC_ref,RA,DEC,d_ra,d_dec
    return d_ra, d_dec


ra_list = [177.94166154725815, 177.94165597107437, 177.9416643609197]
dec_list = [33.23750122888889, 33.23749748861111, 33.23750242916667]

RA_Ref = ra_list[0]
DEC_ref = dec_list[0]

for index in range(len(ra_list)):

    galaxy_centre = deg_to_ref3(
        RA_ref=RA_Ref, DEC_ref=DEC_ref, RA=ra_list[index], DEC=dec_list[index]
    )

    print(galaxy_centre)

stop

# galaxy_centres = [ref3_to_deg(ra=ra, dec=dec, d_ra=0.0, d_dec=0.0) for ra, dec in zip(ra_list, dec_list)]

galaxy_centres = [
    deg_to_ref3(ra_ref=ra_list[0], dec_ref=dec_list[0], ra=ra, dec=dec)
    for ra, dec in zip(ra_list, dec_list)
]

galaxy_centres = [
    deg_to_ref3(ra_ref=ra, dec_ref=dec, ra=ra_list[0], dec=dec_list[0])
    for ra, dec in zip(ra_list, dec_list)
]

print(ra_list)
print(dec_list)

print(galaxy_centres)

stop

# galaxy_centres = al.Grid2DIrregular.from_ra_dec(ra=ra_list, dec=dec_list, origin=bcg_centre)
