import numpy as np
import os
import time
import copy
import math
import random
import sys


# input_filename = None


def ref3_to_deg_cat(file_name, output_name="outputfile_deg.txt"):
    """
    Transform a catalog in a lenstool format where the position are set in relatic arcseconds
    In a catalog with the same format where the positions are change to absolutde degrees
    In the lenstool frame it correspond to changing REFERENCE 3 to REFERENCE 0

    Parameters
    ----------
    file_name : string
                Name of the catalog
    output_name : string
                  Name of the output file. By default the name will be outputfile_deg.txt
    """
    core = []
    ##### I READ THE FILE ######
    f = open(file_name, "r")
    fout = open(output_name, "w")
    for line in f:
        row = line.split()
        if (
            row[0] == "#REFERENCE"
        ):  # I unfortunately remove the working previous if statement
            header = row
            RA_ref = float(header[2])
            DEC_ref = float(header[3])
            fout.write("#REFERENCE 0 " + str(RA_ref) + " " + str(DEC_ref) + "\n")
        elif (
            row[0] == "#" and row[1] == "REFERENCE"
        ):  # Lenstool do not produce file like that usually
            header = row
            print("HERE")
            RA_ref = float(header[3])
            DEC_ref = float(header[4])
            fout.write("#REFERENCE 0 " + str(RA_ref) + " " + str(DEC_ref) + "\n")
        elif line != "":  # No empty line
            core.append(row)
    try:
        print(header)
    except UnboundLocalError:
        print("EXIT ERROR : I did not find #REFERENCE 3 RA_ref DEC_ref")
        sys.exit(1)
    except:
        print(" ")
        print(
            "something else went wrong with the header, if you need assistance please contact Guillaume Mahler gmahler@umich.edu"
        )
        print(" ")
    f.close()

    ##### I WRITE THE REST of THE FILE ######
    for line in core[1:]:
        end_str = ""
        nRA, nDEC = ref3_to_deg(RA_ref, DEC_ref, float(line[1]), float(line[2]))
        for el in line[3:]:
            end_str = end_str + el + "    "
        end_str = end_str + "\n"
        # fout.write(line[0]+'    '+str(nRA)+'    '+str(nDEC)+'    '+line[3]+'    '+line[4]+'    '+line[5]+'    '+line[6]+'    '+line[7]+'\n')
        fout.write(line[0] + "    " + str(nRA) + "    " + str(nDEC) + "    " + end_str)

    fout.close()
    return 0


def ref3_to_deg(RA_ref, DEC_ref, d_ra, d_dec):
    """
    Transform a realtiv sky position RA and DEC positions in arcsencs in absolute coordinate using a reference position

    Parameters
    ----------
    RA_ref : float
             Right ascension position of the reference point (in degrees)
    DEC_ref : float
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
    DEC = DEC_ref + d_dec / 3600.0
    RA = RA_ref - d_ra / 3600.0 / np.cos(DEC_ref / 180.0 * np.pi)

    return RA, DEC

    # This is ref0 to ref3 too


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


def deg_to_ref3_cat(file_name, output_name="outputfile_ref.txt"):
    """
    Transform a catalog in a lenstool format where the position are set in absolute coordinate in degrees
    In a catalog with the same format where the positions are change to be relative coordinate in arcsecs
    In the lenstool frame it correspond to changing REFERENCE 0 to REFERENCE 3

    Parameters
    ----------
    file_name : string
                Name of the catalog
    output_name : string
                  Name of the output file. By default the name will be outputfile_deg.txt
    """
    core = []
    ##### I READ THE FILE ######
    f = open(file_name, "r")
    fout = open(output_name, "w")
    for line in f:
        row = line.split()
        if (
            row[0] == "#REFERENCE"
        ):  # I unfortunately remove the working previous if statement
            header = row
            RA_ref = float(header[2])
            DEC_ref = float(header[3])
            fout.write("#REFERENCE 3 " + str(RA_ref) + " " + str(DEC_ref) + "\n")
        elif (
            row[0] == "#" and row[1] == "REFERENCE"
        ):  # Lenstool do not produce file like that usually
            header = row
            RA_ref = float(header[3])
            DEC_ref = float(header[4])
            fout.write("#REFERENCE 3 " + str(RA_ref) + " " + str(DEC_ref) + "\n")
        elif line != "":  # No empty line
            core.append(row)
    try:
        print(header)
    except UnboundLocalError:
        print("EXIT ERROR : I did not find #REFERENCE 0 RA_ref DEC_ref")
        sys.exit(1)
    except:
        print(" ")
        print(
            "something else went wrong with the header, if you need assistance please contact Guillaume Mahler gmahler@umich.edu"
        )
        print(" ")
    f.close()

    ##### I WRITE THE REST of THE FILE ######
    for line in core[1:]:
        end_str = ""
        nRA, nDEC = deg_to_ref3(RA_ref, DEC_ref, float(line[1]), float(line[2]))
        for el in line[3:]:
            end_str = end_str + el + "    "
        end_str = end_str + "\n"
        # fout.write(line[0]+'    '+str(nRA)+'    '+str(nDEC)+'    '+line[3]+'    '+line[4]+'    '+line[5]+'    '+line[6]+'    '+line[7]+'\n')
        fout.write(line[0] + "    " + str(nRA) + "    " + str(nDEC) + "    " + end_str)
    fout.close()
    return 0
