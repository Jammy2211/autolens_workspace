import autolens as al

from typing import List


def lens_catalogue_to_lists(file: str) -> List[List]:
    """
    Convert a lens catalogue file to lists of every individual entry in the catalogue.

    The example lens catalogue distributed with **PyAutoLens** has entries for each lens galaxy on every row, where
    each column gives the following:

    1) The ID of the galaxy (e.g. 2).
    2) The RA coordinate of the galaxy (e.g. 177.94166154725815).
    3) The DEC coordinate of the galaxy (e.g. 33.23750122888889).
    4) The semi-major axis of the galaxy ellipticity estimate (e.g. 0.000096).
    5) The semi-minor axis of the galaxy ellipticity estimate (e.g. 0.000078).
    6) The positon angle theta of the ellipse (e.g. -85.400000).
    7) The magnitude of the galaxy (e.g. 18.062900).

    This function converts this catalogue into a list of lists.
    """

    with open(file) as f:
        l = f.read().split("\n")

    combined = list(zip(*[item.split("    ") for item in filter(lambda item: item, l)]))
    return [list(map(int, combined[0]))] + [
        list(map(float, column)) for column in combined[1:]
    ]


def source_catalogue_to_lists(file: str) -> List[List]:
    """
    Convert a source catalogue file to lists of every individual entry in the catalogue.

    The example source catalogue distributed with **PyAutoLens** has entries for every multiple image of every source
    galaxy on every row, with spaces between groups of rows corresponding to a different source galaxy.

    The first column gives the unique id of every source galaxy, which is by the `*_per_source` functions to group
    certain entries into lists for each source galaxy.

    Each column gives the following:

    1) The ID of the source (e.g. 1).
    2) The RA coordinate of the multiple image (e.g. 177.9988491).
    3) The DEC coordinate of the multiple image (e.g. 33.22729236).
    4) The error on the position's x measurement, which is just the image pixel scale (e.g. 0.1).
    5) The error on the position's y measurement, which is just the image pixel scale (e.g. 0.1).
    6) A value of 0.0, for some reason.
    7) The redshift of the source galaxy.
    8) Anotehr 0., for some reason.

    This function converts this catalogue into a list of lists.
    """
    with open(file) as f:
        l = f.read().split("\n")

    combined = list(zip(*[item.split(" ") for item in filter(lambda item: item, l)]))

    return [list(map(int, combined[0]))] + [
        list(map(float, column)) for column in combined[1:]
    ]


def list_per_source_from(value_list: List,) -> List[int]:
    """
    Create a list of entries per source, where each list index corresponds to a source galaxy in the catalogue. This
    is performed by iterating the full list of source ids and only keeping the unique values.

    This is used, for example, to create a list of ids per source.

    Parameters
    ----------
    value_list
        The list of values from the source galaxy catalogue (e.g galaxy ids, redshifts) that are converted to a list
        on a per galaxy basis.
    """

    id_per_source = []

    for id in value_list:

        if id not in id_per_source:

            id_per_source.append(id)

    return id_per_source


def list_of_lists_per_source_from(id_list, value_list) -> List[List]:
    """
    Create a list of lists of the values from the source galaxy catalogue that correspond to multiple entries of
    the souerce (e.g. its multiple images), where this list is indexed on a per source basis. This works as follows:

    1) Calculate the number of unique ids in the source galaxy catalogue. This tells us how many source galaxies are
    in our catalogue and thus how many unique entries the list containing their multiple images requires. For example,
    if there are 4 unique sources, there will be 4 lists of multiple images points).

    2) Scale the unique ids so they run from 0 -> max_id, which makes it easier to construct the list of lists.

    3) Create the list of lists, by passing every multiple image to its appropriate list.
    """

    unique_id_total = len(set(id_list))
    list_of_lists_per_source = [[] for i in range(unique_id_total)]

    id_list_from_zero = [id - min(id_list) for id in id_list]

    for multiple_image, id_from_zero in zip(value_list, id_list_from_zero):

        list_of_lists_per_source[id_from_zero].append(multiple_image)

    return list_of_lists_per_source
