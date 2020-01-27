from profiling.imaging.simulator import makers

# Welcome to the PyAutoLens profiling imaging suite maker. Here, we'll make the suite of imaging data that we use to
# profile PyAutoLens. This consists of the following sets of images:

# A source-only image, where the lens mass is an SIE and the source light is a smooth Exponential.
# A source-only image, where the lens mass is an SIE and source light a cuspy Sersic (sersic_index=3).

# Each image is generated at 5 resolutions, 0.2" (LSST), 0.1" (Euclid), 0.05" (HST), 0.03" (HST), 0.01" (Keck AO).

sub_size = 1
data_resolutions = ["lsst", "euclid", "hst", "hst_up", "ao"]

# To simulator each lens, we pass it a name and call its maker. In the makers.py file, you'll see the
makers.make_lens_sie__source_smooth(
    data_resolutions=data_resolutions, sub_size=sub_size
)
makers.make_lens_sie__source_cuspy(data_resolutions=data_resolutions, sub_size=sub_size)
