# %%
"""
__WELCOME__ 

Welcome to a cosma modeling script Python script, which illustrates how to load a strong lens dataset and analyse it on cosma.

This example illustrates how to fit a single dataset with a parallelized Nautilus model-fit. You should
only read this example after reading and understanding this example.

All aspects of this script which are explained in `example_0.py`, for example setting up the cosma dataset and output
directories, are not rexplained in this script. Therefore, if anything does not make sence refer back to `example_0.py`
for an examplanation.
"""

# %%
"""
__COSMA PATHS SETUP__

All of the code below is a repeat of `example_0.py`
"""

from os import path

cosma_path = path.join(path.sep, "cosma7", "data", "dp004", "cosma_username")

dataset_folder = "example"
dataset_name = "simple__no_lens_light"

cosma_dataset_path = path.join(cosma_path, "dataset", dataset_folder, dataset_name)

cosma_output_path = path.join(cosma_path, "output")

workspace_path = "/cosma/home/dp004/cosma_username/autolens_workspace/"

config_path = path.join(workspace_path, "cosma", "config")

from autoconf import conf

conf.instance.push(new_path=config_path, output_path=cosma_output_path)

"""
Cosma submissions require a`batch script`, which tells Cosma the PyAutoLens runners you want it to execute and 
distributes them to nodes and CPUs. 

In this previosu example, the batch script ran a multi-program submission which set off many jobs on single CPUs. 

By inspecting the batch script `autolens_workspace/misc/hpc/cosma/batch/example_1` one can see that only the last line 
has changed, from: 

    srun -n 16 --multi-prog conf/example.conf
    
Too:

    python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 1

This is straight forward to understand, instead of calling a `.conf` file and passing many `python3` commands to set
off multiply jobs we now simply set off a single `python3` command in the batch script. As a result, the batch script
`example_1` has no corresponding `example_1.conf` file.
    
We still pass the integer on the right which is used  to load a specific dataset. This is somewhat optional, but it is
benefitial for scripts which perform single-CPU fits or multi-CPU Nautilus fits to use the same code to load
data.
"""
import sys

cosma_id = int(sys.argv[1])

"""
There is only one more change to the modeling script script that is necessary, which we explain below.

All remaining code is repetition of `example_0.py`.
"""

dataset_type = "imaging"
pixel_scales = 0.1

dataset_name = []
dataset_name.append("example_image_1")  # Index 0
dataset_name.append("example_image_2")  # Index 1
dataset_name.append("example_image_3")  # Index 2
dataset_name.append("example_image_4")  # Index 3
dataset_name.append("example_image_5")  # Index 4
dataset_name.append("example_image_6")  # Index 5
dataset_name.append("example_image_7")  # Index 6
dataset_name.append("example_image_8")  # Index 7
# ...and so on.

dataset_name = dataset_name[cosma_id]

dataset_path = path.join(cosma_dataset_path, dataset_type, dataset_name)

import autofit as af
import autolens as al

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

# Lens:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search__

Here is where we differ from `example_0.py`. 

The only change is that the `number_of_cores` input into `Nautilus` is now 16.


"""
search = af.Nautilus(
    path_prefix=path.join("cosma_example"),
    name="mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=16,
)

"""
All code from here is repeated from `example_0.py`.
"""
analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

In this example, we manually coded in the line `number_of_cores=16`.

We could make this a command line input of the `python3` command in the batch script. For example, if the last line
of the batch script read:

    python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 1 16

We could pass the `number_of_cores` using the command:

    number_of_cores=int(sys.argv[2])


It should also be noted that one cannot combine a `.conf` submission script with multi-CPU Nautilus parallelization.

Which this should, in principle, be possible, the Python multi-processing library does not seem to happy about it when
we do this. So, just don't bother!
"""
