# %%
"""
__WELCOME__ 

Welcome to the cosma pipeline runner, which illustrates how to load a strong lens dataset and analyses it on cosma

This uses the pipeline (Check it out full description of the pipeline):

 `autolens_workspace/pipelines/imaging/no_lens_light/lens_sie__source_inversion.py`.

We will omit details of what the pipeline and analysis does, focusing on cosma usage.
"""

# %%
"""
Setup the path to the cosma output directory.

This exmaple assumes you are using cosma7 and outputting results to the cosma7 output directory:

 `/cosma7/data/dp004/cosma_username`.
"""

cosma_path = "/cosma7/data/dp004/cosma_username"

"""
Use this path to set the path to the dataset directory on COSMA, as well as the folders within this directory the .fits
are stored in.
"""

dataset_folder = "example"
dataset_name = "mass_sie__source_sersic"

cosma_dataset_path = f"{cosma_path}/dataset/{dataset_folder}/{dataset_name}"

"""
Now use it to set the output path on COSMA.
"""

cosma_output_path = f"{cosma_path}/output"

# %%
"""In contrast to the dataset and output folders, our workspace path is in your COSMA home directory.."""
workspace_path = "/cosma/home/dp004/cosma_username/autolens_workspace/"

"""Use this to set the path to the cosma config files in your COSMA home directory.r"""
config_path = f"{workspace_path}/cosma/config"

# %%
"""Set the config and output paths using autoconf, as you would for a laptop run."""

# %%
from autoconf import conf

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

"""
Cosma submissions require a`batch script`, which tells Cosma the PyAutoLens runners you want it to execute and 
distributes them to nodes and CPUs. Lets look at the batch script 

 `autolens_workspace/cosma/batch/example
    
The following options are worth noting:

 `#SBATCH -N 1` - The number of nodes we require, where 1 node contains 28 CPUs.
 `#SBATCH --ntasks=16` - The total number of task we are submitting.
 `#SBATCH --cpus-per-task=1` - The number of tasks per CPU, this should always be 1 for PyAutoLens use.
 `#SBATCH -J example` - The name of the job, which is how it`ll appear on cosma when you inspect it.
 `#SBATCH -o output/output.%A.out` - Python interpreter output is placed in a file in the `output` folder.
 `#SBATCH -o error/error.%A.out` - Python interpreter errors are placed in a file in the `error` folder.
 `#SBATCH -p cosma7` - Signifies we are running the job on COSMA7.
 `#SBATCH -A dp004` - The project code of the submission.
 `#SBATCH -t 48:00:00` - The job will terminate after this length of time (if it does not end naturally).
 `#SBATCH --mail-type=END` - If you input your email, when you`ll get an email about the job (END means once finished).
 `#SBATCH --mail-user=fill@me.co.uk` - The email address COSMA sends the email too.

The following line activates the PyAutoLens virtual environment we set up on cosma for this run:

 `source /cosma/home/dp004/cosma_username/autolens_workspace/activate.sh`

These two lines prevent the NumPy linear algebra libries from using too many resources.
    
    
export CPUS_PER_TASK=1

export OPENBLAS_NUM_THREADS=$CPUS_PER_TASK
export MKL_NUM_THREADS=$CPUS_PER_TASK
export OMP_NUM_THREADS=$CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$CPUS_PER_TASK

This line sets off the job:

    srun -n 16 --multi-prog conf/example.conf

Lets checkout the file `example.conf`:

    0 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 1
    1 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 2
    2 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 3
    3 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 4
    4 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 5
    5 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 6
    6 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 7
    7 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 8
    8 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 9
    9 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 10
    10 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 11
    11 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 12
    12 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 13
    13 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 14
    14 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 15
    15 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 16
    
It is literally just lines of python3 commands, setting off our runner scripts! Thus, it should now be clear how
to set off many cosma jobs - just add each runner you want to run to this script. The numbers on the left running from 
0-15 specify the CPU number and should always run from 0, I`ll explain the numbers on the right in a moment.

A lot of the code below is what you are used to in runner scripts not on COSMA. Line 
"""

# %%
""" AUTOLENS + DATA SETUP """

# %%
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
Okay, so what were those numbers on the right hand side of our `example.conf` file doing? They were inputting an integer,
which we load below using the `sys.argv[1]` command. Thi means, when we set off our 16 cosma jobs, each one will have a
unique integer value in the `cosma_id` variable.
"""
import sys

cosma_id = int(sys.argv[1])

"""We can now use this variable to load a specific piece of data for this run!"""

dataset_type = "imaging"
pixel_scales = 0.1

dataset_name = []
dataset_name.append("")  # Task number beings at 1, so keep index 0 blank
dataset_name.append("example_image_1")  # Index 1
dataset_name.append("example_image_2")  # Index 2
dataset_name.append("example_image_3")  # Index 3
dataset_name.append("example_image_4")  # Index 4
dataset_name.append("example_image_5")  # Index 5
dataset_name.append("example_image_6")  # Index 6
dataset_name.append("example_image_7")  # Index 7
dataset_name.append("example_image_8")  # Index 8
# ...and so on.

"""
Now, we extract the dataset name specific to this cosma id, meaning every CPU run will load and fit a different piece of
data.
"""
dataset_name = dataset_name[cosma_id]

# %%
"""
Create the path where the dataset will be loaded from, which in this case is:

 `/cosma7/data/dp004/cosma_username/dataset/imaging/example_image_1`
"""

# %%
dataset_path = af.util.create_path(
    path=cosma_dataset_path, folders=[dataset_type, dataset_name]
)

"""
COMPLETE

Everything below is the usual runner script, with nothing COSMA specifc there! So, with that, have a go at getting
your own COSMA run going!
"""

# %%
"""Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files."""

# %%
imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
)

# %%
"""Next, we create the mask we'll fit this data-set with."""

# %%
mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

# %%
"""Make a quick subplot to make sure the data looks as we expect."""

# %%
aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
__Settings__

The `SettingsPhaseImaging` describe how the model is fitted to the data in the log likelihood function.

These settings are used and described throughout the `autolens_workspace/examples/model` example scripts, with a 
complete description of all settings given in `autolens_workspace/examples/model/customize/settings.py`.

The settings chosen here are applied to all phases in the pipeline.
"""

# %%
settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

# %%
"""
__Pipeline_Setup_And_Tagging__:

For this runner the `SetupPipeline` customizes:

 - The `Pixelization` used by the `Inversion` of this pipeline.
 - The `Regularization` scheme used by of this pipeline.
 - If there is an `ExternalShear` in the mass model or not.

The `SetupPipeline` `tags` the output path of a pipeline. For example, if `no_shear` is True, the pipeline`s output 
paths are `tagged` with the string `no_shear`.

This means you can run the same pipeline on the same data twice (with and without shear) and the results will go
to different output folders and thus not clash with one another!

The `folders` below specify the path the pipeline results are written to, which is:

 `autolens_workspace/output/dataset_type/dataset_name` 
 `autolens_workspace/output/imaging/mass_sie__source_sersic`
"""

# %%
setup = al.SetupPipeline(
    pixelization=al.pix.VoronoiMagnification,
    regularization=al.reg.Constant,
    no_shear=False,
    folders=["pipelines", dataset_type, dataset_label, dataset_name],
)

# %%
"""
__Pipeline Creation__

To create a pipeline we import it from the pipelines folder and run its `make_pipeline` function, inputting the 
*Setup* and *SettingsPhase* above.
"""

# %%
from autolens_workspace.pipelines.imaging.no_lens_light import (
    lens_sie__source_inversion,
)

pipeline = lens_sie__source_inversion.make_pipeline(setup=setup, settings=settings)

# %%
"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""

# %%
pipeline.run(dataset=imaging, mask=mask)
