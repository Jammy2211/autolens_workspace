import os
import sys

### AUTOFIT + CONFIG SETUP ###

import autofit as af

### NOTE - if you have not already, complete the setup in 'autolens_workspace/runners/cosma/setup' before continuing with this
### cosma pipeline script.

# Welcome to the Cosma pipeline runner. Hopefully, you're familiar with runners at this point and have been using them
# with PyAutoLens to model lenses on your laptop. If not, I'd recommend you get used to doing that, before trying to
# run PyAutoLens on a super-computer. You need some familiarity with the software and lens modeling before trying to
# model a large sample of lenses on a supercomputer!

# This exmaple assumes you are using cosma7 and outputting results to the cosma7 output directory:

# '/cosma7/data/dp004/cosma_username/'.

# If you are not using cosma7 but a different machine, you need to change this path accordingly.

# If you are ready, then let me take you through the Cosma runner. It is remarkably similar to the ordinary pipeline
# runners you're used to, however it makes a few changes for running jobs on cosma:

# 1) The dataset path is over-written to the path '/cosma7/data/dp004/cosma_username/dataset_label' as opposed to the
#    autolens_workspace. As we discussed, on cosma we don't store our data in our autolens_workspace.

# 2) The output path is over-written to the path '/cosma7/data/dp004/cosma_username/output' as opposed to
#    the autolens_workspace. This is for the same reason as the dataset.

# We need to specify where our dataset is stored and output is placed, which uses your cosma user (change this below).
cosma_username = "cosma_username"

# Lets use this to setup our cosma path, which is where our dataset and output are stored.

# (If you are not using cosma7 you need to edit the path below).

cosma_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path="/cosma7/data/dp004/", folder_names=[cosma_username]
)

# The dataset folder is the name of the folder our dataset is stored in, which in this case is 'imaging'. This should
# be named after your sample of lenses (e.g. 'slacs', 'bells').
dataset_label = "imaging"

# Next, lets use this path to setup the dataset path, which for this example is named 'imaging' and found at
# '/cosma7/data/dp004/cosma_username/dataset/imaging/'
cosma_dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=cosma_path, folder_names=["dataset", dataset_label]
)

# We'll do the same for our output path, which is
# '/cosma7/data/dp004/cosma_username/output/imaging/'
cosma_output_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=cosma_path, folder_names=["output", dataset_label]
)

# Next, we need the path to our Cosma autolens_workspace, which can be generated using a relative path assuming our
# runner is located in our Cosma autolens_workspace.
workspace_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"

# Lets now use the above paths to set the config path and output path for our Cosma run.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=cosma_output_path
)

# On Cosma, there are two systems we can submit to, called 'cordelia' and 'cosma'. The similarities and differences
# between these systems are as follows:

# - For both, each job is sent to a 'node' which consists of 16 CPU's. Typically, we'll submit a different PyAutoLens
#   job to each CPU, therefore we'll ideally have a sample of 16 images where each runs on an independent CPU. We'll
#   cover running parallel jobs across multiple CPUs in a later tutorial.

# - A cordelia run has a maximum of 30GB memory, whereas a cosma run's maximum is 128GB. Thus, for high memory jobs
#   (e.g. high resolution imaging), you should use the cosma queue.

# - Cordelia queues are often less busy, so if you have a low memory job, your Cosma run will probably started quicker
#   there. We'll set you up with scripts for both queues, and if one is busy always try the other!

# Cosma and Cordelia submissions require different 'batch scripts'. A batch script basically tells Cosma the PyAutoLens
# job your sending it, and distributes it to nodes and CPUs. Lets first look

# Lets first take a look at a cordelia batch script, which can be found at
# 'autolens_workspace/runners/cosma/batch_cordelia/example' (we'll cover cosma batch scripts at the end of this runner). In the
# same folder, you should see a file 'doc', which will example what each line does.

# When we submit a PyAutoLens job to Cosma, we submit a 'batch' of jobs, whereby each job will run on one CPU of Cosma.
# Thus, if our lens sample contains, lets say, 4 lenses, we'd submit 4 jobs at the same time where each job applies
# our pipeline to each image.

# The fifth line of this batch script - '#SBATCH --arrays=1-8' is what species this. Its telling Cosma we're going to
# run 4 jobs, and the id's of those jobs will be numbered from 1 to 8. Infact, these ids are passed to this runner,
# and we'll use them to ensure that each jobs loads a different image. Lets get the cosma arrays id for our job.
cosma_array_id = int(sys.argv[1])

# Now, I just want to really drive home what the above line is doing. For every job we run on Cosma, the cosma_array_id
# will be different. That is, job 1 will get a cosma_array_id of 1, job 2 will get an id of 2, and so on. This is our
# only unique identifier of every job, thus its our only hope of specifying for each job which image they load!

### AUTOLENS + DATA SETUP ###

import autolens as al

# We're used to specifying the dataset name as a string, so that our pipeline can be applied to multiple
# images with ease. On Cosma, we can apply the same logic, but put these strings in a list such that each Cosma job
# loads a different lens name based on its ID. neat, huh?

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

# Lets extract the dataset name explicitly using our cosma_array_id (rememeber, each job has a different cosma_array_id,
# so each will be given a different dataset_name string).
dataset_name = dataset_name[cosma_array_id]

pixel_scales = 0.2  # Make sure your pixel scale is correct!

# We now use the dataset_name to load a the dataset on each job. The statement below combines
# the cosma_dataset_path and and dataset_name to read dataset_label from the following directory:
# '/cosma7/data/dp004/cosma_username/dataset/imaging/dataset_name'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=cosma_dataset_path, folder_names=[dataset_name]
)

# This loads the imaging dataset, as per usual.
imaging = al.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
)

# Next, we create the mask we'll fit this data-set with.
mask = al.mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

### PIPELINE SETUP ###

# I will assume that you are familiar with pipeline setup from the normal runner scripts.

general_setup = al.setup.General(with_shear=True)

source_setup = al.setup.Source(
    pixelization=al.pix.VoronoiMagnification, regularization=al.reg.Constant
)

from pipelines.beginner.no_lens_light import lens_sie__source_inversion

pipeline = lens_sie__source_inversion.make_pipeline(
    setup=setup, phase_folders=["beginner", dataset_label, dataset_name]
)

pipeline.run(dataset=imaging, mask=mask)

### BATCH SCRIPTS ###

# Finally, lets quickly look at the batch_cosma folder, for submitting jobs to cosma. There really isn't much different
# from a cordelia batch script. It is simply that the last line has to use the 'srun' command to distribute each job
# on the node to an individual CPU. This is a subtley of how scripts run on Cosma, and isn't something you should
# worry too much about.

# At this point, you should be ready to create your own Cosma runner and batch scripts. In general, this is done
# by doing the following:

# - Copy and paste this example.py script and rename it to your new runner (e.g. 'slacs.py').
# - Either keep the data_folder_name as 'share', or change to your cosma username.
# - Change the 'dataset_label' from example to your dataset folder (e.g. 'slacs', 'bells').
# - Change the list of data_names to your dataset names (e.g. 'slacs0123+4567, 'bells3141+5926')
# - Change the imported pipeline from the example above to the one you want to use.
# - Create new batch scripts, by copying the example scripts and changing the job name, task number, arrays ids and
#   last line which points to the name of your runner. For cosma batch scripts, you'll need to copy, paste and edit
#   both the batch script and multiconf.conf scripts.
