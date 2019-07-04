import autofit as af
from autolens.data import ccd

import os
import sys

### NOTE - if you have not already, complete the setup in 'workspace/runners/cosma/setup' before continuing with this
### cosma pipeline script.

# Welcome to the Cosma pipeline runner. Hopefully, you're familiar with runners at this point, and have been using them
# with PyAutoLens to model lenses on your laptop. If not, I'd recommend you get used to doing that, before trying to
# run PyAutoLens on a super-computer. You need some familiarity with the software and lens modeling before trying to
# model a large sample of lenses on a supercomputer!

# If you are ready, then let me take you through the Cosma runner. It is remarkably similar to the ordinary pipeline
# runners you're used to, however it makes a few changes for running jobs on cosma:

# 1) The data path is over-written to the path '/cosma5/data/autolens/cosma_username/data' as opposed to the
#    workspace. As we discussed in the setup, on cosma we don't store our data in our workspace.

# 2) The output path is over-written to the path '/cosma5/data/autolens/cosma_username/output' as opposed to
#    the workspace. This is for the same reason as the data.

# We need to specify where our data is stored and output is placed. In this example, we'll use the 'share' folder, but
# you can change the string below to your cosma username if you want to use your own personal directory.
data_folder_name = 'share'
# data_folder_name = 'cosma_username'

# Lets use this to setup our cosma path, which is where our data and output are stored.
cosma_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path='/cosma5/data/autolens/', folder_names=[data_folder_name])

# The data folder is the name of the folder our data is stored in, which in this case is 'example'. I would typically
# expect this would be named after your sample of lenses (e.g. 'slacs', 'bells'). If you are modelng just one
# lens, it may be best to omit the data_type.
data_type = 'example'

# Next, lets use this path to setup the data path, which for this example is named 'example' and found at
# '/cosma5/data/autolens/share/data/example/'
cosma_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=cosma_path, folder_names=['data', data_type])

# We'll do the same for our output path, which is '/cosma5/data/autolens/share/output/example/'
cosma_output_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=cosma_path, folder_names=['output', data_type])

# Next, we need the path to our Cosma workspace, which can be generated using a relative path given that our runner is
# located in our Cosma workspace.
workspace_path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Lets now use the above paths to set the config path and output path for our Cosma run.
af.conf.instance = af.conf.Config(
    config_path=workspace_path + 'config', output_path=cosma_output_path)

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
# 'workspace/runners/cosma/batch_cordelia/example' (we'll cover cosma batch scripts at the end of this runner). In the
# same folder, you should see a file 'doc', which will example what each line does.

# When we submit a PyAutoLens job to Cosma, we submit a 'batch' of jobs, whereby each job will run on one CPU of Cosma.
# Thus, if our lens sample contains, lets say, 4 lenses, we'd submit 4 jobs at the same time where each job applies
# our pipeline to each image.

# The fifth line of this batch script - '#SBATCH --array=1-8' is what species this. Its telling Cosma we're going to
# run 4 jobs, and the id's of those jobs will be numbered from 1 to 8. Infact, these ids are passed to this runner,
# and we'll use them to ensure that each jobs loads a different image. Lets get the cosma array id for our job.
cosma_array_id = int(sys.argv[1])

# Now, I just want to really drive home what the above line is doing. For every job we run on Cosma, the cosma_array_id
# will be different. That is, job 1 will get a cosma_array_id of 1, job 2 will get an id of 2, and so on. This is our
# only unique identifier of every job, thus its our only hope of specifying for each job which image they load!

# We're used to specifying the data name as a string, so that our pipeline can be applied to multiple
# images with ease. On Cosma, we can apply the same logic, but put these strings in a list such that each Cosma job
# loads a different lens name based on its ID. neat, huh?

data_name = []
data_name.append('') # Task number beings at 1, so keep index 0 blank
data_name.append('example_image_1') # Index 1
data_name.append('example_image_2') # Index 2
data_name.append('example_image_3') # Index 3
data_name.append('example_image_4') # Index 4
data_name.append('example_image_5') # Index 5
data_name.append('example_image_6') # Index 6
data_name.append('example_image_7') # Index 7
data_name.append('example_image_8') # Index 8

# Lets extract the data name explicitly using our cosma_array_id (rememeber, each job has a different cosma_array_id,
# so each will be given a different data_name string).
data_name = data_name[cosma_array_id]

pixel_scale = 0.2 # Make sure your pixel scale is correct!

# We now use the data_name to load a the data-set on each job. The statement below combines
# the cosma_data_path and and data_name to read data from the following directory:
# '/cosma5/data/autolens/share/data/example/data_name'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=cosma_data_path, folder_names=[data_name])

# This loads the CCD imaging data, as per usual.
ccd_data = ccd.load_ccd_data_from_fits(
    image_path=data_path + 'image.fits',
    psf_path=data_path + 'psf.fits',
    noise_map_path=data_path + 'noise_map.fits',
    pixel_scale=pixel_scale)

from workspace.pipelines.no_lens_light.initialize import lens_sie_shear_source_sersic
from workspace.pipelines.no_lens_light.power_law.from_initialize import lens_pl_shear_source_sersic
from workspace_jam.pipelines.advanced.no_lens_light.subhalo.from_power_law import lens_pl_shear_subhalo_source_sersic

pipeline_initializer = lens_sie_shear_source_sersic.make_pipeline(
    phase_folders=[data_type, data_name])

pipeline_power_law = lens_pl_shear_source_sersic.make_pipeline(
    phase_folders=[data_type, data_name])

pipeline_subhalo = lens_pl_shear_subhalo_source_sersic.make_pipeline(
    phase_folders=[data_type, data_name])

pipeline = pipeline_initializer + pipeline_power_law + pipeline_subhalo

pipeline.run(data=ccd_data)

# Finally, lets quickly look at the batch_cosma folder, for submitting jobs to cosma. There really isn't much different
# from a cordelia batch script. It is simply that the last line has to use the 'srun' command to distribute each job
# on the node to an individual CPU. This is a subtley of how scripts run on Cosma, and isn't something you should
# worry too much about.

# At this point, you should be ready to create your own Cosma runner and batch scripts. In general, this is done
# by doing the following:

# - Copy and paste this example.py script and rename it to your new runner (e.g. 'slacs.py').
# - Either keep the data_folder_name as 'share', or change to your cosma username.
# - Change the 'data_type' from example to your data folder (e.g. 'slacs', 'bells').
# - Change the list of data_names to your data names (e.g. 'slacs0123+4567, 'bells3141+5926')
# - Change the imported pipeline from the example above to the one you want to use.
# - Create new batch scripts, by copying the example scripts and changing the job name, task number, array ids and
#   last line which points to the name of your runner. For cosma batch scripts, you'll need to copy, paste and edit
#   both the batch script and multiconf.conf scripts.