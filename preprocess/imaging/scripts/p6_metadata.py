"""
__Preprocess 6: - Metadata (Optional)__

Metadata includes auxiliary information about a strong lens system, that one may want to use during an analysis or
when interpreting the results after the analysis.

The most obvious example of metadata is the redshifts of the source and lens galaxy. By storing these as metadata in
the lens's dataset folder, it is then straight forward to load the redshifts in a runner and pass them to a pipeline,
such that PyAutoLens can then output results in physical units (e.g. kpc instead of arc-seconds, solMass instead of
angular units).

The metadata may also be loaded by the aggregator after a model-fit has completed, such that when one is interpreting
the results of a model fit additional data on a lens can be used to, for example to plot the model-results against other
measurements of a lens not made by PyAutoLens. Examples of such data might be:

- The velocity dispersion of the lens galaxy.
- The stellar mass of the lens galaxy.
- The results of previous strong lens models to the lens performed in previous papers.
"""

import shutil
import os
import json

# %%
import autofit as af

# %%
#%matplotlib inline

# %%
"""
Setup the path to the autolens_workspace, using the correct path name below.
"""

# %%
workspace_path = "path/to/AutoLens/autolens_workspace/"
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace/"

preprocess_path = workspace_path + "/preprocess/imaging/"

# %%
"""
The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the metadata are stored in e.g,
the metadata will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/metadata.json'.
"""

# %%
dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"

# %%
"""
Create the path where the metadata will be output, which in this case is
'/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
"""

# %%
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# %%
"""
The metadata is written as a Python dictionary and can have as many entires as desired added to it. Any information you
want to include int he interpretation of your lens models should be included here.
"""

metadata = {
    "redshihft_lens": 0.5,
    "redshift_source": 1.0,
    "velocity_dispersion": 250000,
    "stellar mass": 1e11,
}

# %%
"""
The metadata is stored in the dataset folder as a .json file. 

We cannot 'dump' a .json file using a string which contains a directory, so we dump it to the location of this
script and move it to the appropriate dataset folder. We first delete existing metadata in the dataset folder.
"""

# %%
metadata_file = "metadata.json"

with open(metadata_file, "w+") as f:
    json.dump(metadata, f, indent=4)


if os.path.exists(dataset_path + "metadata.json"):
    os.remove(dataset_path + "metadata.json")

shutil.move("metadata.json", dataset_path + "metadata.json")
