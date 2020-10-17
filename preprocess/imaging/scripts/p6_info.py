"""
__Preprocess 6: - Info and Pickles (Optional)__

Here, we include auxiliary information about our strong lens dataset that we may use during an analysis or when
interpreting the lens modeling results.

The most obvious example of such information is the redshifts of the source and lens galaxy. By storing these as an
`info` file in the lens`s dataset folder, it is straight forward to load the redshifts in a runner and pass them to a
pipeline, such that PyAutoLens can then output results in physical units (e.g. kpc instead of arc-seconds, solMass
instead of angular units).

The info file may also be loaded by the aggregator after a model-fit has completed, such that when one is interpreting
the results of a model fit additional data on a lens can be used to, for example to plot the model-results against
other measurements of a lens not made by PyAutoLens. Examples of such data might be:

- The velocity dispersion of the lens galaxy.
- The stellar mass of the lens galaxy.
- The results of previous strong lens models to the lens performed in previous papers.
"""

# %%
from pyprojroot import here

workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

#%matplotlib inline

# %%
"""
Setup the path to the autolens_workspace, using the correct path name below.
"""

preprocess_path = f"preprocess/imaging"

# %%
"""
The `dataset label` is the name of the dataset folder and `dataset_name` the folder the info file is stored in e.g,
the info will be output as `/autolens_workspace/dataset/dataset_type/dataset_name/info.json`.
"""

# %%
dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "mass_sie__source_sersic"

# %%
"""
Returns the path where the info will be output, which in this case is
`/autolens_workspace/dataset/imaging/no_lens_light/mass_total__source_bulge`
"""

# %%
dataset_path = f"dataset/{dataset_type}/{dataset_label}/{dataset_name}"

# %%
"""
The info is written as a Python dictionary and can have as many entries as desired added to it. Any information you
want to include int he interpretation of your lens models should be included here.
"""

info = {
    "redshihft_lens": 0.5,
    "setup.redshift_source": 1.0,
    "velocity_dispersion": 250000,
    "stellar mass": 1e11,
}

# %%
"""
The info is stored in the dataset folder as a .json file. 

We cannot `dump` a .json file using a string which contains a directory, so we dump it to the location of this
script and move it to the appropriate dataset folder. We first delete existing info file in the dataset folder.
"""

# %%
import os
import shutil
import json

info_file = "info.json"

with open(info_file, "w+") as f:
    json.dump(info, f, indent=4)

if os.path.exists(f"{dataset_path}/info.json"):
    os.remove(f"{dataset_path}/info.json")

shutil.move("info.json", f"{dataset_path}/info.json")

# %%
"""
For the info to be available to the results of a model-fit, the runner must load the info file from the .json and 
pass it to the phase.run() or pipeline.run() function:

info_file = f"{dataset_path}/info.json"

with open(info_file, "r") as f:
    info = json.load(f)

phase.run(dataset=dataset, mask=mask, info=info)
"""
