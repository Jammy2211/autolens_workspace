export WORKSPACE=/Users/Jammy/Code/PyAuto/autolens_workspace/howtolens/
rm -rf *.ipynb
rm -rf .ipynb_checkpoints
cd scripts
for f in *.py; do python "$f"; done
py_to_notebook
cp -r *.ipynb ../
rm *.ipynb
cd ..
git add *.ipynb
export WORKSPACE=/Users/Jammy/Code/PyAuto/autolens_workspace/
