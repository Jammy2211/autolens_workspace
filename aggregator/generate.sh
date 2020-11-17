echo "Setting up Environment variables."
export PYAUTO_PATH=/home/jammy/PycharmProjects/PyAuto
export PYAUTOLENS_PATH=$PYAUTO_PATH/PyAutoLens
export WORKSPACE_PATH=$PYAUTO_PATH/autolens_workspace
export AGGREGATOR_PATH=$WORKSPACE_PATH/advanced/aggregator
export SCRIPTS_PATH=$AGGREGATOR_PATH/scripts

echo "Removing old notebooks."

echo "Converting scripts to notebooks."
cd $SCRIPTS_PATH

find $WORKSPACE_PATH/config -type f -exec sed -i 's/backend=default/backend=Agg/g' {} +

for f in *.py; do python "$f"; done
py_to_notebook
cp -r *.ipynb ../
rm *.ipynb
cd ..
git add *.ipynb
rm __init__.ipynb

find $WORKSPACE_PATH/config -type f -exec sed -i 's/backend=Agg/backend=default/g' {} +

echo "returning to generate folder."
cd $AGGREGATOR_PATH