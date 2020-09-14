echo "Setting up Environment variables."
export PYAUTO_PATH=/home/jammy/PycharmProjects/PyAuto
export PYAUTOLENS_PATH=$PYAUTO_PATH/PyAutoLens
export WORKSPACE_PATH=$PYAUTO_PATH/autolens_workspace
export HOWTOLENS_PATH=$WORKSPACE_PATH/howtolens
export CHAPTER_PATH=$HOWTOLENS_PATH/chapter_1_introduction
export SCRIPTS_PATH=$CHAPTER_PATH/scripts

echo "Removing old notebooks."

rm $CHAPTER_PATH/*.ipynb
rm -rf $CHAPTER_PATH/.ipynb_checkpoints

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

echo "Moving new notebooks to PyAutolens/howtolens folder."
rm -rf $PYAUTOLENS_PATH/howtolens/chapter_1_introduction
cp -r $HOWTOLENS_PATH/config $PYAUTOLENS_PATH/howtolens
cp -r $HOWTOLENS_PATH/dataset $PYAUTOLENS_PATH/howtolens
rm $PYAUTOLENS_PATH/howtolens/dataset/chapter_*/*.fits
cp -r $HOWTOLENS_PATH/simulators $PYAUTOLENS_PATH/howtolens
cp -r $CHAPTER_PATH $PYAUTOLENS_PATH/howtolens/
cp $PYAUTOLENS_PATH/__init__.py $PYAUTOLENS_PATH/howtolens/

echo "Renaming import autolens_workspace to just howtolens for Sphinx build."
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/from autolens_workspace./from /g' {} +

echo "Moving new notebooks to PyAutolens/howtolens folder."
cd $PYAUTOLENS_PATH
git add -f $PYAUTOLENS_PATH/howtolens/dataset/chapter_1
git add -f $PYAUTOLENS_PATH/howtolens/chapter_1_introduction

echo "returning to generate folder."
cd $HOWTOLENS_PATH/generate