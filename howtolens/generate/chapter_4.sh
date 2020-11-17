echo "Setting up Environment variables."
export PYAUTO_PATH=/mnt/c/Users/Jammy/Code/PyAuto
export PYAUTOLENS_PATH=$PYAUTO_PATH/PyAutoLens
export WORKSPACE_PATH=$PYAUTO_PATH/autolens_workspace
export HOWTOLENS_PATH=$WORKSPACE_PATH/howtolens
export CHAPTER_PATH=$HOWTOLENS_PATH/chapter_4_inversions
export SCRIPTS_PATH=$CHAPTER_PATH/scripts

echo "Removing old notebooks."
rm -rf $HOWTOLENS_PATH/dataset/chapter_4/lens*
rm $CHAPTER_PATH/*.ipynb
rm -rf $CHAPTER_PATH/.ipynb_checkpoints

echo "Running notebooks."
cd $WORKSPACE_PATH
find $WORKSPACE_PATH/config -type f -exec sed -i 's/backend=default/backend=Agg/g' {} +
for f in $SCRIPTS_PATH/*.py; do python "$f"; done
find $WORKSPACE_PATH/config -type f -exec sed -i 's/backend=Agg/backend=default/g' {} +

echo "Converting scripts to notebooks."
cd $SCRIPTS_PATH
py_to_notebook
cp -r *.ipynb ../
rm *.ipynb
cd ..
git add *.ipynb
rm __init__.ipynb

echo "Renaming cd magicmethod"
find $CHAPTER_PATH/*.ipynb -type f -exec sed -i 's/#%cd/%cd/g' {} +

echo "Moving new notebooks to PyAutolens/howtolens folder."
rm -rf $PYAUTOLENS_PATH/howtolens/chapter_4_inversions
cp -r $WORKSPACE_PATH/dataset/howtolens/chapter_4 $PYAUTOLENS_PATH/howtolens/dataset
cp -r $HOWTOLENS_PATH/simulators $PYAUTOLENS_PATH/howtolens
cp -r $CHAPTER_PATH $PYAUTOLENS_PATH/howtolens/
cp $PYAUTOLENS_PATH/__init__.py $PYAUTOLENS_PATH/howtolens/

echo "Adding new **PyAutoLens**/howtolens files to github."
cd $PYAUTOLENS_PATH
git add $PYAUTOLENS_PATH/howtolens/dataset/chapter_4
git add -f $PYAUTOLENS_PATH/howtolens/chapter_4_inversions

echo "returning to generate folder."
cd $HOWTOLENS_PATH/generate