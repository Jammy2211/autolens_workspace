PYAUTO_PATH="/home/jammy/PycharmProjects/PyAuto"
WORKSPACE_PATH=$PYAUTO_PATH"/autolens_workspace"
HOWTOLENS_PATH=$WORKSPACE_PATH"/howtolens"
CHAPTER_PATH=$HOWTOLENS_PATH"/chapter_5_hyper_mode"
SCRIPTS_PATH=$CHAPTER_PATH"/scripts"

rm -rf $HOWTOLENS_PATH/chapter_5/dataset/lens*

rm $CHAPTER_PATH/*.ipynb
rm -rf $CHAPTER_PATH/.ipynb_checkpoints

cd $SCRIPTS_PATH

for f in *.py; do python "$f"; done
py_to_notebook
cp -r *.ipynb ../
rm *.ipynb
cd ..
git add *.ipynb
rm __init__.ipynb

export WORKSPACE=/home/jammy/PycharmProjects/PyAuto/autolens_workspace/

cp $CHAPTER_PATH/*ipynb $PYAUTO_PATH/PyAutoLens/docs/tutorials/chapter_5_hyper_mode/

cd $HOWTOLENS_PATH/generate