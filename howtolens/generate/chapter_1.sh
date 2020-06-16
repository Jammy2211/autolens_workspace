PYAUTO_PATH="/home/jammy/PycharmProjects/PyAuto"
WORKSPACE_PATH=$PYAUTO_PATH"/autolens_workspace"
HOWTOLENS_PATH=$WORKSPACE_PATH"/howtolens"
CHAPTER_PATH=$HOWTOLENS_PATH"/chapter_1_introduction"
SCRIPTS_PATH=$CHAPTER_PATH"/scripts"

export WORKSPACE=$HOWTOLENS_PATH

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

cd HOWTOLENS_PATH/generate