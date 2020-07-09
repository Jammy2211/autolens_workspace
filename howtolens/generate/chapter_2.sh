echo "Setting up Enviroment variables."
export PYAUTO_PATH=/home/jammy/PycharmProjects/PyAuto
export PYAUTOLENS_PATH=$PYAUTO_PATH/PyAutoLens
export WORKSPACE_PATH=$PYAUTO_PATH/autolens_workspace
export HOWTOLENS_PATH=$WORKSPACE_PATH/howtolens
export CHAPTER_PATH=$HOWTOLENS_PATH/chapter_2_lens_modeling
export SCRIPTS_PATH=$CHAPTER_PATH/scripts

echo "Removing old notebooks."
rm -rf $HOWTOLENS_PATH/dataset/chapter_2/lens*
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
rm -rf $PYAUTOLENS_PATH/howtolens/chapter_2_lens_modeling
cp -r $HOWTOLENS_PATH/config $PYAUTOLENS_PATH/howtolens
cp -r $HOWTOLENS_PATH/dataset $PYAUTOLENS_PATH/howtolens
rm $PYAUTOLENS_PATH/howtolens/dataset/chapter_*/*.fits
cp -r $HOWTOLENS_PATH/simulators $PYAUTOLENS_PATH/howtolens
cp -r $CHAPTER_PATH $PYAUTOLENS_PATH/howtolens/
cp $PYAUTOLENS_PATH/__init__.py $PYAUTOLENS_PATH/howtolens/

echo "Renaming import autolens_workspace to just howtolens for Sphinx build."
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/from autolens_workspace./from /g' {} +

echo "Renaming all code which runs phases"
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/aplt.FitImaging/# aplt.FitImaging/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/result_local_maxima =/# result_local_maxima =/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/result_custom_priors = phase/# result_custom_priors = phase/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/result_light_traces_mass =/# result_light_traces_mass =/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/phase1_result = phase1.run/# phase1_result = phase1.run/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/phase2_result = phase2.run/# phase2_result = phase2.run/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/phase_2_pass.run/# phase_2_pass.run/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/result_slow = phase_slow/# result_slow = phase_slow/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/result_fast = phase_fast/# result_fast = phase_fast/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/result_light_trace_mass = phase_light_traces/# result_light_trace_mass = phase_light_traces/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/phase_with_custom_mask.run/# phase_with_custom_mask.run/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/phase_with_positions.run/# phase_with_positions.run/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/phase_with_x2_positions.run/# phase_with_x2_positions.run/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/print(result/# print(result/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/agg =/# agg =/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/agg =/# agg =/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/samples = lis/# samples = lis/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/print(samples[0]/# print(samples[0]/g' {} +
find $PYAUTOLENS_PATH/howtolens/ -type f -exec sed -i 's/result =/# result = /g' {} +

echo "Adding new __PyAutoLens__/howtolens files to github."
cd $PYAUTOLENS_PATH
git add $PYAUTOLENS_PATH/howtolens/dataset/chapter_2
git add -f $PYAUTOLENS_PATH/howtolens/chapter_2_lens_modeling

echo "returning to generate folder."
cd $HOWTOLENS_PATH/generate