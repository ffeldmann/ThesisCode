ENV_NAME="${ENV_NAME:-ffg_test}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# create environment
echo "################### create env $ENV_NAME ###################"
conda create --name $ENV_NAME python=3.6
echo "################### activate env $ENV_NAME ###################"
source activate $ENV_NAME

echo "################### install torch pytorch cuda ###################"
conda install pytorch=1.3.1 torchvision cudatoolkit=10.1 -c pytorch
echo "################### install pip requirements ###################"
pip install -r requirements.txt
# pip uninstall torch torchvision -y

echo "################### show differences to requirements.txt ###################"
pip freeze | diff requirements.txt -

echo "################### Install Forward-Warp layer ###################"
cd src/forward-warp
cd Forward_Warp/cuda
rm -rf *_cuda.egg-info build dist __pycache__
pip install -e .
cd ../..
rm -rf *_cuda.egg-info build dist __pycache__
pip install -e .
cd ../..

echo "################### show differences to requirements.txt ###################"
pip freeze | diff requirements.txt -

#install apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git#egg=apex

echo "################### Install flownet2_pytorch layers ###################"
cd AnimalPose/models/flownet2_pytorch
cd ./networks/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
pip install -e .
cd ../resample2d_package
rm -rf *_cuda.egg-info build dist __pycache__
pip install -e .
cd ../channelnorm_package
rm -rf *_cuda.egg-info build dist __pycache__
pip install -e .
cd ../..
cd ../../..

echo "################### show differences to requirements.txt ###################"
pip freeze | diff requirements.txt -

echo "################### run tests ###################"
python -m pytest tests/
