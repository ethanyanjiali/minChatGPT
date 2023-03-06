# Install driver 515 and CUDA 11.7
sudo apt update
sudo apt install wget
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/515.43.04/NVIDIA-Linux-x86_64-515.43.04.run
sudo sh NVIDIA-Linux-x86_64-515.43.04.run
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run

# Install Python 3.8
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev
sudo apt install lzma liblzma-dev
curl -O https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tar.xz
tar -xf Python-3.8.2.tar.xz
./configure --enable-optimizations
make -j 8
sudo make altinstall

# Python deps
python3.8 -m venv env
source env/bin/activate
pip install pip --upgrade
pip install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
pip install -r requirements.txt
