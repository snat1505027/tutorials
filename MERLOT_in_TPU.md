## Create conda environment for MERLOT-Reserve in a new TPU node

### Setup Anaconda in new TPU-VM
1. Download Anaconda 64-Bit (x86) Installer (659 MB) version using this command: 
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
```
2. Create conda envorment and install dependencies with following commands:

```
conda create --name mreserve python=3.8 && conda activate mreserve
conda install -y python=3.8 tqdm numpy pyyaml scipy ipython cython typing h5py pandas matplotlib

# Install jax on TPUs instead of locally... JAX 0.3.15 is supported)
pip install "jax[tpu]>=0.2.18" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# get merlot_reserve codebase in home directory. Unzip the tar file with `tar -xvf merlot_reserve.tar`
cd merlot_reserve
pip install -r requirements.txt
pip3 install python-dotenv
sudo apt-get install parallel
pip install wandb
sudo apt-get install -y libsndfile-dev
sudo apt install ffmpeg
```

3. Install and run `jupyter lab` to modify the code:

```
pip install jupyterlab
jupyter lab --ContentsManager.allow_hidden=True
```
