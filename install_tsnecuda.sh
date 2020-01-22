# Change CUDA version if needed
conda install faiss-gpu cudatoolkit=10.0 -c pytorch
sudo apt install libopenblas-dev

wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2
tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'lib/*'
tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'site-packages/*'

# Fill your username below. Change python version if needed
cp -r site-packages/* /home/username/anaconda3/lib/python3.7/site-packages/
sudo mkdir -p /usr/local/cuda/lib64
cp ./lib/libfaiss.so /usr/local/cuda/lib64/

# To force update the library path
sudo echo /usr/local/cuda/lib64/ >> /etc/ld.so.conf
sudo ldconfig
