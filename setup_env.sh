# Create environment
conda create -n vre python=3.8 -y
conda activate vre
conda activate vre

# Install jupyter-lab
conda install -c conda-forge jupyterlab=3.6.2 -y

pip install -r requirements.txt

# Convert ipynb to pdf
sudo apt-get install pandoc
sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-plain-generic