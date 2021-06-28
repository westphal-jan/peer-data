# peer-data
## Setup
To get the submodule do `git submodule update --init --recursive`
To create the conda environment with the required packages, run `conda env create -f environment.yml`.

For server without sudo: you need to install git-lfs from binary. (For powerpc, we need to use different binary and hope it works)
- `cd ~`
- Downlaod binary: `curl -fsSLO https://github.com/git-lfs/git-lfs/releases/download/v2.13.3/git-lfs-linux-386-v2.13.3.tar.gz`
- Unpack: `tar -zvxf ./git-lfs-linux-386-v2.13.3.tar.gz`
- Specify local install dir because no sudo access `mkdir local-git-lfs && export PREFIX=~/local-git-lfs`
- Execute install script: `./install.sh`
- Add to PATH `export PATH=${PATH}:/home/<user>/local-git-lfs/bin`