# HPC Cluster Troubleshooting Guide

This guide addresses common issues when setting up the Bhaashik translation environment on HPC clusters with older system configurations.

## Issue 1: Using Wrong Conda Installation

### Problem
System uses Anaconda from /scratch instead of your Miniconda installation.

```bash
$ which conda
/scratch/username/anaconda3/bin/conda  # Wrong installation
```

### Solution A: Override PATH in ~/.bashrc

Add to your `~/.bashrc` file:

```bash
# Override conda to use Miniconda in home directory
export PATH="$HOME/miniconda3/bin:$PATH"

# Initialize conda for bash shell
source $HOME/miniconda3/etc/profile.d/conda.sh

# Disable auto-activation of base environment (optional)
export CONDA_AUTO_ACTIVATE_BASE=false
```

Reload your shell:

```bash
source ~/.bashrc

# Verify correct conda
which conda
conda --version
```

### Solution B: Use Full Path (No ~/.bashrc Changes)

```bash
# Use full path to your Miniconda
$HOME/miniconda3/bin/conda env create -f environment.yml

# Initialize conda in current session
source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate bhaashik-translation
```

### Solution C: For Slurm Job Scripts

```bash
#!/bin/bash
#SBATCH --job-name=bhaashik-translate
#SBATCH --output=logs/translate_%j.out

# Method 1: Override PATH
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh

# Method 2: Use full path
# source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate bhaashik-translation

# Your commands here...
```

## Issue 2: NumPy/Pandas Build from Source (Old GCC)

### Problem

```
C compiler for the host machine: cc (gcc 4.8.5 "cc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-44)")
Building NumPy from source...
```

This happens when:
- HPC cluster has old GCC (< 7.0)
- pip tries to build NumPy/pandas from source
- Build fails or takes very long

### Solution: The environment.yml has been updated to fix this

The new configuration:
- Uses Python 3.10 (more compatible with older systems)
- Installs numpy and pandas via conda (pre-built binaries)
- Uses numpy < 2.0 (better compatibility)

```yaml
dependencies:
  - python=3.10
  - numpy>=1.24.0,<2.0  # Installed from conda
  - pandas>=2.0.0,<2.2   # Installed from conda
  - pip:
    # Other packages...
```

### If the issue persists:

**Option 1: Load newer GCC module**

```bash
module avail gcc
module load gcc/9.3.0  # or newer version

# Then create environment
conda env create -f environment.yml
```

**Option 2: Use conda for all packages**

Create environment manually:

```bash
# Create environment with all packages from conda
conda create -n bhaashik-translation python=3.10 \
  numpy=1.24 pandas=2.0 pyyaml requests tqdm pip -y

conda activate bhaashik-translation

# Install only pure Python packages via pip (no compilation needed)
pip install --no-deps openai anthropic conllu python-dotenv \
  colorama rich click loguru sacrebleu nltk httpx aiohttp \
  pytest pytest-asyncio pytest-cov
```

**Option 3: Use pre-built wheels**

```bash
# Create base environment
conda create -n bhaashik-translation python=3.10 -y
conda activate bhaashik-translation

# Install numpy/pandas from conda
conda install numpy=1.24 pandas=2.0 -y

# Install everything else via pip with pre-built wheels only
pip install --only-binary :all: openai anthropic conllu \
  python-dotenv colorama rich click loguru sacrebleu nltk \
  httpx aiohttp pytest pytest-asyncio pytest-cov \
  google-cloud-translate azure-ai-translation-text
```

## Issue 3: Internet Access Restricted

### Problem
Compute nodes don't have internet access.

### Solution: Download on Login Node

```bash
# On login node (with internet)
conda env create -f environment.yml --download-only

# Then on compute node (no internet needed)
srun --pty bash
conda env create -f environment.yml --offline
```

Or create a conda-pack archive:

```bash
# On login node: create and pack environment
conda env create -f environment.yml
conda activate bhaashik-translation
conda install conda-pack -y
conda pack -n bhaashik-translation -o bhaashik-translation.tar.gz

# Transfer to compute node and unpack
srun --pty bash
mkdir -p $HOME/conda-envs/bhaashik-translation
tar -xzf bhaashik-translation.tar.gz -C $HOME/conda-envs/bhaashik-translation
source $HOME/conda-envs/bhaashik-translation/bin/activate
conda-unpack
```

## Issue 4: Disk Quota Exceeded

### Problem
Conda packages exceed home directory quota.

### Solution: Use Scratch Space for Conda Packages

```bash
# Set conda to use scratch for packages
export CONDA_PKGS_DIRS="/scratch/$USER/conda/pkgs"
export CONDA_ENVS_PATH="/scratch/$USER/conda/envs"

# Create environment in scratch
conda env create -f environment.yml -p /scratch/$USER/conda/envs/bhaashik-translation

# Activate with full path
conda activate /scratch/$USER/conda/envs/bhaashik-translation
```

Add to ~/.bashrc:

```bash
export CONDA_PKGS_DIRS="/scratch/$USER/conda/pkgs"
export CONDA_ENVS_PATH="/scratch/$USER/conda/envs"
```

## Issue 5: SSL Certificate Verification Errors

### Problem

```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

### Solution

```bash
# Temporary fix (not recommended for production)
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
# Or
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>

# Better: Update certificates
conda install certifi ca-certificates -y
```

## Issue 6: Module Conflicts

### Problem
Other loaded modules conflict with conda environment.

### Solution

```bash
# Purge all modules before activating conda
module purge
module load miniconda3  # Only load conda

# Or in job script:
#!/bin/bash
#SBATCH directives...

module purge
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate bhaashik-translation
```

## Issue 7: Permission Denied Errors

### Problem
Can't write to conda directories.

### Solution

Check and fix permissions:

```bash
# Check ownership
ls -la $HOME/miniconda3

# Fix permissions if needed
chmod -R u+w $HOME/miniconda3

# Ensure .conda directory is writable
mkdir -p $HOME/.conda
chmod -R u+w $HOME/.conda
```

## Minimal Installation (Absolute Minimum)

If all else fails, use only essential packages:

```bash
# Create minimal Python 3.10 environment
conda create -n bhaashik-translation python=3.10 pip -y
conda activate bhaashik-translation

# Install only what's absolutely needed
pip install openai conllu pyyaml python-dotenv tqdm

# Test it works
python -c "import openai, conllu; print('✓ Core packages working')"
```

You can add other packages as needed later.

## Verification Checklist

After installation, verify everything works:

```bash
# Activate environment
conda activate bhaashik-translation

# Check Python version
python --version  # Should be 3.10.x or 3.12.x

# Check critical packages
python -c "import openai; print(f'OpenAI: {openai.__version__}')"
python -c "import anthropic; print(f'Anthropic: {anthropic.__version__}')"
python -c "import conllu; print(f'CoNLL-U: {conllu.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Verify no compilation happened
python -c "import numpy; print(f'NumPy built with: {numpy.__config__.show()}')"
```

## Getting Help

If issues persist:

1. Check cluster documentation: `module help` or cluster wiki
2. Contact cluster support with:
   - Error messages
   - Output of: `module list`, `which conda`, `conda --version`
   - Your environment.yml file
   - GCC version: `gcc --version`
3. Try the minimal installation approach above

## Common HPC Cluster Configurations

### SLURM with CentOS 7 (Old GCC 4.8.5)
- Use Python 3.10
- Install numpy/pandas via conda
- Load gcc/9+ module if available

### SLURM with Rocky Linux 8+
- Can use Python 3.12
- Usually works out of the box

### PBS/Torque Clusters
- Similar issues to SLURM
- Adjust job script directives accordingly
- Use `qsub` instead of `sbatch`

### SGE Clusters
- Use `qsub` with appropriate flags
- May need to specify shell: `#$ -S /bin/bash`
