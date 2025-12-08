# HPC/Slurm Cluster Setup Guide

This guide provides instructions for setting up the Bhaashik translation environment on HPC clusters with Slurm job scheduling.

## Prerequisites

- Access to an HPC cluster with Slurm
- Miniconda installed in your home directory (recommended) or module available
- Internet access from compute nodes (or pre-downloaded packages)

**Important:** If you're experiencing issues, see [HPC-TROUBLESHOOTING.md](HPC-TROUBLESHOOTING.md) for detailed solutions.

## Environment Setup

### Option 1: Using environment.yml (Recommended)

```bash
# If using your own Miniconda installation in home directory
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh

# OR if using a module (check with: module avail)
# module load miniconda3

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate bhaashik-translation
```

**Note:** The environment.yml is optimized for HPC clusters with older GCC versions. It uses:
- Python 3.10 (better compatibility)
- NumPy and Pandas from conda (pre-built, no compilation)
- All other packages from pip

### Option 2: Manual Installation

If the environment file fails, you can install packages manually:

```bash
# Create base environment
conda create -n bhaashik-translation python=3.12 -y

# Activate environment
conda activate bhaashik-translation

# Install conda packages
conda install -c conda-forge pyyaml requests tqdm pip -y

# Install pip packages
pip install openai anthropic conllu python-dotenv colorama rich click loguru pandas sacrebleu nltk httpx aiohttp pytest pytest-asyncio pytest-cov
```

### Option 3: Using pip only (if conda fails)

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install all packages via pip
pip install -r requirements.txt
```

## Configuration

### 1. Set up API keys

Create a `.env` file in the project root:

```bash
# OpenAI API key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Other API keys
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_CLOUD_API_KEY=your_google_key_here
AZURE_API_KEY=your_azure_key_here
```

### 2. Configure paths

Edit translation configuration files to use appropriate paths for your cluster:

```yaml
# In config files, use absolute paths like:
input_dir: /scratch/user/bhaashik/input
output_dir: /scratch/user/bhaashik/output
cache_dir: /scratch/user/bhaashik/cache
```

## Running Translation Jobs

### Interactive Session

```bash
# Request interactive node
srun -N 1 -n 1 -c 4 --mem=8G --time=02:00:00 --pty bash

# Activate environment
conda activate bhaashik-translation

# Run translation
python -m universal_translate.translate \
    --input input/converted/Hindi/plain-text/by_file/AGRICULTURE/ \
    --output output/Bhojpuri/ \
    --source-lang Hindi \
    --target-lang Bhojpuri \
    --provider openai \
    --model gpt-4o-mini
```

### Batch Job

Create a Slurm job script `translate_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=bhaashik-translate
#SBATCH --output=logs/translate_%j.out
#SBATCH --error=logs/translate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=general

# Load conda module
module load miniconda3

# Activate environment
conda activate bhaashik-translation

# Set up environment variables
export OPENAI_API_KEY=$(cat ~/.openai_key)

# Run translation
python -m universal_translate.translate \
    --input input/converted/Hindi/plain-text/by_file/AGRICULTURE/ \
    --output output/Bhojpuri/ \
    --source-lang Hindi \
    --target-lang Bhojpuri \
    --provider openai \
    --model gpt-4o-mini \
    --batch-size 50 \
    --max-workers 4

echo "Translation job completed"
```

Submit the job:

```bash
sbatch translate_job.sh
```

### Monitor Job Progress

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f logs/translate_<job_id>.out

# View error logs
tail -f logs/translate_<job_id>.err
```

## Common Issues and Solutions

### Issue 1: Conda Environment Creation Fails with Container Errors

**Error**: `Missing system requirements... user.max_user_namespaces = 28633`

**Cause**: The old environment.yml file included PyTorch with CUDA, triggering rootless container requirements.

**Solution**: Use the updated environment.yml file which removes PyTorch dependencies.

### Issue 2: No Internet Access from Compute Nodes

**Solution**: Pre-download packages on login node, then transfer:

```bash
# On login node
conda env create -f environment.yml --download-only

# Then create environment on compute node
srun conda env create -f environment.yml --offline
```

### Issue 3: Module Load Failures

**Solution**: Check available modules and versions:

```bash
module avail conda
module avail python
```

Load the appropriate module before creating the environment.

### Issue 4: API Rate Limits

**Solution**: Adjust batch size and add delays:

```python
# In your translation script
--batch-size 20  # Reduce from default 50
--delay 1.0      # Add 1 second delay between requests
```

### Issue 5: Disk Space Issues

**Solution**: Use scratch space for cache:

```bash
# Set cache directories to scratch
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/transformers
```

## Performance Optimization

### Parallel Processing

For large datasets, split input into chunks and run parallel jobs:

```bash
# Split input by domain
sbatch translate_job.sh --input AGRICULTURE --output Bhojpuri/AGRICULTURE
sbatch translate_job.sh --input DISEASE --output Bhojpuri/DISEASE
sbatch translate_job.sh --input ENTERTAINMENT --output Bhojpuri/ENTERTAINMENT
```

### Resource Allocation

Recommended resources per job:
- **CPU**: 4 cores (for parallel file processing)
- **Memory**: 8 GB (API-based translation is lightweight)
- **Time**: 24 hours (for ~10,000 sentences with API delays)

For larger datasets:
```bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
```

## Checking Installation

Verify your installation:

```bash
# Activate environment
conda activate bhaashik-translation

# Check Python version
python --version  # Should be 3.12.x

# Check installed packages
pip list | grep -E "openai|anthropic|conllu"

# Test import
python -c "import openai; import conllu; print('✓ Installation successful')"
```

## Support

If you encounter issues not covered here:
1. Check the main README.md
2. Review CLAUDE.md for project architecture
3. Open an issue on GitHub with:
   - Your cluster's specifications
   - Error messages
   - Slurm job logs

## Differences from Local Setup

Key differences when running on HPC:
- No GPU/CUDA required (API-based translation)
- Use scratch storage for large files
- Respect cluster policies for:
  - Job time limits
  - API rate limits
  - Network usage
  - Storage quotas
