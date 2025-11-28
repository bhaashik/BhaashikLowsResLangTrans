#!/bin/bash
# ============================================================================
# WSL2 Ubuntu Setup Script for GPT-5 Nano Translation
# ============================================================================
# This script sets up your WSL2 environment for Hindi translation
# Run with: bash setup_wsl2.sh

set -e  # Exit on error

echo "======================================================================"
echo "WSL2 UBUNTU SETUP FOR GPT-5 NANO TRANSLATION"
echo "======================================================================"
echo ""

# ============================================================================
# Check if running in WSL2
# ============================================================================
if ! grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null ; then
    echo "⚠️  Warning: This doesn't appear to be WSL2"
    echo "   Continuing anyway..."
fi

# ============================================================================
# Check Conda
# ============================================================================
echo "Step 1: Checking Conda installation..."
echo "----------------------------------------------------------------------"

if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found!"
    echo ""
    echo "Please install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    exit 1
else
    echo "✅ Conda found: $(which conda)"
    conda --version
fi

echo ""

# ============================================================================
# Activate Conda Environment
# ============================================================================
echo "Step 2: Setting up Conda environment..."
echo "----------------------------------------------------------------------"

# Initialize conda for bash if not already done
if [ ! -f ~/.bashrc ] || ! grep -q "conda initialize" ~/.bashrc; then
    echo "Initializing conda for bash..."
    eval "$(conda shell.bash hook)"
    conda init bash
    source ~/.bashrc
fi

# Activate environment
echo "Enter Conda environment name [NLPLResourceDownload]: "
read -r ENV_NAME
ENV_NAME=${ENV_NAME:-NLPLResourceDownload}

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✅ Environment '$ENV_NAME' found"
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
else
    echo "❌ Environment '$ENV_NAME' not found"
    echo ""
    echo "Creating new environment..."
    conda create -n "$ENV_NAME" python=3.10 -y
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
fi

echo "✅ Activated environment: $ENV_NAME"
echo ""

# ============================================================================
# Install Dependencies
# ============================================================================
echo "Step 3: Installing dependencies..."
echo "----------------------------------------------------------------------"

# Update pip
echo "Updating pip..."
pip install --upgrade pip --quiet

# Install OpenAI SDK
echo "Installing OpenAI SDK..."
pip install openai --quiet

echo "✅ Dependencies installed"
echo ""

# ============================================================================
# Verify Installation
# ============================================================================
echo "Step 4: Verifying installation..."
echo "----------------------------------------------------------------------"

python3 << 'PYTHON_CHECK'
import sys
try:
    import openai
    print(f"✅ OpenAI SDK version: {openai.__version__}")
    print(f"✅ Python version: {sys.version.split()[0]}")
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
PYTHON_CHECK

echo ""

# ============================================================================
# Check API Key
# ============================================================================
echo "Step 5: Checking API key..."
echo "----------------------------------------------------------------------"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set"
    echo ""
    echo "Would you like to set it now? (y/n)"
    read -r response
    
    if [ "$response" = "y" ]; then
        echo ""
        echo "Enter your OpenAI API key (starts with sk-proj-...):"
        read -r api_key
        
        # Add to .bashrc
        echo "" >> ~/.bashrc
        echo "# OpenAI API Key for Translation" >> ~/.bashrc
        echo "export OPENAI_API_KEY=\"$api_key\"" >> ~/.bashrc
        
        # Export for current session
        export OPENAI_API_KEY="$api_key"
        
        echo "✅ API key saved to ~/.bashrc"
        echo "   (Will be available in future sessions)"
    else
        echo ""
        echo "You can set it later with:"
        echo "  export OPENAI_API_KEY='your-key-here'"
        echo ""
        echo "Or add to ~/.bashrc:"
        echo "  echo 'export OPENAI_API_KEY=\"your-key-here\"' >> ~/.bashrc"
    fi
else
    echo "✅ OPENAI_API_KEY is set"
    # Show first 10 chars for verification
    echo "   Key starts with: ${OPENAI_API_KEY:0:10}..."
fi

echo ""

# ============================================================================
# Create Project Directory
# ============================================================================
echo "Step 6: Setting up project directory..."
echo "----------------------------------------------------------------------"

DEFAULT_PROJECT_DIR="$HOME/nlp_translation"
echo "Enter project directory [$DEFAULT_PROJECT_DIR]: "
read -r PROJECT_DIR
PROJECT_DIR=${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}

# Create directory
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "✅ Project directory: $PROJECT_DIR"
echo ""

# Create example corpus structure
echo "Creating example corpus structure..."
mkdir -p hindi_corpus/news
mkdir -p hindi_corpus/literature
mkdir -p hindi_corpus/social

# Create sample file
cat > hindi_corpus/news/sample.txt << 'EOF'
यह एक परीक्षण वाक्य है।
आज मौसम बहुत अच्छा है।
मुझे हिंदी भाषा पसंद है।
EOF

echo "✅ Example corpus created at: hindi_corpus/"
echo ""

# ============================================================================
# Check Script Location
# ============================================================================
echo "Step 7: Locating translation script..."
echo "----------------------------------------------------------------------"

SCRIPT_NAME="translate_domain_corpus.py"

# Check if script exists in current directory
if [ -f "$SCRIPT_NAME" ]; then
    echo "✅ Found script: $SCRIPT_NAME"
    chmod +x "$SCRIPT_NAME"
elif [ -f "../$SCRIPT_NAME" ]; then
    echo "✅ Found script: ../$SCRIPT_NAME"
    cp "../$SCRIPT_NAME" .
    chmod +x "$SCRIPT_NAME"
else
    echo "⚠️  Script not found in current directory"
    echo "   Make sure to copy translate_domain_corpus.py here"
    echo "   Location: $PROJECT_DIR/"
fi

echo ""

# ============================================================================
# Final Summary
# ============================================================================
echo "======================================================================"
echo "✅ SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  • Conda environment: $ENV_NAME"
echo "  • Python: $(python --version 2>&1 | cut -d' ' -f2)"
echo "  • OpenAI SDK: Installed ✅"
echo "  • API Key: $([ -z "$OPENAI_API_KEY" ] && echo "Not set ⚠️" || echo "Set ✅")"
echo "  • Project directory: $PROJECT_DIR"
echo "  • Example corpus: hindi_corpus/"
echo ""
echo "Next steps:"
echo "----------------------------------------------------------------------"
echo ""
echo "1. Prepare your Hindi corpus in domain-organized structure:"
echo "   hindi_corpus/"
echo "   ├── news/"
echo "   │   └── article1.txt"
echo "   ├── literature/"
echo "   │   └── story1.txt"
echo "   └── social/"
echo "       └── post1.txt"
echo ""
echo "2. Activate the environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo "3. Run the translation script:"
echo "   python translate_domain_corpus.py"
echo ""
echo "4. Or test with the example corpus:"
echo "   cd $PROJECT_DIR"
echo "   python translate_domain_corpus.py"
echo "   # When prompted, use: hindi_corpus"
echo ""
echo "======================================================================"
echo ""

# Save environment info
cat > setup_info.txt << EOF
Setup completed: $(date)
Conda environment: $ENV_NAME
Python version: $(python --version 2>&1)
OpenAI SDK version: $(python -c "import openai; print(openai.__version__)")
Project directory: $PROJECT_DIR
API Key set: $([ -z "$OPENAI_API_KEY" ] && echo "No" || echo "Yes")
EOF

echo "ℹ️  Setup information saved to: setup_info.txt"
echo ""
echo "Happy translating! 🚀"
