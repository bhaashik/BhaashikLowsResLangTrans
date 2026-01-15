#!/bin/bash
#
# QuickStart Script for BhaashikLowsResLangTrans
# This script helps you get started quickly with the translation system
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}BhaashikLowsResLangTrans QuickStart${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ Conda not found${NC}"
    echo "Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓ Conda found${NC}"

# Check if environment exists
if conda env list | grep -q "NLPLResourceDownload"; then
    echo -e "${YELLOW}Environment 'NLPLResourceDownload' already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " recreate
    if [[ $recreate == "y" || $recreate == "Y" ]]; then
        echo "Removing existing environment..."
        conda env remove -n NLPLResourceDownload -y
    fi
fi

# Create environment
if ! conda env list | grep -q "NLPLResourceDownload"; then
    echo -e "${BLUE}Creating conda environment...${NC}"
    conda env create -f environment.yml
    echo -e "${GREEN}✓ Environment created${NC}"
fi

# Activate environment
echo -e "${BLUE}Activating environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate NLPLResourceDownload

# Install pip packages
echo -e "${BLUE}Installing Python packages...${NC}"
pip install -r requirements.txt

# Setup .env file
if [ ! -f .env ]; then
    echo -e "${BLUE}Setting up .env file...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠ Please edit .env file with your configuration${NC}"
    echo -e "   Especially set BASE_DIR to your preferred location"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Verify setup
echo -e "${BLUE}Running verification...${NC}"
python scripts/verify.py --environment

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}QuickStart Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit .env file:"
echo "   nano .env"
echo ""
echo "2. Download essential resources (FREE, ~70GB):"
echo "   python scripts/download.py --essential"
echo ""
echo "3. Try a simple translation:"
echo "   python scripts/translate.py --src en --tgt hi --text \"Hello!\" --indictrans2"
echo ""
echo "4. Run examples:"
echo "   python examples/example_1_basic_translation.py"
echo ""
echo "5. Read documentation:"
echo "   - SETUP.md - Detailed setup guide"
echo "   - USAGE.md - Usage examples"
echo "   - CLAUDE.md - Architecture guide"
echo ""
echo -e "${BLUE}========================================${NC}"
