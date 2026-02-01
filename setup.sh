#!/bin/bash
# Setup script for Cancer Literature Search System

echo "🔬 Cancer Literature Search - Setup Script"
echo "==========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi
echo "✓ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Remove it? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        rm -rf venv
        python3 -m venv venv
        echo "✓ New virtual environment created"
    else
        echo "✓ Using existing virtual environment"
    fi
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "(This may take 5-10 minutes on first run)"
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p papers
mkdir -p index
echo "✓ Directories created"
echo ""

# Check for API key
echo "Checking for Anthropic API key..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY not found in environment"
    echo ""
    echo "To enable LLM features (summaries, Q&A), you need an API key."
    echo "Get one from: https://console.anthropic.com/"
    echo ""
    echo "Set it with:"
    echo "  export ANTHROPIC_API_KEY='your-key-here'"
    echo ""
    echo "Or create a .env file:"
    echo "  echo 'ANTHROPIC_API_KEY=your-key-here' > .env"
    echo ""
else
    echo "✓ API key found"
fi
echo ""

# Download sample data (optional)
echo "Would you like to download sample cancer research papers? (y/n)"
echo "(Note: This will download ~50MB of PDFs from PubMed Central)"
read -r download_sample

if [ "$download_sample" = "y" ]; then
    echo "Downloading sample papers..."
    python3 << 'EOF'
import urllib.request
import os

# Sample open-access cancer papers from PMC
papers = [
    ("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6000873/pdf/main.pdf", "immunotherapy_review.pdf"),
    ("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5540396/pdf/oncotarget-08-47221.pdf", "car_t_therapy.pdf"),
    # Add more paper URLs here
]

os.makedirs("papers", exist_ok=True)

for url, filename in papers:
    filepath = os.path.join("papers", filename)
    if not os.path.exists(filepath):
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"✓ {filename} downloaded")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
    else:
        print(f"✓ {filename} already exists")
EOF
    echo ""
fi

# Run tests
echo "Would you like to run tests? (y/n)"
read -r run_tests

if [ "$run_tests" = "y" ]; then
    echo "Running tests..."
    pytest test_semantic_search.py -v --tb=short
    echo ""
fi

# Setup complete
echo "==========================================="
echo "✅ Setup Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Add PDF files to the 'papers/' directory"
echo "   - Need at least 100 cancer research papers"
echo "   - Get them from PubMed, Google Scholar, etc."
echo ""
echo "2. Build the search index:"
echo "   python cli.py --rebuild"
echo ""
echo "3. Start searching:"
echo "   python cli.py                    # Interactive CLI"
echo "   streamlit run app.py             # Web interface"
echo ""
echo "For help:"
echo "   python cli.py --help"
echo ""
echo "Documentation:"
echo "   README.md         - User guide"
echo "   ARCHITECTURE.md   - Technical details"
echo ""
echo "Enjoy exploring cancer research! 🔬"
