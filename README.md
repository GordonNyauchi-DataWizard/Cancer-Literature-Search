# 🔬 Cancer Medical Literature Semantic Search 

A production-ready semantic search application for exploring cancer research papers using AI-powered embeddings and large language models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Demo Video]
- [License](#license)

## 🎯 Overview

This system provides intelligent semantic search over cancer medical literature using state-of-the-art natural language processing techniques. It combines:

- **Sentence Transformers** for generating semantic embeddings
- **Cosine Similarity** for efficient document retrieval
- **Claude AI (Anthropic)** for intelligent summarization and question answering
- **Retrieval-Augmented Generation (RAG)** for accurate, citation-backed answers

### Why This Project?

Medical literature grows exponentially, making it challenging for researchers and clinicians to stay current. This tool:

1. **Enables semantic search** - Find relevant papers by meaning, not just keywords
2. **Provides AI summaries** - Quickly understand key findings across multiple papers
3. **Answers questions** - Get specific answers with citations from your corpus
4. **Compares findings** - Identify agreements, contradictions, and research gaps

## ✨ Features

### Core Capabilities

- ✅ **Semantic Search**: Find relevant passages using natural language queries
- ✅ **Question Answering**: Ask questions and get AI-generated answers with citations
- ✅ **Comparative Analysis**: Compare findings across multiple research papers
- ✅ **LLM Summarization**: Automatically summarize search results
- ✅ **Efficient Indexing**: Fast similarity search using normalized embeddings
- ✅ **Persistent Storage**: Save and reload indexes for quick startup

### Interfaces

- 🖥️ **Command-Line Interface**: Interactive CLI for power users
- 🌐 **Web Interface**: User-friendly Streamlit app (deployable on cloud platforms)
- 📚 **Python API**: Use as a library in your own applications

### Technical Features

- 📄 **PDF Processing**: Automatic text extraction with overlap chunking
- 🔢 **Batch Processing**: Efficient embedding generation
- 💾 **Index Persistence**: Save/load indexes to avoid reprocessing
- 🎯 **Top-K Retrieval**: Return most relevant results
- 🔧 **Configurable**: Easy parameter tuning
- ⚡ **Production-Ready**: Error handling, logging, documentation

## 🏗️ Architecture

The system consists of four main components:

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     CLI      │  │   Streamlit  │  │  Python API  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                          │
│  ┌──────────────────────────────────────────────────┐       │
│  │           CancerSearchApp                         │       │
│  │  • Orchestrates search and LLM operations        │       │
│  │  • Manages index lifecycle                       │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Processing Layer                           │
│  ┌──────────────────┐           ┌──────────────────┐        │
│  │ SemanticIndex    │           │   LLMEnhancer    │        │
│  │ • Embeddings     │           │ • Summarization  │        │
│  │ • Search         │           │ • Q&A            │        │
│  │ • Persistence    │           │ • Comparison     │        │
│  └──────────────────┘           └──────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     PDFs     │  │  Embeddings  │  │   Metadata   │      │
│  │   (papers/)  │  │  (numpy)     │  │   (pickle)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Anthropic API key for LLM features

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/cancer-literature-search.git
cd cancer-literature-search
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First-time installation may take 5-10 minutes as it downloads the sentence transformer model (~90MB).

### Step 4: Set Up API Key (Optional)

For LLM-powered features (summaries, Q&A):

```bash
# Linux/Mac:
export ANTHROPIC_API_KEY='your-api-key-here'

# Windows:
set ANTHROPIC_API_KEY=your-api-key-here
```

Or create a `.env` file:

```bash
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```

Get your API key from: https://console.anthropic.com/

## 🎯 Quick Start

### 1. Add Your PDF Files

Create a `papers/` directory and add at least 100 cancer research papers:

```bash
mkdir papers
# Add your PDF files to this directory
```

**Where to find papers?**
- [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/) - Free full-text articles
- [Google Scholar](https://scholar.google.com/) - Search and download
- [bioRxiv](https://www.biorxiv.org/) - Preprints
- University library databases

### 2. Build the Search Index

```bash
python cli.py --rebuild
```

This will:
- Extract text from all PDFs
- Split into chunks with overlap
- Generate embeddings
- Save index for future use

**Expected time**: ~1-2 minutes per 100 papers

### 3. Start Searching!

#### Option A: Interactive CLI

```bash
python cli.py
```

Then use commands like:
```
🔍 > search immunotherapy side effects
🔍 > ask What is CAR-T cell therapy?
🔍 > compare checkpoint inhibitors vs chemotherapy
```

#### Option B: Web Interface

```bash
streamlit run app.py
```

Opens a web browser with a user-friendly interface!

#### Option C: Single Query

```bash
# Quick search
python cli.py --query "BRCA1 mutations"

# Quick question
python cli.py --ask "What are the main types of immunotherapy?"
```

## 📖 Usage

### Command-Line Interface

#### Interactive Mode

```bash
python cli.py
```

Available commands:
- `search <query>` - Semantic search
- `ask <question>` - Question answering
- `compare <query>` - Comparative analysis
- `help` - Show help
- `quit` - Exit

#### Single Query Mode

```bash
# Search
python cli.py --query "tumor microenvironment"

# Question answering
python cli.py --ask "How does immunotherapy work?"

# Comparative analysis
python cli.py --compare "different CAR-T approaches"

# Custom number of results
python cli.py --query "KRAS mutations" --top-k 15
```

#### Rebuild Index

```bash
python cli.py --rebuild
```

### Web Interface (Streamlit)

```bash
streamlit run app.py
```

Features:
- 🔍 **Semantic Search**: Natural language search
- 💬 **Question Answering**: Ask questions, get answers with citations
- 📊 **Comparative Analysis**: Compare findings across papers
- ⚙️ **Settings**: Adjust number of results, enable/disable AI summaries
- 💡 **Examples**: Pre-loaded example queries

The web app automatically loads at `http://localhost:8501`

### Python API

Use as a library in your own code:

```python
from semantic_search import CancerSearchApp

# Initialize
app = CancerSearchApp()
app.build_or_load_index()

# Search
results = app.search("immunotherapy", top_k=10, enhance=True)
print(results['summary'])

for result in results['results']:
    print(f"{result['pdf_file']}: {result['text'][:100]}...")

# Question answering
answer = app.answer_question("What are common side effects of chemotherapy?")
print(answer)
```

## 📁 Project Structure

```
cancer-literature-search/
├── semantic_search.py      # Core search engine implementation
├── cli.py                  # Command-line interface
├── app.py                  # Streamlit web interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── ARCHITECTURE.md        # Technical architecture details
├── papers/                # PDF files (you add these)
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
└── index/                 # Generated search index (auto-created)
    ├── chunks.pkl         # Document chunks
    ├── embeddings.npy     # Embedding vectors
    └── config.json        # Index configuration
```

## ⚙️ Configuration

Edit `Config` class in `semantic_search.py`:

```python
class Config:
    # Directories
    PDF_DIR = "papers"           # Where PDFs are stored
    INDEX_DIR = "index"          # Where index is saved
    
    # Embedding Model
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking
    MAX_CHARS = 900             # Characters per chunk
    OVERLAP = 120               # Overlap between chunks
    
    # Search
    TOP_K = 10                  # Number of results
    BATCH_SIZE = 64             # Embedding batch size
    
    # LLM
    LLM_MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 4096
```

### Alternative Embedding Models

You can use different sentence transformer models:

```python
# Faster, smaller model (current default)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 90MB

# More accurate, larger model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 420MB

# Specialized for scientific text
MODEL_NAME = "allenai/specter"  # 440MB
```

See all models: https://www.sbert.net/docs/pretrained_models.html

## 💡 Examples

### Example 1: Finding Treatment Information

```bash
python cli.py
🔍 > search targeted therapy for lung cancer
```

Returns relevant passages about targeted therapies with similarity scores.

### Example 2: Getting Specific Answers

```bash
python cli.py
🔍 > ask What are the different types of immunotherapy?
```

Returns:
```
📝 Answer:
According to the research papers, there are several main types of immunotherapy:

1. Checkpoint Inhibitors: These drugs block proteins that prevent immune cells...
   (Source: immunotherapy_review.pdf, Page 5)

2. CAR-T Cell Therapy: Modified T cells that target specific cancer proteins...
   (Source: car_t_mechanisms.pdf, Page 12)

[Full detailed answer with citations...]
```

### Example 3: Comparative Analysis

```bash
python cli.py
🔍 > compare surgery vs radiation for early-stage cancer
```

Returns analysis comparing approaches, outcomes, and contexts.

## 📚 API Reference

### SemanticIndex

```python
index = SemanticIndex(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build index from PDFs
index.build_index(pdf_dir="papers")

# Search
results = index.search(query="immunotherapy", top_k=10)

# Save/load
index.save("index")
index.load("index")
```

### LLMEnhancer

```python
llm = LLMEnhancer(api_key="your-key")

# Summarize results
summary = llm.summarize_results(results)

# Answer question
answer = llm.answer_question(question, results)

# Compare results
comparison = llm.compare_results(results)
```

### CancerSearchApp

```python
app = CancerSearchApp()

# Build/load index
app.build_or_load_index(pdf_dir="papers", force_rebuild=False)

# Search with enhancement
response = app.search(query, top_k=10, enhance=True)

# Question answering
answer = app.answer_question(question, top_k=10)
```

## 🔧 Troubleshooting

### Issue: "No PDF files found"

**Solution**: Ensure PDFs are in the `papers/` directory:
```bash
ls papers/  # Should list your PDF files
```

### Issue: "No text extracted. If these are scanned PDFs, you may need OCR"

**Solution**: Your PDFs are scanned images. Use OCR:
```bash
pip install pytesseract
# Install tesseract system package
# Then modify extract_pdf_chunks() to use OCR
```

### Issue: "LLM features unavailable"

**Solution**: Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY='your-key-here'
```

### Issue: Out of memory during indexing

**Solution**: Reduce batch size in `Config`:
```python
BATCH_SIZE = 32  # or 16
```

### Issue: Slow search performance

**Solution**: 
1. Reduce `TOP_K` value
2. Use a smaller embedding model
3. Reduce number of chunks (increase `MAX_CHARS`)

### Issue: Poor search results

**Solutions**:
1. Use more specific queries
2. Increase `TOP_K` to see more results
3. Try a different embedding model (e.g., all-mpnet-base-v2)
4. Adjust `MAX_CHARS` for better chunk granularity

## 🚀 Deployment

### Deploy Web App to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add `ANTHROPIC_API_KEY` to secrets
5. Deploy!

### Deploy to Hugging Face Spaces

1. Create account on [huggingface.co](https://huggingface.co)
2. Create new Space with Streamlit
3. Upload code and requirements.txt
4. Add API key to Space secrets
5. Your app is live!

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more embedding model options
- [ ] Support for more document types (Word, HTML)
- [ ] Advanced filtering (by date, journal, etc.)
- [ ] Visualization of search results
- [ ] Export results to various formats
- [ ] Multi-language support
- [ ] Fine-tuned models for medical text

## 📹 Demo Video
   https://vimeo.com/manage/videos/1160874062

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Sentence Transformers by UKPLab
- Anthropic for Claude AI
- Open-source medical research community

## 📞 Contact

For questions or issues:
- Open an issue on GitHub
- Email: [gnyauchi@uci.edu]
- Twitter: [@nnyauchi]




