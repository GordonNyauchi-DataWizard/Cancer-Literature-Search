"""
Cancer Medical Literature Semantic Search System

This module implements a production-ready semantic search engine for cancer research papers.
It uses sentence transformers for embeddings and provides LLM-enhanced results.

Key Components:
- PDF text extraction and chunking
- Sentence transformer embeddings
- Cosine similarity search
- LLM-powered summarization and Q&A
"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle

# PDF Processing
from PyPDF2 import PdfReader

# Embeddings
from sentence_transformers import SentenceTransformer

# Progress bars
from tqdm import tqdm

# LLM Integration (using Anthropic Claude API)
import anthropic


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the semantic search system."""
    
    # Paths
    PDF_DIR = "papers"
    INDEX_DIR = "index"
    
    # Embedding Model
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking Parameters
    MAX_CHARS = 900  # Maximum characters per chunk
    OVERLAP = 120    # Overlap between chunks to preserve context
    
    # Search Parameters
    TOP_K = 10       # Number of top results to return
    BATCH_SIZE = 64  # Batch size for embedding generation
    
    # LLM Parameters
    LLM_MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 4096


# ============================================================================
# TEXT PROCESSING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text from PDFs.
    
    This function:
    1. Removes hyphenated line breaks (e.g., "exam-\\nple" -> "example")
    2. Replaces newlines with spaces
    3. Collapses multiple whitespaces into single spaces
    4. Strips leading/trailing whitespace
    
    Args:
        text: Raw text string from PDF extraction
        
    Returns:
        Cleaned text string
        
    Example:
        >>> clean_text("Hello-\\nWorld  \\n  Test")
        'HelloWorld Test'
    """
    # Remove hyphenated line breaks
    text = text.replace("-\n", "")
    
    # Replace newlines with spaces
    text = text.replace("\n", " ")
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def extract_pdf_chunks(
    pdf_path: str,
    max_chars: int = Config.MAX_CHARS,
    overlap: int = Config.OVERLAP
) -> List[Dict]:
    """
    Extract text from a PDF and split it into overlapping chunks.
    
    Why chunking?
    - LLMs and embedding models have token limits
    - Smaller chunks = more precise search results
    - Overlap ensures context isn't lost at chunk boundaries
    
    Process:
    1. Read PDF page by page
    2. Extract and clean text from each page
    3. Split text into chunks with specified overlap
    4. Create unique IDs for each chunk
    
    Args:
        pdf_path: Path to the PDF file
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of dictionaries containing:
        - chunk_id: Unique identifier (format: filename::page::chunk_number)
        - pdf_file: Original PDF filename
        - page: Page number (1-indexed)
        - text: Chunk text content
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If PDF cannot be read
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to read PDF {pdf_path}: {str(e)}")
    
    pdf_file = os.path.basename(pdf_path)
    chunks = []

    # Process each page
    for i, page in enumerate(reader.pages):
        # Extract raw text
        raw_text = page.extract_text() or ""
        text = clean_text(raw_text)
        
        # Skip empty pages
        if not text.strip():
            continue

        # Create overlapping chunks from page text
        start = 0
        page_num = i + 1
        
        while start < len(text):
            # Calculate chunk end position
            end = min(start + max_chars, len(text))
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Create unique chunk ID
                chunk_id = f"{pdf_file}::p{page_num}::c{len(chunks)}"
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "pdf_file": pdf_file,
                    "page": page_num,
                    "text": chunk_text
                })
            
            # Break if we've reached the end
            if end == len(text):
                break
            
            # Move start position with overlap
            start = max(0, end - overlap)

    return chunks


# ============================================================================
# EMBEDDING & INDEXING
# ============================================================================

class SemanticIndex:
    """
    Semantic search index using sentence transformers.
    
    This class handles:
    - Building embeddings for document chunks
    - Storing and loading the index
    - Performing similarity searches
    """
    
    def __init__(self, model_name: str = Config.MODEL_NAME):
        """
        Initialize the semantic index.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.chunks = []
        self.embeddings = None
        
    def build_index(self, pdf_dir: str) -> Tuple[int, int]:
        """
        Build the semantic search index from PDFs in a directory.
        
        Process:
        1. Find all PDF files
        2. Extract and chunk text from each PDF
        3. Generate embeddings for all chunks
        4. Normalize embeddings for efficient cosine similarity
        
        Args:
            pdf_dir: Directory containing PDF files
            
        Returns:
            Tuple of (number of PDFs processed, number of chunks created)
            
        Raises:
            FileNotFoundError: If no PDFs found in directory
            ValueError: If no text could be extracted
        """
        # Find all PDF files
        pdf_files = sorted([
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.lower().endswith(".pdf")
        ])

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

        print("\n📚 Found PDFs:")
        for f in pdf_files:
            print(f"   - {os.path.basename(f)}")

        # Extract chunks from all PDFs
        print("\n📄 Extracting text chunks...")
        all_chunks = []
        for pdf in tqdm(pdf_files, desc="Processing PDFs"):
            chunks = extract_pdf_chunks(pdf, Config.MAX_CHARS, Config.OVERLAP)
            all_chunks.extend(chunks)

        if len(all_chunks) == 0:
            raise ValueError(
                "No text extracted. If these are scanned PDFs, you may need OCR."
            )

        print(f"\n✓ Total chunks extracted: {len(all_chunks)}")

        # Generate embeddings
        print(f"\n🔢 Generating embeddings using {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        texts = [chunk["text"] for chunk in all_chunks]
        
        # Process in batches for efficiency
        embeddings_list = []
        for i in tqdm(
            range(0, len(texts), Config.BATCH_SIZE),
            desc="Embedding chunks"
        ):
            batch = texts[i:i + Config.BATCH_SIZE]
            # Normalize embeddings so cosine similarity = dot product
            emb = self.model.encode(batch, normalize_embeddings=True)
            embeddings_list.append(emb)

        # Stack all embeddings into single array
        self.embeddings = np.vstack(embeddings_list).astype(np.float32)
        self.chunks = all_chunks
        
        print(f"✓ Embeddings shape: {self.embeddings.shape}")
        
        return len(pdf_files), len(all_chunks)
    
    def search(self, query: str, top_k: int = Config.TOP_K) -> List[Dict]:
        """
        Search the index for chunks similar to the query.
        
        Process:
        1. Encode the query using the same model
        2. Compute cosine similarity with all chunks
        3. Return top K most similar chunks
        
        Cosine Similarity:
        - Measures angle between vectors (0 to 1)
        - 1 = identical direction (most similar)
        - 0 = perpendicular (unrelated)
        - Since embeddings are normalized, dot product = cosine similarity
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing:
            - All chunk metadata (chunk_id, pdf_file, page, text)
            - similarity: Cosine similarity score (0-1)
            
        Raises:
            ValueError: If index hasn't been built yet
        """
        if self.embeddings is None or self.model is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]
        
        # Compute cosine similarities (dot product since normalized)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            result = self.chunks[idx].copy()
            result["similarity"] = float(similarities[idx])
            results.append(result)
        
        return results
    
    def save(self, save_dir: str):
        """
        Save the index to disk for later use.
        
        Saves:
        - chunks.pkl: Document chunks metadata
        - embeddings.npy: Embedding vectors
        - config.json: Model configuration
        
        Args:
            save_dir: Directory to save index files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save chunks
        with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        
        # Save embeddings
        np.save(os.path.join(save_dir, "embeddings.npy"), self.embeddings)
        
        # Save config
        config = {"model_name": self.model_name}
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f)
        
        print(f"\n✓ Index saved to {save_dir}")
    
    def load(self, load_dir: str):
        """
        Load a previously saved index from disk.
        
        Args:
            load_dir: Directory containing saved index files
            
        Raises:
            FileNotFoundError: If index files not found
        """
        # Load chunks
        with open(os.path.join(load_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        
        # Load embeddings
        self.embeddings = np.load(os.path.join(load_dir, "embeddings.npy"))
        
        # Load config and model
        with open(os.path.join(load_dir, "config.json"), "r") as f:
            config = json.load(f)
            self.model_name = config["model_name"]
        
        self.model = SentenceTransformer(self.model_name)
        
        print(f"\n✓ Index loaded from {load_dir}")
        print(f"  - Chunks: {len(self.chunks)}")
        print(f"  - Embeddings: {self.embeddings.shape}")


# ============================================================================
# LLM INTEGRATION
# ============================================================================

class LLMEnhancer:
    """
    LLM-powered enhancement for search results.
    
    Capabilities:
    - Summarize top search results
    - Answer questions based on retrieved context
    - Perform comparative analysis
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM enhancer with Anthropic Claude.
        
        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            print("\n⚠️  Warning: No Anthropic API key found.")
            print("   Set ANTHROPIC_API_KEY environment variable for LLM features.")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def summarize_results(self, results: List[Dict]) -> str:
        """
        Generate a concise summary of search results.
        
        This helps users quickly understand what information was found
        without reading through all chunks.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Summary text
        """
        if not self.client:
            return "LLM summarization unavailable. Set ANTHROPIC_API_KEY."
        
        # Prepare context from results
        context = self._format_results_for_llm(results)
        
        prompt = f"""You are analyzing search results from cancer medical literature.

Here are the top search results:

{context}

Please provide a concise summary (3-5 sentences) of the main findings across these results. Focus on:
1. Key themes or topics
2. Important findings or conclusions
3. Any notable patterns or connections

Summary:"""
        
        try:
            message = self.client.messages.create(
                model=Config.LLM_MODEL,
                max_tokens=Config.MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def answer_question(self, query: str, results: List[Dict]) -> str:
        """
        Answer a question based on retrieved search results.
        
        This performs retrieval-augmented generation (RAG):
        1. Semantic search retrieves relevant chunks
        2. LLM uses those chunks to answer the question
        
        Args:
            query: User's question
            results: Search results providing context
            
        Returns:
            Answer text with citations
        """
        if not self.client:
            return "LLM Q&A unavailable. Set ANTHROPIC_API_KEY."
        
        context = self._format_results_for_llm(results)
        
        prompt = f"""You are a medical research assistant specializing in cancer literature.

Question: {query}

Here is relevant context from research papers:

{context}

Please answer the question based on the provided context. Include:
1. A clear, direct answer
2. Key supporting evidence from the papers
3. Citations (mention the paper name and page when referencing specific information)
4. Any important caveats or limitations

If the context doesn't contain enough information to answer fully, say so.

Answer:"""
        
        try:
            message = self.client.messages.create(
                model=Config.LLM_MODEL,
                max_tokens=Config.MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def compare_results(self, results: List[Dict]) -> str:
        """
        Perform comparative analysis across search results.
        
        Useful for understanding:
        - Different perspectives or approaches
        - Contradictory findings
        - Evolution of understanding over time
        
        Args:
            results: Search results to compare
            
        Returns:
            Comparative analysis text
        """
        if not self.client:
            return "LLM comparison unavailable. Set ANTHROPIC_API_KEY."
        
        context = self._format_results_for_llm(results)
        
        prompt = f"""You are analyzing multiple excerpts from cancer research papers.

Here are the excerpts:

{context}

Please provide a comparative analysis that identifies:
1. Common themes or agreements across sources
2. Differences in approaches, findings, or perspectives
3. Any contradictions or debates
4. How different papers complement each other

Analysis:"""
        
        try:
            message = self.client.messages.create(
                model=Config.LLM_MODEL,
                max_tokens=Config.MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error generating comparison: {str(e)}"
    
    def _format_results_for_llm(self, results: List[Dict]) -> str:
        """
        Format search results into readable context for LLM.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted context string
        """
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"[Result {i}] (Similarity: {result['similarity']:.3f})\n"
                f"Source: {result['pdf_file']}, Page {result['page']}\n"
                f"Text: {result['text']}\n"
            )
        return "\n".join(formatted)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class CancerSearchApp:
    """
    Main application class that ties everything together.
    
    This provides a simple interface for:
    - Building/loading the search index
    - Performing searches
    - Getting LLM-enhanced results
    """
    
    def __init__(self):
        """Initialize the application."""
        self.index = SemanticIndex()
        self.llm = LLMEnhancer()
        self.is_indexed = False
    
    def build_or_load_index(self, pdf_dir: str = Config.PDF_DIR, force_rebuild: bool = False):
        """
        Build a new index or load existing one.
        
        Args:
            pdf_dir: Directory containing PDF files
            force_rebuild: If True, rebuild even if saved index exists
        """
        index_path = Config.INDEX_DIR
        
        # Try to load existing index
        if not force_rebuild and os.path.exists(index_path):
            try:
                print("\n🔍 Found existing index, loading...")
                self.index.load(index_path)
                self.is_indexed = True
                return
            except Exception as e:
                print(f"\n⚠️  Failed to load index: {e}")
                print("Building new index instead...")
        
        # Build new index
        print("\n🏗️  Building new search index...")
        num_pdfs, num_chunks = self.index.build_index(pdf_dir)
        
        # Save for future use
        self.index.save(index_path)
        self.is_indexed = True
        
        print(f"\n✅ Index built successfully!")
        print(f"   - PDFs processed: {num_pdfs}")
        print(f"   - Total chunks: {num_chunks}")
    
    def search(
        self,
        query: str,
        top_k: int = Config.TOP_K,
        enhance: bool = True
    ) -> Dict:
        """
        Perform a semantic search with optional LLM enhancement.
        
        Args:
            query: Search query
            top_k: Number of results to return
            enhance: Whether to include LLM-generated summary
            
        Returns:
            Dictionary containing:
            - query: Original query
            - results: List of search results
            - summary: LLM-generated summary (if enhance=True)
            
        Raises:
            ValueError: If index hasn't been built
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_or_load_index() first.")
        
        # Perform search
        results = self.index.search(query, top_k)
        
        response = {
            "query": query,
            "results": results
        }
        
        # Add LLM enhancement if requested
        if enhance and self.llm.client:
            response["summary"] = self.llm.summarize_results(results)
        
        return response
    
    def answer_question(self, question: str, top_k: int = Config.TOP_K) -> str:
        """
        Answer a question using RAG (Retrieval-Augmented Generation).
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve for context
            
        Returns:
            LLM-generated answer with citations
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_or_load_index() first.")
        
        # Retrieve relevant context
        results = self.index.search(question, top_k)
        
        # Generate answer
        answer = self.llm.answer_question(question, results)
        
        return answer


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_results(response: Dict):
    """
    Pretty print search results.
    
    Args:
        response: Response dictionary from search()
    """
    print("\n" + "="*80)
    print(f"QUERY: {response['query']}")
    print("="*80)
    
    if "summary" in response:
        print("\n📝 SUMMARY:")
        print(response["summary"])
        print("\n" + "-"*80)
    
    print(f"\n📊 TOP {len(response['results'])} RESULTS:\n")
    
    for i, result in enumerate(response['results'], 1):
        print(f"\n[{i}] Similarity: {result['similarity']:.4f}")
        print(f"    Source: {result['pdf_file']}")
        print(f"    Page: {result['page']}")
        print(f"    Text: {result['text'][:200]}...")
        print()


if __name__ == "__main__":
    """
    Example usage demonstrating the main features.
    """
    print("🔬 Cancer Medical Literature Semantic Search System")
    print("="*80)
    
    # Initialize app
    app = CancerSearchApp()
    
    # Build or load index
    try:
        app.build_or_load_index()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease create a 'papers' directory and add PDF files.")
        exit(1)
    
    # Example searches
    print("\n" + "="*80)
    print("EXAMPLE SEARCHES")
    print("="*80)
    
    # Example 1: Basic search
    print("\n1️⃣  Basic Semantic Search")
    response = app.search("immunotherapy treatment options", top_k=5)
    print_results(response)
    
    # Example 2: Question answering
    print("\n2️⃣  Question Answering (RAG)")
    question = "What are the main side effects of immunotherapy?"
    answer = app.answer_question(question, top_k=5)
    print(f"\nQ: {question}")
    print(f"\nA: {answer}")
    
    print("\n" + "="*80)
    print("✅ Demo complete! Check cli.py for interactive usage.")
    print("="*80)
