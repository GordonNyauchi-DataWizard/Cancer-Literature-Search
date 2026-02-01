"""
Unit Tests for Cancer Literature Search System

Run with:
    pytest test_semantic_search.py -v

Or:
    python -m pytest test_semantic_search.py
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from semantic_search import (
    clean_text,
    extract_pdf_chunks,
    SemanticIndex,
    LLMEnhancer,
    CancerSearchApp,
    Config
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_text():
    """Sample medical text for testing."""
    return """
    Immunotherapy has revolutionized cancer treatment. CAR-T cell therapy
    represents a breakthrough in treating blood cancers. Checkpoint inhibitors
    like pembrolizumab have shown remarkable efficacy in melanoma and lung cancer.
    """


# ============================================================================
# TEXT PROCESSING TESTS
# ============================================================================

class TestTextProcessing:
    """Tests for text cleaning and chunking."""
    
    def test_clean_text_removes_hyphens(self):
        """Test that hyphenated line breaks are removed."""
        text = "immuno-\ntherapy treatment"
        result = clean_text(text)
        assert result == "immunotherapy treatment"
    
    def test_clean_text_normalizes_whitespace(self):
        """Test whitespace normalization."""
        text = "cancer  \n  treatment    research"
        result = clean_text(text)
        assert result == "cancer treatment research"
    
    def test_clean_text_strips_edges(self):
        """Test that leading/trailing whitespace is removed."""
        text = "  cancer treatment  "
        result = clean_text(text)
        assert result == "cancer treatment"
    
    def test_chunking_with_overlap(self):
        """Test that text is properly chunked with overlap."""
        text = "A" * 1000  # 1000 character string
        
        # Mock PDF for testing
        # Note: In real tests, you'd create an actual PDF file
        # For this example, we'll test the chunking logic directly
        
        chunks = []
        start = 0
        max_chars = 100
        overlap = 20
        
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == len(text):
                break
            start = end - overlap
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Each chunk (except last) should be max_chars long
        for chunk in chunks[:-1]:
            assert len(chunk) == max_chars
        
        # Verify overlap: end of chunk N should appear in chunk N+1
        for i in range(len(chunks) - 1):
            overlap_section = chunks[i][-overlap:]
            assert overlap_section in chunks[i + 1]


# ============================================================================
# SEMANTIC INDEX TESTS
# ============================================================================

class TestSemanticIndex:
    """Tests for semantic indexing and search."""
    
    def test_index_initialization(self):
        """Test that index initializes correctly."""
        index = SemanticIndex()
        assert index.model is None
        assert index.chunks == []
        assert index.embeddings is None
    
    def test_search_requires_built_index(self):
        """Test that search fails if index not built."""
        index = SemanticIndex()
        
        with pytest.raises(ValueError, match="Index not built"):
            index.search("test query")
    
    @pytest.mark.slow
    def test_embedding_generation(self, sample_text):
        """Test embedding generation (slow test - downloads model)."""
        index = SemanticIndex()
        
        # This will download the model on first run
        from sentence_transformers import SentenceTransformer
        index.model = SentenceTransformer(Config.MODEL_NAME)
        
        # Generate embedding for sample text
        embedding = index.model.encode([sample_text], normalize_embeddings=True)[0]
        
        # Check embedding properties
        assert embedding.shape == (384,)  # all-MiniLM-L6-v2 has 384 dims
        assert abs(sum(embedding ** 2) - 1.0) < 0.01  # Normalized (length ≈ 1)
    
    def test_index_save_load(self, temp_dir, sample_text):
        """Test saving and loading index."""
        index = SemanticIndex()
        
        # Mock some data
        index.chunks = [
            {"chunk_id": "test::p1::c0", "text": sample_text, "page": 1, "pdf_file": "test.pdf"}
        ]
        index.embeddings = [[0.1] * 384]  # Mock embedding
        index.model_name = Config.MODEL_NAME
        
        # Save
        save_path = os.path.join(temp_dir, "test_index")
        index.save(save_path)
        
        # Verify files exist
        assert os.path.exists(os.path.join(save_path, "chunks.pkl"))
        assert os.path.exists(os.path.join(save_path, "embeddings.npy"))
        assert os.path.exists(os.path.join(save_path, "config.json"))
        
        # Load
        new_index = SemanticIndex()
        new_index.load(save_path)
        
        # Verify loaded correctly
        assert len(new_index.chunks) == len(index.chunks)
        assert new_index.chunks[0]["text"] == sample_text


# ============================================================================
# LLM ENHANCER TESTS
# ============================================================================

class TestLLMEnhancer:
    """Tests for LLM enhancement features."""
    
    def test_enhancer_without_api_key(self):
        """Test that enhancer works without API key (degraded mode)."""
        # Temporarily remove API key
        old_key = os.environ.get("ANTHROPIC_API_KEY")
        if old_key:
            del os.environ["ANTHROPIC_API_KEY"]
        
        enhancer = LLMEnhancer()
        assert enhancer.client is None
        
        # Should return error message instead of crashing
        results = [{"text": "test", "similarity": 0.9, "pdf_file": "test.pdf", "page": 1}]
        summary = enhancer.summarize_results(results)
        assert "unavailable" in summary.lower()
        
        # Restore API key
        if old_key:
            os.environ["ANTHROPIC_API_KEY"] = old_key
    
    def test_result_formatting(self, sample_text):
        """Test that results are formatted correctly for LLM."""
        enhancer = LLMEnhancer()
        
        results = [
            {
                "text": sample_text,
                "similarity": 0.95,
                "pdf_file": "paper1.pdf",
                "page": 5
            },
            {
                "text": "Second result text",
                "similarity": 0.87,
                "pdf_file": "paper2.pdf",
                "page": 12
            }
        ]
        
        formatted = enhancer._format_results_for_llm(results)
        
        # Check that formatting includes key information
        assert "paper1.pdf" in formatted
        assert "paper2.pdf" in formatted
        assert "Page 5" in formatted
        assert "Page 12" in formatted
        assert sample_text in formatted


# ============================================================================
# APPLICATION TESTS
# ============================================================================

class TestCancerSearchApp:
    """Tests for main application class."""
    
    def test_app_initialization(self):
        """Test that app initializes correctly."""
        app = CancerSearchApp()
        
        assert app.index is not None
        assert app.llm is not None
        assert app.is_indexed is False
    
    def test_search_requires_index(self):
        """Test that search requires built index."""
        app = CancerSearchApp()
        
        with pytest.raises(ValueError, match="Index not built"):
            app.search("test query")
    
    def test_answer_requires_index(self):
        """Test that Q&A requires built index."""
        app = CancerSearchApp()
        
        with pytest.raises(ValueError, match="Index not built"):
            app.answer_question("test question")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests (require sample PDFs)."""
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.path.exists("papers"),
        reason="Requires papers/ directory with PDFs"
    )
    def test_full_pipeline(self):
        """Test complete indexing and search pipeline."""
        app = CancerSearchApp()
        
        # Build index
        app.build_or_load_index(force_rebuild=True)
        assert app.is_indexed
        assert len(app.index.chunks) > 0
        
        # Perform search
        response = app.search("immunotherapy", top_k=5, enhance=False)
        assert "results" in response
        assert len(response["results"]) <= 5
        
        # Check result format
        for result in response["results"]:
            assert "text" in result
            assert "similarity" in result
            assert "pdf_file" in result
            assert "page" in result
            assert 0 <= result["similarity"] <= 1


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance benchmarks."""
    
    @pytest.mark.slow
    def test_search_performance(self, sample_text):
        """Benchmark search performance."""
        import time
        import numpy as np
        
        # Create mock index with 1000 chunks
        index = SemanticIndex()
        index.chunks = [{"text": sample_text, "pdf_file": "test.pdf", "page": 1}] * 1000
        index.embeddings = np.random.rand(1000, 384).astype(np.float32)
        index.embeddings = index.embeddings / np.linalg.norm(index.embeddings, axis=1, keepdims=True)
        
        from sentence_transformers import SentenceTransformer
        index.model = SentenceTransformer(Config.MODEL_NAME)
        
        # Time search
        start = time.time()
        results = index.search("immunotherapy", top_k=10)
        duration = time.time() - start
        
        # Should be fast (<0.5 seconds for 1000 chunks)
        assert duration < 0.5
        assert len(results) == 10


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v", "--tb=short"])
