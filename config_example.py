# Configuration File for Cancer Literature Search System
# Copy this to config.py and customize as needed

import os

class Config:
    """
    Configuration parameters for the semantic search system.
    
    Customize these values based on your needs:
    - Larger MAX_CHARS = fewer, longer chunks (better for context)
    - Smaller MAX_CHARS = more, shorter chunks (better for precision)
    - Larger OVERLAP = more redundancy (better recall, more storage)
    - Larger TOP_K = more results (slower LLM processing)
    """
    
    # ========================================================================
    # PATHS
    # ========================================================================
    
    # Directory containing PDF files
    PDF_DIR = os.getenv("PDF_DIR", "papers")
    
    # Directory for storing index
    INDEX_DIR = os.getenv("INDEX_DIR", "index")
    
    # ========================================================================
    # EMBEDDING MODEL
    # ========================================================================
    
    # Sentence transformer model to use for embeddings
    # Options:
    #   - "sentence-transformers/all-MiniLM-L6-v2"  (fast, 90MB)
    #   - "sentence-transformers/all-mpnet-base-v2" (accurate, 420MB)
    #   - "allenai/specter"                          (scientific, 440MB)
    #   - "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb" (medical)
    
    MODEL_NAME = os.getenv(
        "MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # ========================================================================
    # CHUNKING PARAMETERS
    # ========================================================================
    
    # Maximum characters per chunk
    # Trade-offs:
    #   - Too small: Loss of context, many chunks
    #   - Too large: Less precise matching, fewer chunks
    # Recommended: 800-1200 for academic papers
    MAX_CHARS = int(os.getenv("MAX_CHARS", "900"))
    
    # Overlap between consecutive chunks (characters)
    # Prevents sentences from being split across chunks
    # Recommended: 10-15% of MAX_CHARS
    OVERLAP = int(os.getenv("OVERLAP", "120"))
    
    # ========================================================================
    # SEARCH PARAMETERS
    # ========================================================================
    
    # Number of top results to return
    # More results = more comprehensive but slower LLM processing
    TOP_K = int(os.getenv("TOP_K", "10"))
    
    # Minimum similarity score to include in results (0.0 to 1.0)
    # Higher threshold = more relevant but potentially fewer results
    MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.0"))
    
    # ========================================================================
    # PERFORMANCE PARAMETERS
    # ========================================================================
    
    # Batch size for embedding generation
    # Larger = faster but more memory
    # For GPU: 128-256, For CPU: 32-64
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
    
    # Enable GPU if available
    USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
    
    # ========================================================================
    # LLM PARAMETERS
    # ========================================================================
    
    # Anthropic API key (required for LLM features)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Claude model to use
    # Options:
    #   - "claude-sonnet-4-20250514"  (balanced)
    #   - "claude-opus-4-20250514"    (most capable)
    #   - "claude-haiku-4-20250514"   (fastest, cheapest)
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    
    # Maximum tokens for LLM responses
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    
    # Enable LLM features
    ENABLE_SUMMARIZATION = os.getenv("ENABLE_SUMMARIZATION", "true").lower() == "true"
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    # Log level (DEBUG, INFO, WARNING, ERROR)
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Log file path (None for stdout only)
    LOG_FILE = os.getenv("LOG_FILE", None)
    
    # ========================================================================
    # ADVANCED OPTIONS
    # ========================================================================
    
    # Use approximate nearest neighbor search (for large datasets)
    # Requires FAISS installation
    USE_ANN = os.getenv("USE_ANN", "false").lower() == "true"
    
    # Cache search results
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    
    # Cache size (number of queries)
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1000"))
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    @classmethod
    def validate(cls):
        """Validate configuration parameters."""
        errors = []
        
        # Check paths
        if not os.path.exists(cls.PDF_DIR):
            errors.append(f"PDF directory not found: {cls.PDF_DIR}")
        
        # Check parameters
        if cls.MAX_CHARS < 100:
            errors.append(f"MAX_CHARS too small: {cls.MAX_CHARS}")
        
        if cls.OVERLAP >= cls.MAX_CHARS:
            errors.append(f"OVERLAP ({cls.OVERLAP}) must be < MAX_CHARS ({cls.MAX_CHARS})")
        
        if cls.TOP_K < 1:
            errors.append(f"TOP_K must be >= 1: {cls.TOP_K}")
        
        # Warn about missing API key
        if not cls.ANTHROPIC_API_KEY and cls.ENABLE_SUMMARIZATION:
            print("⚠️  Warning: ANTHROPIC_API_KEY not set. LLM features will be disabled.")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*80)
        print("CONFIGURATION")
        print("="*80)
        print(f"\nPaths:")
        print(f"  PDF Directory:   {cls.PDF_DIR}")
        print(f"  Index Directory: {cls.INDEX_DIR}")
        print(f"\nEmbedding:")
        print(f"  Model:           {cls.MODEL_NAME}")
        print(f"  Batch Size:      {cls.BATCH_SIZE}")
        print(f"  Use GPU:         {cls.USE_GPU}")
        print(f"\nChunking:")
        print(f"  Max Characters:  {cls.MAX_CHARS}")
        print(f"  Overlap:         {cls.OVERLAP}")
        print(f"\nSearch:")
        print(f"  Top K:           {cls.TOP_K}")
        print(f"  Min Similarity:  {cls.MIN_SIMILARITY}")
        print(f"\nLLM:")
        print(f"  Model:           {cls.LLM_MODEL}")
        print(f"  Max Tokens:      {cls.MAX_TOKENS}")
        print(f"  API Key Set:     {bool(cls.ANTHROPIC_API_KEY)}")
        print(f"  Summarization:   {cls.ENABLE_SUMMARIZATION}")
        print("="*80 + "\n")


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class FastConfig(Config):
    """Optimized for speed."""
    MAX_CHARS = 500
    OVERLAP = 50
    TOP_K = 5
    BATCH_SIZE = 128
    LLM_MODEL = "claude-haiku-4-20250514"


class AccurateConfig(Config):
    """Optimized for accuracy."""
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    MAX_CHARS = 1200
    OVERLAP = 200
    TOP_K = 20
    LLM_MODEL = "claude-opus-4-20250514"


class MedicalConfig(Config):
    """Optimized for medical/scientific text."""
    MODEL_NAME = "allenai/specter"
    MAX_CHARS = 1000
    OVERLAP = 150
    TOP_K = 15


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Validate and print configuration.
    
    Run with:
        python config_example.py
    """
    
    print("Testing default configuration:")
    Config.validate()
    Config.print_config()
    
    print("\nTesting fast configuration:")
    FastConfig.print_config()
    
    print("\nTesting accurate configuration:")
    AccurateConfig.print_config()
    
    print("\nTesting medical configuration:")
    MedicalConfig.print_config()
