"""
PDF-based RAG (Retrieval-Augmented Generation) for HFMD Guidelines
Loads PDF guidelines and provides relevant chunks for LLM prompts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import pickle

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("[WARN] langchain not installed. PDF RAG disabled.")


class PDFRAGLoader:
    """RAG system for loading and querying PDF guidelines"""
    
    def __init__(
        self,
        pdf_paths: List[str],
        cache_dir: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize PDF RAG loader.
        
        Args:
            pdf_paths: List of paths to PDF files
            cache_dir: Directory to cache vectorstore (optional)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain is required for PDF RAG. Install with: pip install langchain langchain-community pypdf faiss-cpu")
        
        self.pdf_paths = [Path(p) for p in pdf_paths]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.embeddings = None
        
        # Check if PDFs exist
        for pdf_path in self.pdf_paths:
            if not pdf_path.exists():
                print(f"[WARN] PDF not found: {pdf_path}")
    
    def _get_cache_path(self) -> Optional[Path]:
        """Get cache file path based on PDF paths"""
        if not self.cache_dir:
            return None
        
        # Create cache filename from PDF names
        cache_name = "_".join([p.stem for p in self.pdf_paths]) + ".faiss"
        return self.cache_dir / cache_name
    
    def load_and_index(self, force_reload: bool = False) -> bool:
        """
        Load PDFs and create vector index.
        
        Args:
            force_reload: Force reload even if cache exists
            
        Returns:
            True if successful, False otherwise
        """
        cache_path = self._get_cache_path()
        
        # Try to load from cache
        if not force_reload and cache_path and cache_path.exists():
            try:
                print(f"[INFO] Loading cached vectorstore from {cache_path}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                self.vectorstore = FAISS.load_local(
                    str(cache_path.parent),
                    self.embeddings,
                    cache_path.name
                )
                print(f"[INFO] Loaded {self.vectorstore.index.ntotal} vectors from cache")
                return True
            except Exception as e:
                print(f"[WARN] Failed to load cache: {e}. Reloading PDFs...")
        
        # Load PDFs
        all_docs = []
        for pdf_path in self.pdf_paths:
            if not pdf_path.exists():
                continue
            
            try:
                print(f"[INFO] Loading PDF: {pdf_path}")
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata["source_file"] = pdf_path.name
                
                all_docs.extend(docs)
                print(f"[INFO] Loaded {len(docs)} pages from {pdf_path.name}")
            except Exception as e:
                print(f"[ERROR] Failed to load {pdf_path}: {e}")
        
        if not all_docs:
            print("[ERROR] No documents loaded from PDFs")
            return False
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        splits = text_splitter.split_documents(all_docs)
        print(f"[INFO] Split into {len(splits)} chunks")
        
        # Create embeddings and vectorstore
        try:
            print("[INFO] Creating embeddings (this may take a minute)...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            print(f"[INFO] Created vectorstore with {self.vectorstore.index.ntotal} vectors")
            
            # Save cache
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.vectorstore.save_local(
                    str(cache_path.parent),
                    cache_path.name
                )
                print(f"[INFO] Saved vectorstore cache to {cache_path}")
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to create vectorstore: {e}")
            return False
    
    def search(
        self,
        query: str,
        k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dicts with 'content', 'metadata', 'score'
        """
        if not self.vectorstore:
            print("[ERROR] Vectorstore not initialized. Call load_and_index() first.")
            return []
        
        try:
            # Search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + score)
                
                if score_threshold and similarity < score_threshold:
                    continue
                
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": similarity,
                    "distance": score
                })
            
            return formatted_results
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []
    
    def get_context_for_prompt(
        self,
        query: str,
        k: int = 3,
        max_length: int = 1500
    ) -> str:
        """
        Get formatted context string for LLM prompt.
        
        Args:
            query: Search query (e.g., "HFMD seasonality weather transmission")
            k: Number of chunks to retrieve
            max_length: Maximum total character length
            
        Returns:
            Formatted context string
        """
        results = self.search(query, k=k)
        
        if not results:
            return ""
        
        context_parts = []
        total_len = 0
        
        for i, result in enumerate(results):
            content = result["content"].strip()
            source = result["metadata"].get("source_file", "Unknown")
            page = result["metadata"].get("page", "?")
            score = result["score"]
            
            chunk_text = f"[Source: {source}, Page {page}, Relevance: {score:.2f}]\n{content}\n"
            
            if total_len + len(chunk_text) > max_length:
                break
            
            context_parts.append(chunk_text)
            total_len += len(chunk_text)
        
        if not context_parts:
            return ""
        
        header = "=== RELEVANT GUIDELINE EXCERPTS ===\n\n"
        footer = "\n=== END GUIDELINE EXCERPTS ===\n"
        
        return header + "\n---\n\n".join(context_parts) + footer


# Singleton instance for HFMD
_HFMD_RAG_INSTANCE: Optional[PDFRAGLoader] = None


def get_hfmd_rag_loader(force_reload: bool = False) -> Optional[PDFRAGLoader]:
    """
    Get or create singleton RAG loader for HFMD guidelines.
    
    Args:
        force_reload: Force reload PDFs even if already initialized
        
    Returns:
        PDFRAGLoader instance or None if initialization failed
    """
    global _HFMD_RAG_INSTANCE
    
    if not LANGCHAIN_AVAILABLE:
        return None
    
    if _HFMD_RAG_INSTANCE and not force_reload:
        return _HFMD_RAG_INSTANCE
    
    # Find PDF paths
    try:
        base_dir = Path(__file__).resolve().parents[1]  # experiments/core/ -> experiments/
        data_dir = base_dir / "data_for_model" / "手足口病"
        root_dir = base_dir.parent  # med_deepseek/
        
        pdf_paths = [
            data_dir / "Diagnosis for hand, foot and mouth disease guideline.pdf",
            root_dir / "data" / "Epidemic_guide.pdf"  # med_deepseek/data/
        ]
        
        # Filter existing PDFs
        existing_pdfs = [str(p) for p in pdf_paths if p.exists()]
        
        if not existing_pdfs:
            print("[WARN] No HFMD guideline PDFs found. RAG disabled.")
            return None
        
        print(f"[INFO] Found {len(existing_pdfs)} HFMD guideline PDFs")
        
        # Cache directory
        cache_dir = base_dir / ".cache" / "rag"
        
        # Create loader
        loader = PDFRAGLoader(
            pdf_paths=existing_pdfs,
            cache_dir=str(cache_dir),
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Load and index
        success = loader.load_and_index(force_reload=force_reload)
        
        if not success:
            print("[ERROR] Failed to initialize HFMD RAG loader")
            return None
        
        _HFMD_RAG_INSTANCE = loader
        return loader
    
    except Exception as e:
        print(f"[ERROR] Failed to create HFMD RAG loader: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_rag_loader():
    """Test function to verify RAG loader works"""
    print("\n" + "="*80)
    print("Testing HFMD RAG Loader")
    print("="*80 + "\n")
    
    loader = get_hfmd_rag_loader(force_reload=True)
    
    if not loader:
        print("❌ Failed to initialize RAG loader")
        return False
    
    print("✅ RAG loader initialized successfully\n")
    
    # Test queries
    test_queries = [
        "HFMD seasonal transmission pattern peak",
        "hand foot mouth disease weather temperature",
        "school transmission outbreak children",
        "incubation period symptoms diagnosis"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print("="*80)
        
        context = loader.get_context_for_prompt(query, k=2, max_length=1000)
        
        if context:
            print(context)
        else:
            print("❌ No results found")
    
    print("\n" + "="*80)
    print("RAG Loader Test Complete")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    test_rag_loader()
