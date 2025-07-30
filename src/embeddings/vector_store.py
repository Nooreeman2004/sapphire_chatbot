"""
Build vector store from embedded chunks - standalone version.
No external imports needed.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import faiss
import uuid

# LangChain imports
from langchain.vectorstores import FAISS as LangChainFAISS
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain compatible wrapper for SentenceTransformer embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


def load_embedded_chunks(embeddings_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all embedded chunk files."""
    all_chunks = {}
    
    # Expected embedded files
    embedded_files = [
        'embedded_faq_chunks.json',
        'embedded_policy_chunks.json', 
        'embedded_product_chunks.json',
        'embedded_web_content_chunks.json'
    ]
    
    for filename in embedded_files:
        file_path = embeddings_dir / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                # Remove 'embedded_' prefix and '.json' suffix for chunk type
                chunk_type = filename.replace('embedded_', '').replace('.json', '')
                all_chunks[chunk_type] = chunks
                logger.info(f"Loaded {len(chunks)} chunks from {filename}")
        else:
            logger.warning(f"File not found: {filename}")
    
    return all_chunks


def create_langchain_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    """Convert chunks to LangChain Document objects."""
    documents = []
    for chunk in chunks:
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        
        # Add additional metadata
        metadata.update({
            'chunk_id': chunk.get('chunk_id', ''),
            'chunk_type': chunk.get('chunk_type', ''),
            'product_id': chunk.get('product_id', ''),
            'title': chunk.get('title', ''),
            'question': chunk.get('question', ''),
            'answer': chunk.get('answer', ''),
            'policy_type': chunk.get('policy_type', '')
        })
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents


def build_langchain_faiss_store(all_chunks: Dict[str, List[Dict[str, Any]]], 
                                output_dir: Path) -> None:
    """Build LangChain FAISS vector store."""
    logger.info("Building LangChain FAISS vector store...")
    
    # Create embedding model
    embeddings = SentenceTransformerEmbeddings()
    
    # Combine all chunks into documents
    all_documents = []
    for chunk_type, chunks in all_chunks.items():
        documents = create_langchain_documents(chunks)
        all_documents.extend(documents)
    
    logger.info(f"Created {len(all_documents)} documents")
    
    # Create FAISS vector store
    vector_store = LangChainFAISS.from_documents(all_documents, embeddings)
    
    # Save vector store
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(output_dir))
    
    logger.info(f"LangChain FAISS vector store saved to: {output_dir}")
    
    # Test the vector store
    test_langchain_store(vector_store)


def build_custom_faiss_store(all_chunks: Dict[str, List[Dict[str, Any]]], 
                             output_dir: Path) -> None:
    """Build custom FAISS vector store using pre-computed embeddings."""
    logger.info("Building custom FAISS vector store...")
    
    # Combine all chunks and extract embeddings
    all_documents = []
    all_embeddings = []
    
    for chunk_type, chunks in all_chunks.items():
        for chunk in chunks:
            if 'embedding' in chunk:
                all_documents.append(chunk)
                all_embeddings.append(chunk['embedding'])
    
    if not all_embeddings:
        logger.error("No embeddings found in chunks!")
        return
    
    logger.info(f"Total documents: {len(all_documents)}")
    logger.info(f"Embedding dimension: {len(all_embeddings[0])}")
    
    # Create FAISS index
    embedding_dim = len(all_embeddings[0])
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings_array)
    
    # Add to index
    index.add(embeddings_array)
    
    # Save everything
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, str(output_dir / "faiss.index"))
    
    # Save documents
    with open(output_dir / "documents.json", 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Custom FAISS vector store saved to: {output_dir}")
    
    # Test the vector store
    test_custom_store(index, all_documents)


def test_langchain_store(vector_store: LangChainFAISS) -> None:
    """Test the LangChain vector store."""
    logger.info("Testing LangChain vector store...")
    
    test_queries = [
        "What is the return policy?",
        "How do I contact customer service?",
        "Tell me about shipping options"
    ]
    
    for query in test_queries:
        try:
            results = vector_store.similarity_search_with_score(query, k=3)
            logger.info(f"Query: '{query}' - Found {len(results)} results")
            
            for i, (doc, score) in enumerate(results):
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                logger.info(f"  Result {i+1} (score: {score:.4f}): {content_preview}")
                
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")


def test_custom_store(index: faiss.Index, documents: List[Dict[str, Any]]) -> None:
    """Test the custom FAISS store."""
    logger.info("Testing custom FAISS vector store...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        test_queries = [
            "What is the return policy?",
            "How do I contact customer service?"
        ]
        
        for query in test_queries:
            try:
                # Encode query
                query_embedding = model.encode([query], convert_to_numpy=True)
                faiss.normalize_L2(query_embedding)
                
                # Search
                scores, indices = index.search(query_embedding, 3)
                
                logger.info(f"Query: '{query}' - Found {len(indices[0])} results")
                
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(documents):
                        doc = documents[idx]
                        content_preview = doc.get('content', '')[:100] + "..."
                        logger.info(f"  Result {i+1} (score: {score:.4f}): {content_preview}")
                        
            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
                
    except ImportError:
        logger.warning("SentenceTransformers not available for testing")



def main():
    """Main function to build vector stores."""
    print("🚀 Building Vector Stores from Embedded Chunks")
    print("=" * 60)
    
    # Define paths
    current_dir = Path(__file__).parent
    embeddings_dir = current_dir.parent.parent / "data" / "embeddings"
    
    # Check if embeddings directory exists
    if not embeddings_dir.exists():
        print(f"❌ Embeddings directory not found: {embeddings_dir}")
        print("Please run create_embeddings.py first to generate embeddings.")
        return
    
    # Load embedded chunks
    print(f"📂 Loading embedded chunks from: {embeddings_dir}")
    all_chunks = load_embedded_chunks(embeddings_dir)
    
    if not all_chunks:
        print("❌ No embedded chunks found!")
        return
    
    total_chunks = sum(len(chunks) for chunks in all_chunks.values())
    print(f"✅ Loaded {total_chunks} chunks from {len(all_chunks)} files")
    
    # Build LangChain FAISS store
    print("\n🔧 Building LangChain FAISS vector store...")
    try:
        langchain_output = current_dir.parent.parent / "data" / "vector_store_langchain"
        build_langchain_faiss_store(all_chunks, langchain_output)
        print("✅ LangChain FAISS vector store built successfully!")
    except Exception as e:
        print(f"❌ Error building LangChain store: {e}")
    
    # Build custom FAISS store
    print("\n🔧 Building custom FAISS vector store...")
    try:
        custom_output = current_dir.parent.parent / "data" / "vector_store_custom"
        build_custom_faiss_store(all_chunks, custom_output)
        print("✅ Custom FAISS vector store built successfully!")
    except Exception as e:
        print(f"❌ Error building custom store: {e}")
    
    print("\n🎉 Vector store building completed!")
    print("\nCreated directories:")
    print("- data/vector_store_langchain/ (LangChain FAISS - recommended)")
    print("- data/vector_store_custom/ (Custom FAISS)")
    print("\nYou can now use these vector stores in your RAG pipeline!")
    
class LangChainVectorStore:
    """Wrapper for LangChain FAISS vector store for Sapphire RAG pipeline."""
    
    def __init__(self, store_type: str = "faiss"):
        self.store_type = store_type.lower()
        self.vector_store = None
        self.embeddings = SentenceTransformerEmbeddings()

    def load(self, vector_store_path: str):
        """Load the FAISS vector store from the specified path."""
        try:
            logger.info(f"Loading {self.store_type} vector store from: {vector_store_path}")
            if self.store_type == "faiss":
                self.vector_store = LangChainFAISS.load_local(
                    vector_store_path, self.embeddings, allow_dangerous_deserialization=True
                )
            else:
                raise ValueError(f"Unsupported store type: {self.store_type}")
            logger.info("Vector store loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search the vector store for relevant documents."""
        try:
            if self.vector_store is None:
                raise ValueError("Vector store not initialized")
            logger.info(f"Searching vector store with query: {query}")
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []


if __name__ == "__main__":
    main()