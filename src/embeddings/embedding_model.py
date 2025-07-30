# D:\sapphire_chatbot\src\embeddings\embedding_model.py
"""
Create embeddings for all processed chunks.
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Class for handling embedding generation."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        logger.info("Embeddings generated successfully.")
        return embeddings

def load_chunks_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load chunks from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
        return chunks
    except Exception as e:
        logger.error(f"Error loading chunks from {file_path}: {e}")
        return []

def create_embeddings_for_chunks(chunks: List[Dict[str, Any]], model: EmbeddingModel) -> List[Dict[str, Any]]:
    """Create embeddings for a list of chunks."""
    if not chunks:
        return []
    
    # Extract text content for embedding
    texts_to_embed = []
    for chunk in chunks:
        text = ""
        if 'title' in chunk and chunk['title']:
            text += chunk['title'] + " "
        if 'content' in chunk and chunk['content']:
            text += chunk['content']
        texts_to_embed.append(text.strip())
    
    # Generate embeddings
    embeddings_list = model.encode(texts_to_embed)
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings_list[i]
    
    return chunks

def process_all_chunk_files():
    """Process all chunk files and create embeddings."""
    current_dir = Path(__file__).parent
    chunks_dir = current_dir.parent.parent / "data" / "processed_chunks"
    output_dir = current_dir.parent.parent / "data" / "embeddings"
    
    logger.info(f"Looking for chunks in: {chunks_dir}")
    logger.info(f"Will save embeddings to: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not chunks_dir.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        return
    
    model = EmbeddingModel()
    json_files = list(chunks_dir.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {chunks_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    total_chunks = 0
    
    for file_path in json_files:
        try:
            logger.info(f"Processing file: {file_path.name}")
            chunks = load_chunks_from_file(file_path)
            if not chunks:
                logger.warning(f"No chunks found in {file_path.name}")
                continue
            
            embedded_chunks = create_embeddings_for_chunks(chunks, model)
            output_file = output_dir / f"embedded_{file_path.name}"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved embedded chunks to: {output_file}")
            total_chunks += len(embedded_chunks)
            
            if embedded_chunks and 'embedding' in embedded_chunks[0]:
                embedding_dim = len(embedded_chunks[0]['embedding'])
                logger.info(f"  - {len(embedded_chunks)} chunks with {embedding_dim}-dimensional embeddings")
        
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            continue
    
    logger.info(f"Finished processing all files. Total embedded chunks: {total_chunks}")

def main():
    """Main function."""
    print("Starting chunk embedding process...")
    try:
        process_all_chunk_files()
        print("Embedding process completed successfully!")
    except Exception as e:
        logger.error(f"Error in embedding process: {e}")
        print("Embedding process failed. Check logs for details.")

if __name__ == "__main__":
    main()