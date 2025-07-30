"""
Simple script to run chunk embeddings from the embeddings directory.
"""

import sys
import os
from pathlib import Path

# Add the current directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from process_chunks_embeddings import ChunkEmbeddingProcessor

def main():
    # Set paths relative to current directory
    chunks_dir = "../../data/processed_chunks"
    output_dir = "../../data/embeddings"
    
    print("Starting chunk embedding process...")
    print(f"Input directory: {chunks_dir}")
    print(f"Output directory: {output_dir}")
    
    processor = ChunkEmbeddingProcessor(chunks_dir=chunks_dir, output_dir=output_dir)
    
    # Process all chunk files
    processor.process_all_files()
    
    # Show statistics
    processor.get_embedding_stats()
    
    print("Embedding process completed!")

if __name__ == "__main__":
    main()