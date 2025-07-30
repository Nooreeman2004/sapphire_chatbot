"""
Data chunking module for Sapphire RAG pipeline.
Implements semantic product chunking and Q&A chunking strategies.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductChunker:
    """Semantic chunking for product data."""
    
    def __init__(self):
        self.chunk_types = {
            'product_overview': 'Product overview and basic information',
            'product_details': 'Detailed product specifications',
            'product_care': 'Care instructions and fabric information',
            'product_availability': 'Pricing, availability, and purchasing info'
        }
    
    def chunk_product(self, product: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create semantic chunks for a single product.
        
        Args:
            product: Product dictionary
            
        Returns:
            List of product chunks
        """
        chunks = []
        
        # Product Overview Chunk
        overview_content = self._create_overview_content(product)
        chunks.append({
            'chunk_id': str(uuid.uuid4()),
            'chunk_type': 'product_overview',
            'product_id': product.get('id', ''),
            'title': product.get('title', ''),
            'content': overview_content,
            'metadata': {
                'category': product.get('category', ''),
                'subcategory': product.get('subcategory', ''),
                'collection': product.get('collection', ''),
                'badge': product.get('badge', ''),
                'price_numeric': product.get('price_numeric', 0)
            }
        })
        
        # Product Details Chunk
        details_content = self._create_details_content(product)
        chunks.append({
            'chunk_id': str(uuid.uuid4()),
            'chunk_type': 'product_details',
            'product_id': product.get('id', ''),
            'title': product.get('title', ''),
            'content': details_content,
            'metadata': {
                'fabric': product.get('fabric', ''),
                'pattern': product.get('pattern', ''),
                'silhouette': product.get('silhouette', ''),
                'colors': product.get('colors', []),
                'sizes': product.get('sizes', [])
            }
        })
        
        # Product Care Chunk
        if product.get('care_instructions') or product.get('fabric_info'):
            care_content = self._create_care_content(product)
            chunks.append({
                'chunk_id': str(uuid.uuid4()),
                'chunk_type': 'product_care',
                'product_id': product.get('id', ''),
                'title': product.get('title', ''),
                'content': care_content,
                'metadata': {
                    'fabric': product.get('fabric', ''),
                    'care_instructions': product.get('care_instructions', '')
                }
            })
        
        # Product Availability Chunk
        availability_content = self._create_availability_content(product)
        chunks.append({
            'chunk_id': str(uuid.uuid4()),
            'chunk_type': 'product_availability',
            'product_id': product.get('id', ''),
            'title': product.get('title', ''),
            'content': availability_content,
            'metadata': {
                'price': product.get('price', ''),
                'price_numeric': product.get('price_numeric', 0),
                'availability': product.get('availability', ''),
                'url': product.get('url', '')
            }
        })
        
        return chunks
    
    def _create_overview_content(self, product: Dict[str, Any]) -> str:
        """Create overview content for product."""
        content_parts = []
        
        # Basic product info
        title = product.get('title', '')
        category = product.get('category', '')
        subcategory = product.get('subcategory', '')
        collection = product.get('collection', '')
        description = product.get('description', '')
        
        content_parts.append(f"Product: {title}")
        
        if category:
            content_parts.append(f"Category: {category}")
        if subcategory:
            content_parts.append(f"Subcategory: {subcategory}")
        if collection:
            content_parts.append(f"Collection: {collection}")
        if description:
            content_parts.append(f"Description: {description}")
        
        # Badge information
        badge = product.get('badge', '')
        if badge:
            content_parts.append(f"Special Badge: {badge}")
        
        return " | ".join(content_parts)
    
    def _create_details_content(self, product: Dict[str, Any]) -> str:
        """Create detailed specifications content."""
        content_parts = []
        
        title = product.get('title', '')
        content_parts.append(f"Product: {title}")
        
        # Fabric and design details
        fabric = product.get('fabric', '')
        pattern = product.get('pattern', '')
        silhouette = product.get('silhouette', '')
        
        if fabric:
            content_parts.append(f"Fabric: {fabric}")
        if pattern:
            content_parts.append(f"Pattern: {pattern}")
        if silhouette:
            content_parts.append(f"Silhouette: {silhouette}")
        
        # Colors and sizes
        colors = product.get('colors', [])
        sizes = product.get('sizes', [])
        
        if colors:
            content_parts.append(f"Available Colors: {', '.join(colors)}")
        if sizes:
            content_parts.append(f"Available Sizes: {', '.join(sizes)}")
        
        # Features
        features = product.get('features', [])
        if features:
            content_parts.append(f"Features: {', '.join(features)}")
        
        # Fabric info if available
        fabric_info = product.get('fabric_info', {})
        if fabric_info:
            if fabric_info.get('description'):
                content_parts.append(f"Fabric Description: {fabric_info['description']}")
            if fabric_info.get('season'):
                content_parts.append(f"Best Season: {fabric_info['season']}")
        
        # Silhouette info
        silhouette_info = product.get('silhouette_info', '')
        if silhouette_info:
            content_parts.append(f"Silhouette Details: {silhouette_info}")
        
        return " | ".join(content_parts)
    
    def _create_care_content(self, product: Dict[str, Any]) -> str:
        """Create care instructions content."""
        content_parts = []
        
        title = product.get('title', '')
        fabric = product.get('fabric', '')
        content_parts.append(f"Product: {title}")
        
        if fabric:
            content_parts.append(f"Fabric: {fabric}")
        
        # Care instructions
        care_instructions = product.get('care_instructions', '')
        if care_instructions:
            content_parts.append(f"Care Instructions: {care_instructions}")
        
        # Fabric care info
        fabric_info = product.get('fabric_info', {})
        if fabric_info and fabric_info.get('care'):
            content_parts.append(f"Fabric Care: {fabric_info['care']}")
        
        return " | ".join(content_parts)
    
    def _create_availability_content(self, product: Dict[str, Any]) -> str:
        """Create availability and pricing content."""
        content_parts = []
        
        title = product.get('title', '')
        content_parts.append(f"Product: {title}")
        
        # Pricing
        price = product.get('price', '')
        if price:
            content_parts.append(f"Price: {price}")
        
        # Availability
        availability = product.get('availability', '')
        if availability:
            content_parts.append(f"Availability: {availability}")
        
        # URL for purchasing
        url = product.get('url', '')
        if url:
            content_parts.append(f"Product URL: {url}")
        
        return " | ".join(content_parts)


class FAQChunker:
    """Q&A chunking for FAQ data."""
    
    def chunk_faq(self, faq_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a chunk for a single FAQ item.
        
        Args:
            faq_item: FAQ dictionary with question, answer, category
            
        Returns:
            FAQ chunk
        """
        question = faq_item.get('question', '')
        answer = faq_item.get('answer', '')
        category = faq_item.get('category', '')
        
        content = f"Question: {question} | Answer: {answer}"
        
        return {
            'chunk_id': str(uuid.uuid4()),
            'chunk_type': 'faq',
            'content': content,
            'question': question,
            'answer': answer,
            'metadata': {
                'category': category,
                'type': 'faq'
            }
        }


class PolicyChunker:
    """Chunking for policy documents."""
    
    def chunk_policy(self, policy_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a chunk for a single policy item.
        
        Args:
            policy_item: Policy dictionary
            
        Returns:
            Policy chunk
        """
        policy_type = policy_item.get('policy_type', '')
        title = policy_item.get('title', '')
        content = policy_item.get('content', '')
        details = policy_item.get('details', [])
        
        content_parts = [f"Policy: {title}", f"Content: {content}"]
        
        if details:
            content_parts.append(f"Details: {' | '.join(details)}")
        
        full_content = " | ".join(content_parts)
        
        return {
            'chunk_id': str(uuid.uuid4()),
            'chunk_type': 'policy',
            'policy_type': policy_type,
            'title': title,
            'content': full_content,
            'metadata': {
                'policy_type': policy_type,
                'type': 'policy'
            }
        }


class WebContentChunker:
    """Chunking for web scraped content."""
    
    def chunk_web_content(self, page_data: Dict[str, Any], chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Create chunks for web page content.
        
        Args:
            page_data: Web page data
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of web content chunks
        """
        chunks = []
        url = page_data.get('url', '')
        title = page_data.get('title', '')
        text_content = page_data.get('text_content', '')
        
        # Split content into chunks
        content_chunks = self._split_text(text_content, chunk_size)
        
        for i, chunk_text in enumerate(content_chunks):
            chunks.append({
                'chunk_id': str(uuid.uuid4()),
                'chunk_type': 'web_content',
                'content': chunk_text,
                'metadata': {
                    'url': url,
                    'title': title,
                    'chunk_index': i,
                    'type': 'web_content'
                }
            })
        
        return chunks
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of specified size."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class DataProcessor:
    """Main data processing class that orchestrates all chunking."""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create the full directory path if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.product_chunker = ProductChunker()
        self.faq_chunker = FAQChunker()
        self.policy_chunker = PolicyChunker()
        self.web_chunker = WebContentChunker()
    
    def process_all_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all data files and create chunks.
        
        Returns:
            Dictionary with chunk types and their chunks
        """
        all_chunks = {
            'product_chunks': [],
            'faq_chunks': [],
            'policy_chunks': [],
            'web_content_chunks': []
        }
        
        # Process product files
        product_files = [
            'sapphire_products.json',
            'sapphire_products_comprehensive.json'
        ]
        
        for filename in product_files:
            file_path = self.data_dir / filename
            if file_path.exists():
                logger.info(f"Processing product file: {filename}")
                chunks = self._process_product_file(file_path)
                all_chunks['product_chunks'].extend(chunks)
            else:
                logger.warning(f"Product file not found: {file_path}")
        
        # Process FAQ file
        faq_file = self.data_dir / 'faqs.json'
        if faq_file.exists():
            logger.info("Processing FAQ file")
            chunks = self._process_faq_file(faq_file)
            all_chunks['faq_chunks'].extend(chunks)
        else:
            logger.warning(f"FAQ file not found: {faq_file}")
        
        # Process policy file
        policy_file = self.data_dir / 'exchange_return_policy.json'
        if policy_file.exists():
            logger.info("Processing policy file")
            chunks = self._process_policy_file(policy_file)
            all_chunks['policy_chunks'].extend(chunks)
        else:
            logger.warning(f"Policy file not found: {policy_file}")
        
        # Process web content file
        web_file = self.data_dir / 'sapphire_rag_complete.json'
        if web_file.exists():
            logger.info("Processing web content file")
            chunks = self._process_web_content_file(web_file)
            all_chunks['web_content_chunks'].extend(chunks)
        else:
            logger.warning(f"Web content file not found: {web_file}")
        
        # Save all chunks
        self._save_chunks(all_chunks)
        
        return all_chunks
    
    def _process_product_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a product JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []
        
        chunks = []
        products = data.get('products', [])
        
        for product in products:
            product_chunks = self.product_chunker.chunk_product(product)
            chunks.extend(product_chunks)
        
        logger.info(f"Created {len(chunks)} product chunks from {file_path.name}")
        return chunks
    
    def _process_faq_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process FAQ JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                faqs = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []
        
        chunks = []
        for faq in faqs:
            chunk = self.faq_chunker.chunk_faq(faq)
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} FAQ chunks")
        return chunks
    
    def _process_policy_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process policy JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                policies = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []
        
        chunks = []
        for policy in policies:
            chunk = self.policy_chunker.chunk_policy(policy)
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} policy chunks")
        return chunks
    
    def _process_web_content_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process web content JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []
        
        chunks = []
        pages = data.get('pages', [])
        
        for page in pages:
            page_chunks = self.web_chunker.chunk_web_content(page)
            chunks.extend(page_chunks)
        
        logger.info(f"Created {len(chunks)} web content chunks")
        return chunks
    
    def _save_chunks(self, all_chunks: Dict[str, List[Dict[str, Any]]]):
        """Save all chunks to separate files."""
        for chunk_type, chunks in all_chunks.items():
            if chunks:
                output_file = self.output_dir / f"{chunk_type}.json"
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(chunks, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved {len(chunks)} chunks to {output_file}")
                except Exception as e:
                    logger.error(f"Error saving {output_file}: {e}")


if __name__ == "__main__":
    # Get the script directory and construct proper paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up two levels to project root
    
    data_dir = project_root / "data"
    output_dir = project_root / "data" / "processed_chunks"
    
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if data directory exists
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure your data files are in the correct location")
        exit(1)
    
    processor = DataProcessor(str(data_dir), str(output_dir))
    all_chunks = processor.process_all_data()
    
    print("Data processing completed!")
    for chunk_type, chunks in all_chunks.items():
        print(f"{chunk_type}: {len(chunks)} chunks")