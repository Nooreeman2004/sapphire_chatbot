"""
Helper utilities for Sapphire RAG pipeline.
Includes JSON parsing, text cleaning, and other utility functions.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import hashlib
from datetime import datetime


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_price(text: str) -> Optional[float]:
    """
    Extract price from text (supports PKR format).
    
    Args:
        text: Text containing price
        
    Returns:
        Extracted price as float, or None if not found
    """
    # Pattern for PKR prices like "Rs.2,990.00" or "PKR 2990"
    patterns = [
        r'Rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'PKR\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:Rs|PKR)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            price_str = match.group(1).replace(',', '')
            try:
                return float(price_str)
            except ValueError:
                continue
    
    return None


def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a unique ID.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of the random part
        
    Returns:
        Generated ID
    """
    random_part = str(uuid.uuid4()).replace('-', '')[:length]
    return f"{prefix}{random_part}" if prefix else random_part


def hash_text(text: str) -> str:
    """
    Generate a hash for text content.
    
    Args:
        text: Input text
        
    Returns:
        SHA-256 hash of the text
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at word boundary
        if end < len(text):
            # Find the last space before the end
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start <= 0:
            start = end
    
    return chunks


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}{sep}{i}", item))
        else:
            items.append((new_key, v))
    
    return dict(items)


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def format_currency(amount: float, currency: str = "PKR") -> str:
    """
    Format currency amount.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency.upper() == "PKR":
        return f"Rs.{amount:,.2f}"
    else:
        return f"{currency} {amount:,.2f}"


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted URLs
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure it's not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    return Path(file_path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO formatted timestamp
    """
    return datetime.now().isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse ISO timestamp string to datetime object.
    
    Args:
        timestamp_str: ISO formatted timestamp string
        
    Returns:
        Datetime object
    """
    return datetime.fromisoformat(timestamp_str)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def safe_get(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary value using dot notation.
    
    Args:
        data: Dictionary to search
        key_path: Dot-separated key path (e.g., "user.profile.name")
        default: Default value if key not found
        
    Returns:
        Value at key path or default
    """
    keys = key_path.split('.')
    current = data
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


if __name__ == "__main__":
    # Example usage
    
    # Test text cleaning
    dirty_text = "  This is   a test\n\nwith   extra   spaces!  "
    clean = clean_text(dirty_text)
    print(f"Original: '{dirty_text}'")
    print(f"Cleaned: '{clean}'")
    
    # Test price extraction
    price_text = "The shirt costs Rs.2,990.00"
    price = extract_price(price_text)
    print(f"Extracted price: {price}")
    
    # Test ID generation
    test_id = generate_id("PROD_", 8)
    print(f"Generated ID: {test_id}")
    
    # Test text chunking
    long_text = "This is a very long text that needs to be split into smaller chunks for processing. " * 10
    chunks = chunk_text(long_text, chunk_size=100, overlap=20)
    print(f"Text split into {len(chunks)} chunks")
    
    # Test currency formatting
    amount = 2990.50
    formatted = format_currency(amount)
    print(f"Formatted currency: {formatted}")
    
    print("Helper utilities test completed!")

