"""
Utilities module for Sapphire RAG pipeline.
"""

from .logger import setup_logging, get_logger, SapphireLogger
from .helpers import (
    load_json, save_json, clean_text, extract_price,
    generate_id, hash_text, chunk_text, flatten_dict,
    validate_email, format_currency, extract_urls,
    sanitize_filename, get_file_size, format_file_size,
    get_timestamp, parse_timestamp, merge_dicts, safe_get
)

__all__ = [
    'setup_logging', 'get_logger', 'SapphireLogger',
    'load_json', 'save_json', 'clean_text', 'extract_price',
    'generate_id', 'hash_text', 'chunk_text', 'flatten_dict',
    'validate_email', 'format_currency', 'extract_urls',
    'sanitize_filename', 'get_file_size', 'format_file_size',
    'get_timestamp', 'parse_timestamp', 'merge_dicts', 'safe_get'
]

