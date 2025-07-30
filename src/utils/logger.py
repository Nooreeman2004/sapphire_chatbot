"""
Logging utility for Sapphire RAG pipeline.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


class SapphireLogger:
    """
    Custom logger for Sapphire RAG pipeline with file rotation and formatting.
    """
    
    def __init__(
        self,
        name: str = "sapphire_rag",
        level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: str = "10MB",
        backup_count: int = 5
    ):
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_file = log_file
        self.max_file_size = self._parse_size(max_file_size)
        self.backup_count = backup_count
        
        self.logger = self._setup_logger()
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with appropriate handlers and formatting."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.log_file:
            # Create log directory if it doesn't exist
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: str = "10MB",
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    sapphire_logger = SapphireLogger(
        level=level,
        log_file=log_file,
        max_file_size=max_file_size,
        backup_count=backup_count
    )
    
    return sapphire_logger.get_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default logger for the application
default_logger = setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "./logs/sapphire_rag.log")
)


if __name__ == "__main__":
    # Example usage
    logger = setup_logging(
        level="DEBUG",
        log_file="./logs/test.log",
        max_file_size="5MB",
        backup_count=3
    )
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    print("Logging test completed. Check ./logs/test.log")

