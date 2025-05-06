"""
Logging configuration for the sales forecasting application.
"""

import logging
import sys
import os
from datetime import datetime

def setup_logger(name, log_level='INFO', log_file=None):
    """
    Set up and configure logger
    
    Args:
        name (str): Logger name
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): File to save logs to
        
    Returns:
        Logger: Configured logger instance
    """
    # Convert string log level to logging constant
    level = getattr(logging, log_level.upper())
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else 'logs'
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
