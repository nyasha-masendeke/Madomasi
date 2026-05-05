"""Utility functions for logging, config loading, and image processing."""
import logging

def setup_logger(name: str, log_file: str = "logs/app.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)
