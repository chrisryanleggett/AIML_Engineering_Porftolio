import logging
import sys

def setup_logger(name="CodebaseAgent"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    
    # File Handler
    file_handler = logging.FileHandler("agent.log")
    file_handler.setFormatter(formatter)
    
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger