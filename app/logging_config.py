import logging

def setup_logging(log_file):
    """
    Setup logging configuration.

    Parameters:
    - log_file (str): Path to the log file.

    Returns:
    - None
    """
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(message)s')

def log_message(message):
    """
    Log a message.

    Parameters:
    - message (str): The message to log.

    Returns:
    - None
    """
    logging.info(message)

def log_error(error_message):
    """
    Log an error message.

    Parameters:
    - error_message (str): The error message to log.

    Returns:
    - None
    """
    logging.error(error_message)