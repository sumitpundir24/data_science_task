import logging
import os

def setup_logging(log_file_name='application.log'):
    """
    Sets up logging for the application.
    Args:
    log_file_name (str): Name of the log file. Default is 'application.log'.
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_file_path = os.path.join('logs', log_file_name)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logging.info('Logging setup complete.')

def log_info(message):
    """
    Logs an informational message.
    Args:
    message (str): The message to log.
    """
    logging.info(message)

def log_error(message):
    """
    Logs an error message.
    Args:
    message (str): The message to log.
    """
    logging.error(message)

def log_warning(message):
    """
    Logs a warning message.
    Args:
    message (str): The message to log.
    """
    logging.warning(message)
