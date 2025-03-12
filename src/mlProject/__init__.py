import os 
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"

os.makedirs(log_dir, exist_ok=True)

log_filepath = os.path.join(log_dir,'running_log.log')

logging.basicConfig(
    level = logging.INFO,
    format= logging_str,
    handlers=[
        logging.FileHandler(log_filepath), # FileHandler outputs the logging message inside the log file. 
        logging.StreamHandler(sys.stdout) #StreamHandler outputs the logging message in the terminal.
    ]
)

logger = logging.getLogger("mlProjectLogger")