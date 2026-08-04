import logging
import os
import json
from logging.handlers import RotatingFileHandler

LOGS_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "agent_runtime.jsonl")

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "module": f"{record.module}:{record.lineno}",
            "message": record.getMessage()
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record, ensure_ascii=False)

def get_logger(name="LegalAgent"):
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        console_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(module)s | %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.propagate = False

    return logger

logger = get_logger()
