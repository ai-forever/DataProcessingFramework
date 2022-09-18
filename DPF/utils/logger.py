import logging
import logging.config
import os
import sys

LOGGERS_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '[%(asctime)s][%(levelname)s]: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': logging.DEBUG,
            'stream': sys.stdout
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'default',
            'filename': '',
            'level': logging.INFO,
        }
    },
    'loggers': {
        'template': {
            'handlers': ['console', 'file'],
            'level': logging.DEBUG
        },
    }
}


def init_logger(filename, logger_name='filter_logger', logging_dir='./logs/'):
    os.makedirs(logging_dir, exist_ok=True)
    
    LOGGERS_CONFIG['handlers']['file']['filename'] = os.path.join(logging_dir, filename)
    LOGGERS_CONFIG['loggers'][logger_name] = {
        'handlers': ['console', 'file'],
        'level': logging.DEBUG
    }
    
    logging.config.dictConfig(LOGGERS_CONFIG)
    logger = logging.getLogger(logger_name)
    logger.info(f'Logger initialized')
    
    return logger