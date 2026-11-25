import os
import logging.config
from importlib import resources
from typing import Any, Optional



version_file_path = resources.files("h_adminsim").joinpath("version.txt")
LOGGING_NAME = f"H-AdminSim_{open(version_file_path).read().strip()}"
VERBOSE = True


def set_logging(name: str = LOGGING_NAME, verbose: bool = True, log_file: Optional[str] = None) -> None:
    """
    Configure the named logger with a colored console handler and an optional file handler.

    Args:
        name (str, optional): Logger name. Defaults to LOGGING_NAME.
        verbose (bool, optional): Log at INFO (on rank -1/0) instead of ERROR. Defaults to True.
        log_file (Optional[str], optional): Path of a log file to append to as well. Defaults to None.
    """
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR

    class ColorFormatter(logging.Formatter):
        """Custom formatter to add colors to log messages using colorstr."""
        def format(self, record: logging.LogRecord) -> str:
            """
            Color the message by level, then format the record.

            Args:
                record (logging.LogRecord): The log record.

            Returns:
                str: The formatted message.
            """
            if record.levelname == "ERROR":
                record.msg = colorstr("red", record.msg)
            elif record.levelname == "WARNING":
                record.msg = colorstr("yellow", record.msg)
            # elif record.levelname == "INFO":
            #     record.msg = colorstr("green", record.msg)
            elif record.levelname == "DEBUG":
                record.msg = colorstr("blue", record.msg)
            return super().format(record)

    handlers = {
        name: {
            'class': 'logging.StreamHandler',
            'formatter': name,
            'level': level
        }
    }

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers[f'{name}_file'] = {
            'class': 'logging.FileHandler',
            'formatter': name,
            'level': level,
            'filename': log_file,
            'mode': 'a',
            'encoding': 'utf-8'
        }

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                '()': ColorFormatter,  # Use the custom formatter
                'format': '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
            }
        },
        'handlers': handlers,
        'loggers': {
            name: {
                'level': level,
                'handlers': list(handlers.keys()),
                'propagate': False
            }
        }
    })
    

set_logging(LOGGING_NAME, verbose=VERBOSE) 
LOGGER = logging.getLogger(LOGGING_NAME)



def colorstr(*input: Any) -> str:
    """
    Wrap a string in ANSI color/style codes, e.g. ``colorstr('red', 'bold', 'text')``.
    With a single argument, 'blue' and 'bold' are applied.

    Args:
        *input (Any): Color/style names ('red', 'bright_blue', 'bold', 'underline', ...) followed by the value to color.

    Returns:
        str: The colored string, terminated by the reset code.
    """
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']



def log(message: Any, level: str = 'info', color: bool = False) -> None:
    """
    Log a message through the package logger.

    Args:
        message (Any): Message to log.
        level (str, optional): 'warning', 'error', or anything else for INFO. Defaults to 'info'.
        color (bool, optional): Color INFO messages with ``colorstr``. Defaults to False.
    """
    if level.lower() == 'warning':
        LOGGER.warning(message)
    elif level.lower() == 'error':
        LOGGER.error(message)
    else:
        if color:
            LOGGER.info(colorstr(message))
        else:
            LOGGER.info(message)



class Information:
    """Attribute container that also supports ``obj[key]`` and ``key in obj``."""
    def __init__(self, **kwargs: Any):
        """
        Store the keyword arguments as attributes.

        Args:
            **kwargs (Any): Attribute names and values.
        """
        self.__dict__.update(kwargs)

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]
    
    def update(self, **kwargs: Any) -> 'Information':
        """
        Set or overwrite attributes.

        Args:
            **kwargs (Any): Attribute names and values.

        Returns:
            Information: This object.
        """
        self.__dict__.update(kwargs)
        return self
    
    def __contains__(self, key: str) -> bool:
        return key in self.__dict__
