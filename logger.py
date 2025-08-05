import os
from datetime import datetime

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Logger:
    """
    A utility class for standardized logging throughout the application.
    Supports different log levels, colors, and can write to both console and file.
    """
    
    # Log levels
    LEVELS = {
        "DEBUG": 0,
        "INFO": 1,
        "SUCCESS": 2,
        "WARNING": 3,
        "ERROR": 4
    }
    
    def __init__(self, name="CRBot", device_serial=None, log_level="DEBUG", log_to_file=False, log_dir="models/logs"):
        """
        Initialize a new Logger instance.
        
        Args:
            name (str): The name of the logger/module
            device_serial (str, optional): The serial number of the device (for multi-device setups)
            log_level (str): Minimum level of logs to display ("DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR")
            log_to_file (bool): Whether to also write logs to a file
            log_dir (str): Directory to store log files if log_to_file is True
        """
        self.name = name
        self.device_serial = device_serial
        self.log_level = self.LEVELS.get(log_level.upper(), 1)  # Default to INFO
        self.log_to_file = log_to_file
        self.log_file = None
        
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            device_suffix = f"_{device_serial}" if device_serial else ""
            self.log_file = os.path.join(log_dir, f"{name}{device_suffix}_{timestamp}.log")
    
    def _format_message(self, level, message):
        """Format a log message with timestamp, level, name and device info."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        device_info = f"[{self.device_serial}]" if self.device_serial else ""
        return f"{timestamp} {level} [{self.name}]{device_info}: {message}"
    
    def _write_to_file(self, formatted_message):
        """Write a log message to file if enabled."""
        if self.log_to_file and self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(formatted_message + "\n")
    
    def debug(self, message):
        """Log a debug message (lowest priority)."""
        if self.log_level <= self.LEVELS["DEBUG"]:
            formatted = self._format_message("DEBUG", message)
            print(f"{bcolors.OKBLUE}{formatted}{bcolors.ENDC}")
            self._write_to_file(formatted)
    
    def info(self, message):
        """Log an informational message."""
        if self.log_level <= self.LEVELS["INFO"]:
            formatted = self._format_message("INFO", message)
            print(formatted)
            self._write_to_file(formatted)
    
    def success(self, message):
        """Log a success message."""
        if self.log_level <= self.LEVELS["SUCCESS"]:
            formatted = self._format_message("SUCCESS", message)
            print(f"{bcolors.OKGREEN}{formatted}{bcolors.ENDC}")
            self._write_to_file(formatted)
    
    def warning(self, message):
        """Log a warning message."""
        if self.log_level <= self.LEVELS["WARNING"]:
            formatted = self._format_message("WARNING", message)
            print(f"{bcolors.WARNING}{formatted}{bcolors.ENDC}")
            self._write_to_file(formatted)
    
    def error(self, message):
        """Log an error message (highest priority)."""
        if self.log_level <= self.LEVELS["ERROR"]:
            formatted = self._format_message("ERROR", message)
            print(f"{bcolors.FAIL}{formatted}{bcolors.ENDC}")
            self._write_to_file(formatted)
    
    def set_level(self, level):
        """Change the log level."""
        self.log_level = self.LEVELS.get(level.upper(), self.log_level)
    
    def set_device_serial(self, device_serial):
        """Update the device serial number."""
        self.device_serial = device_serial


# Create a default logger instance for easy import
default_logger = Logger()

# Helper functions for quick access
def debug(message):
    default_logger.debug(message)

def info(message):
    default_logger.info(message)

def success(message):
    default_logger.success(message)

def warning(message):
    default_logger.warning(message)

def error(message):
    default_logger.error(message)
