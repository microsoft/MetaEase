import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional
import atexit
import threading

class PrintCapture:
    """Capture print statements and redirect them to logger."""
    def __init__(self, logger):
        self.logger = logger
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def write(self, text):
        if text.strip():  # Only log non-empty lines
            self.logger.info(text.rstrip())
        self.original_stdout.write(text)

    def flush(self):
        self.original_stdout.flush()

    def restore(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


class AsyncLogger:
    _instance: Optional["AsyncLogger"] = None
    _logger: Optional[logging.Logger] = None
    _print_capture: Optional[PrintCapture] = None

    def __init__(self, log_dir: str, verbose: bool = False):
        if AsyncLogger._instance is not None:
            raise RuntimeError(
                "AsyncLogger is a singleton. Use AsyncLogger.init() to create and AsyncLogger.get() to retrieve."
            )

        self.log_dir = log_dir
        self.verbose = verbose
        self.log_file = os.path.join(log_dir, "experiment.log")
        os.makedirs(log_dir, exist_ok=True)

        # Initialize the logger
        self._setup_logger()

    def _setup_logger(self):
        """Setup the logger with thread-safe configuration."""
        logger = logging.getLogger('ExperimentLogger')
        logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers
        if logger.handlers:
            return

        # Create handlers
        # File handler with rotation
        # Ensure log file exists immediately (even if empty) so it's created even if process crashes early
        # The directory is already created in __init__, but ensure the file exists
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'w') as f:
                    f.write('')  # Create empty file
            except (OSError, IOError):
                pass  # If we can't create it, the handler will try later

        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            delay=False  # Changed to False so file is opened immediately when handler is created
        )
        file_handler.setLevel(logging.DEBUG)

        # Console handler (enabled if verbose flag is True)
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '[%(asctime)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Create formatters and add it to the handlers
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(threadName)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(file_formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)

        # Store logger instance
        self._logger = logger

        # Initialize print capture
        self._print_capture = PrintCapture(self._logger)

    @classmethod
    def init(cls, log_dir: str, verbose: bool = False) -> "AsyncLogger":
        """Initialize the singleton logger instance."""
        if cls._instance is None:
            cls._instance = cls(log_dir, verbose)
            # Register cleanup on exit
            atexit.register(cls._instance.shutdown)
        return cls._instance

    @classmethod
    def get(cls) -> "AsyncLogger":
        """Get the singleton logger instance."""
        if cls._instance is None:
            raise RuntimeError("Logger not initialized. Call AsyncLogger.init() first.")
        return cls._instance

    def shutdown(self):
        """Gracefully shutdown the logger."""
        if self._print_capture:
            self._print_capture.restore()
        if self._logger:
            for handler in self._logger.handlers[:]:
                handler.close()
                self._logger.removeHandler(handler)

    def log_klee_time(self, time_taken: float) -> None:
        """Log the time taken by KLEE execution."""
        self._logger.info(f"KLEE Execution Time: {time_taken:.3f} seconds")

    def log_max_gap(self, max_gap: float, num_inputs: int) -> None:
        """Log the maximum gap found from KLEE inputs."""
        if max_gap is not None:
            self._logger.info(f"Max Gap: {max_gap:.3f} (from {num_inputs} inputs)")
        else:
            self._logger.info(f"Max Gap: None (from {num_inputs} inputs)")

    def log_current_max_gap(self, max_gap: float) -> None:
        """Log the current max gap."""
        self._logger.info(f"Current max global gap: {max_gap:.3f}")

    def log_final_max_gap(self, max_gap: float) -> None:
        """Log the final max gap."""
        self._logger.info(f"Final max global gap: {max_gap:.3f}")

    def log_entry(self, entry: str) -> None:
        """Log an entry to the log file."""
        self._logger.info(entry)

    def log(self, message: str, level: str = "info") -> None:
        """Generic log method that supports different log levels."""
        log_method = getattr(self._logger, level.lower(), self._logger.info)
        log_method(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self._logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self._logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self._logger.critical(message)

    def set_verbose(self, verbose: bool) -> None:
        """Dynamically enable or disable console output."""
        if self.verbose == verbose:
            return  # No change needed
        
        self.verbose = verbose
        
        # Remove existing console handlers
        for handler in self._logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.handlers.RotatingFileHandler):
                self._logger.removeHandler(handler)
        
        # Add console handler if verbose is True
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '[%(asctime)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self._logger.addHandler(console_handler)

    def capture_prints(self, enable: bool = True) -> None:
        """Enable or disable capturing of print statements."""
        if enable and self._print_capture:
            sys.stdout = self._print_capture
            sys.stderr = self._print_capture
        elif not enable and self._print_capture:
            self._print_capture.restore()

    def restore_prints(self) -> None:
        """Restore original stdout/stderr."""
        if self._print_capture:
            self._print_capture.restore()
