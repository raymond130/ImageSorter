import logging
import logging.handlers
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "imagesorter.log"
LOGGER_NAME = "imagesorter"


def setup_logging() -> logging.Logger:
    """
    Configure the "imagesorter" logger once per process and return it.

    Everything (DEBUG and up) goes to logs/imagesorter.log, which rotates at
    1 MB and keeps 3 old files. INFO and up also goes to the console. Safe to
    call on every Streamlit rerun: handlers are only added the first time.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # avoid duplicate lines via Streamlit's root logger

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    LOG_DIR.mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    logger.info("Logging started (file: %s)", LOG_FILE)
    return logger


def read_log_tail(max_lines: int = 200) -> str:
    """Return the last max_lines lines of the current log file ('' if none)."""
    try:
        with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-max_lines:])
    except OSError:
        return ""
