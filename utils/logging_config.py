"""
Structured logging configuration for RAG Legal Search.

Provides JSON-formatted structured logging with:
- Correlation IDs for request tracing
- Performance timing decorators
- Contextual log enrichment
- Configurable log levels and outputs
"""

import logging
import json
import os
import sys
import time
import uuid
import functools
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Optional

# Context variable for request correlation
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class StructuredFormatter(logging.Formatter):
    """
    JSON structured log formatter.
    
    Outputs each log record as a single JSON line with standardized fields
    for easy ingestion by log aggregation tools (ELK, Datadog, Loki, etc.).
    """
    
    LEVEL_MAP = {
        "DEBUG": "debug",
        "INFO": "info",
        "WARNING": "warn",
        "ERROR": "error",
        "CRITICAL": "fatal",
    }
    
    def __init__(self, service_name: str = "rag-legal-search"):
        super().__init__()
        self.service_name = service_name
        self.hostname = os.environ.get("HOSTNAME", "localhost")
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": self.LEVEL_MAP.get(record.levelname, record.levelname.lower()),
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "host": self.hostname,
        }
        
        # Add correlation ID if present
        cid = correlation_id.get("")
        if cid:
            log_entry["correlation_id"] = cid
        
        # Add exception info
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["error"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "stacktrace": self.formatException(record.exc_info),
            }
        
        # Add extra fields from record
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__
            and k not in ("message", "msg", "args")
        }
        if extras:
            log_entry["extra"] = extras
        
        return json.dumps(log_entry, default=str)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable log formatter for local development.
    
    Uses color-coded output with structured context appended.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        cid = correlation_id.get("")
        cid_str = f" [{cid[:8]}]" if cid else ""
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        msg = (
            f"{color}{timestamp} "
            f"{record.levelname:<8}{self.RESET}"
            f"{cid_str} "
            f"{record.name}: {record.getMessage()}"
        )
        
        if record.exc_info and record.exc_info[0] is not None:
            msg += f"\n{self.formatException(record.exc_info)}"
        
        return msg


def setup_logging(
    level: str = "INFO",
    json_output: bool = True,
    service_name: str = "rag-legal-search",
) -> logging.Logger:
    """
    Configure application-wide structured logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Use JSON format (True) or human-readable (False)
        service_name: Service name for log entries
    
    Returns:
        Configured root logger for the application
    """
    log_level = os.environ.get("LOG_LEVEL", level).upper()
    use_json = os.environ.get("LOG_FORMAT", "json" if json_output else "text") == "json"
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    if use_json:
        handler.setFormatter(StructuredFormatter(service_name))
    else:
        handler.setFormatter(HumanReadableFormatter())
    
    # Configure root logger
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Reduce noise from third-party libraries
    for lib in ("urllib3", "chromadb", "httpcore", "httpx", "openai"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    logger = logging.getLogger(service_name)
    logger.info(
        "Logging initialized",
        extra={"log_level": log_level, "format": "json" if use_json else "text"},
    )
    return logger


def new_correlation_id() -> str:
    """Generate and set a new correlation ID for the current context."""
    cid = uuid.uuid4().hex
    correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get current correlation ID, creating one if none exists."""
    cid = correlation_id.get("")
    if not cid:
        cid = new_correlation_id()
    return cid


def timed(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    metric_name: Optional[str] = None,
) -> Callable:
    """
    Decorator that logs function execution time.
    
    Args:
        logger: Logger to use (defaults to function's module logger)
        level: Log level for the timing message
        metric_name: Optional metric name for metrics collection
    
    Example:
        @timed(metric_name="search_latency")
        def search(query: str) -> list:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logger or logging.getLogger(func.__module__)
            func_name = f"{func.__qualname__}"
            
            _logger.log(level, f"Starting {func_name}")
            start = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                _logger.log(
                    level,
                    f"Completed {func_name}",
                    extra={
                        "duration_ms": round(elapsed_ms, 2),
                        "metric": metric_name or func_name,
                    },
                )
                
                # Record metric if collector is available
                try:
                    from utils.metrics import get_collector
                    collector = get_collector()
                    collector.observe_histogram(
                        metric_name or f"{func_name}_duration_ms",
                        elapsed_ms,
                    )
                except ImportError:
                    pass
                
                return result
                
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                _logger.error(
                    f"Failed {func_name}",
                    extra={
                        "duration_ms": round(elapsed_ms, 2),
                        "error_type": type(exc).__name__,
                    },
                    exc_info=True,
                )
                raise
        
        return wrapper
    return decorator
