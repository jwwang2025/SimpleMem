"""
Run-level logging for the LoCoMo / LoCoMo-Plus evaluation.

Design (single-writer)
----------------------
- Stdlib ``logging`` only (no extra dependency).
- Every run writes TWO files under ``<out_dir>/logs``:
    * ``run_YYYYmmdd_HHMMSS.log`` - immutable per-run log
    * ``latest.log``               - overwritten, always the newest run
- A Tee replaces ``sys.stdout`` / ``sys.stderr`` and is the ONLY writer of the
  log files: it forwards every byte to the original console AND appends a
  cleaned line to both log files. The logging StreamHandler is attached to the
  stdout Tee, so logger output and SimpleMem's internal ``print`` output share
  one file handle each - lines never interleave/overwrite and appear exactly
  once per sink.
- Uncaught exceptions are logged with a full traceback before crashing.
- UTF-8 on-disk encoding for Windows safety; every write is flushed immediately.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

_orig_stdout = sys.stdout
_orig_stderr = sys.stderr

_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_EXCEPTHOOK_SET = False


class _Tee:
    """
    Write-through mirror: console behaviour is untouched (raw writes, including
    tqdm's ``\\r``), while the log files receive a cleaned, non-empty line.
    This object is the single owner of the log-file handles.
    """

    def __init__(self, original, log_paths):
        self._original = original
        # truncate/create this run's files once
        self._logs = [open(Path(p), "w", encoding="utf-8", errors="replace", buffering=1)
                      for p in log_paths]

    def write(self, data):
        # console first, unchanged
        try:
            self._original.write(data)
        except Exception:
            pass
        # mirror a cleaned version to every log file
        text = str(data).replace("\r", "").rstrip()
        if text:
            for log_file in self._logs:
                log_file.write(text + "\n")
                log_file.flush()

    def flush(self):
        for stream in [self._original] + self._logs:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return self._original.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._original.fileno()

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._original, "errors", "replace")


def setup_run_logging(
    out_dir,
    level: str = "INFO",
    run_tag: Optional[str] = None,
) -> Tuple[logging.Logger, dict]:
    """
    Configure logging for one evaluation run.

    Returns (logger, {"run_log": Path, "latest_log": Path, "logs_dir": Path}).
    """
    global _EXCEPTHOOK_SET
    out_dir = Path(out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_tag}" if run_tag else ""
    run_log = logs_dir / f"run_{stamp}{suffix}.log"
    latest_log = logs_dir / "latest.log"
    log_paths = (run_log, latest_log)

    # Install fresh Tees (always wrap the import-time original streams, so a
    # second setup() in the same process never nests Tees).
    sys.stdout = _Tee(_orig_stdout, log_paths)
    sys.stderr = _Tee(_orig_stderr, log_paths)

    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    # Logger -> stdout Tee -> console + both log files (single file writer).
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    root.addHandler(handler)

    if not _EXCEPTHOOK_SET:
        def _excepthook(exc_type, exc, tb):
            logging.getLogger("eval").error(
                "Uncaught exception", exc_info=(exc_type, exc, tb)
            )

        sys.excepthook = _excepthook
        _EXCEPTHOOK_SET = True

    logger = logging.getLogger("eval")
    logger.info("Logging initialised | run log: %s", run_log)
    logger.info("Latest log pointer  : %s", latest_log)
    return logger, {
        "run_log": str(run_log),
        "latest_log": str(latest_log),
        "logs_dir": str(logs_dir),
    }


def teardown_run_logging() -> None:
    """Close mirrored log handles and restore the import-time streams."""
    for name, original in (("stdout", _orig_stdout), ("stderr", _orig_stderr)):
        stream = getattr(sys, name)
        for log_file in getattr(stream, "_logs", []):
            try:
                log_file.close()
            except Exception:
                pass
        setattr(sys, name, original)
