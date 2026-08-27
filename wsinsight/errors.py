"""Exceptions used in WSInsight."""

from __future__ import annotations

import os
import traceback

TRACEBACK_ENV_VAR = "WSINSIGHT_TRACEBACK"


def traceback_enabled() -> bool:
    """Whether ``WSINSIGHT_TRACEBACK`` is set, requesting full tracebacks.

    Presence alone enables it, so ``WSINSIGHT_TRACEBACK=`` also counts.
    """
    return TRACEBACK_ENV_VAR in os.environ


def format_cli_error(exc: BaseException) -> str:
    """Render a top-level failure: full traceback, or just the message."""
    if traceback_enabled():
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return f"WSInsight failed. Traceback:\n{tb}"

    message = str(exc).strip()
    detail = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    return (
        f"WSInsight failed. Error message:\n{detail}\n\n"
        f"(set {TRACEBACK_ENV_VAR}=1 for the full traceback)"
    )


class WsinferException(Exception):
    """Base class for wsinsight exceptions."""


class UnknownArchitectureError(WsinferException):
    """Architecture is unknown and cannot be found."""


class WholeSlideImageDirectoryNotFound(WsinferException, FileNotFoundError):
    ...


class DuplicateFilePrefixesFound(WsinferException):
    """A duplicate file prefix has been found.

    An example of duplicate file prefixes is files a.svs and a.tif. WSInsight relies on
    the stems as a unique ID, so we cannot allow duplicate stems.
    """


class WholeSlideImagesNotFound(WsinferException, FileNotFoundError):
    ...


class ResultsDirectoryNotFound(WsinferException, FileNotFoundError):
    ...


class PatchDirectoryNotFound(WsinferException, FileNotFoundError):
    ...


class CannotReadSpacing(WsinferException):
    ...


class NoBackendException(WsinferException):
    ...


class BackendNotAvailable(WsinferException):
    """The requested backend is not available."""
