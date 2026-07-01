"""Backend selection and utilities for reading whole-slide imagery metadata."""

from __future__ import annotations

import logging
import os
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import tifffile
from PIL import Image

from .errors import BackendNotAvailable
from .errors import CannotReadSpacing
from .errors import DuplicateFilePrefixesFound
from .errors import NoBackendException
from .uri_path import URIPath

logger = logging.getLogger(__name__)

# Vendor-typical mapping from Aperio AppMag (nominal objective magnification)
# to microns-per-pixel. Used as a last-resort fallback when no MPP metadata
# can be read. Values are approximate and assume Aperio/Leica scanners; a
# warning is emitted whenever this fallback is hit so users can verify
# against expected resolution.
_MPP_FROM_APPMAG: dict[int, float] = {
    40: 0.25,
    20: 0.50,
    10: 1.00,
    4:  2.50,
}

_allowed_backends = {"openslide", "tiffslide"}

try:
    import openslide

    # Test that OpenSlide object exists. If it doesn't, an error will be thrown and
    # caught. For some reason, it is possible that openslide-python can be installed
    # but the OpenSlide object (and other openslide things) are not available.
    openslide.OpenSlide  # noqa: B018
    HAS_OPENSLIDE = True
    logger.debug("Imported openslide")
except Exception as err:
    HAS_OPENSLIDE = False
    logger.debug(f"Unable to import openslide due to error: {err}")

try:
    import tiffslide

    HAS_TIFFSLIDE = True
    logger.debug("Imported tiffslide")
except Exception as err:
    HAS_TIFFSLIDE = False
    logger.debug(f"Unable to import tiffslide due to error: {err}")

if not HAS_TIFFSLIDE and not HAS_OPENSLIDE:
    raise NoBackendException(
        "No backend is available. Please install openslide or tiffslide."
    )

# Backend selection: an explicit ``WSINSIGHT_WSI_BACKEND`` env var wins;
# otherwise prefer OpenSlide when its bindings are importable (richer
# Aperio metadata, including ``aperio.AppMag`` used by the MPP fallback);
# fall back to TiffSlide if OpenSlide isn't installed.
_override = os.getenv("WSINSIGHT_WSI_BACKEND", "").strip().lower()
if _override in _allowed_backends:
    _BACKEND: str = _override
elif HAS_OPENSLIDE:
    _BACKEND = "openslide"
else:
    _BACKEND = "tiffslide"


def set_backend(name: str) -> None:
    """Configure whether WSInsight uses `openslide` or `tiffslide` backends."""
    global _BACKEND
    if name not in _allowed_backends:
        raise ValueError(f"Unknown backend: '{name}'")
    if name == "openslide" and not HAS_OPENSLIDE:
        raise BackendNotAvailable(
            "OpenSlide is not available. Please install the OpenSlide compiled"
            " library and the Python package 'openslide-python'."
            " See https://openslide.org/ for more information."
        )
    elif name == "tiffslide":
        if not HAS_TIFFSLIDE:
            raise BackendNotAvailable(
                "TiffSlide is not available. Please install 'tiffslide'."
            )

    logger.debug(f"Set backend to {name}")

    _BACKEND = name


def get_wsi_cls() -> type[openslide.OpenSlide] | type[tiffslide.TiffSlide]:
    """Return the active whole-slide reader class for the selected backend."""
    if _BACKEND not in _allowed_backends:
        raise ValueError(
            f"Unknown backend: '{_BACKEND}'. Please contact the developer!"
        )
    if _BACKEND == "openslide":
        return openslide.OpenSlide  # type: ignore
    elif _BACKEND == "tiffslide":
        return tiffslide.TiffSlide
    else:
        raise ValueError("Contact the developer, slide backend not known")


# Set the slide backend based on the environment.
# Prioritize TiffSlide if the user has it installed.
# if HAS_TIFFSLIDE:
#     set_backend("tiffslide")
# elif HAS_OPENSLIDE:
#     set_backend("openslide")
# else:
#     raise NoBackendException("No backend found! Please install openslide or tiffslide")


# For typing an object that has a method `read_region`.
class CanReadRegion(Protocol):
    """Protocol describing objects that expose ``read_region`` like OpenSlide."""
    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        pass


def _get_mpp_openslide(slide_path: str | Path) -> tuple[float, float]:
    """Read MPP using OpenSlide.

    Parameters
    ----------
    slide_path : str or Path
        The path to the whole slide image.

    Returns
    -------
    mppx, mppy
        Two floats representing the micrometers per pixel in x and y dimensions.

    Raises
    ------
    CannotReadSpacing if spacing cannot be read from the whole slide iamge.
    """
    logger.debug("Attempting to read MPP using OpenSlide")
    if not HAS_OPENSLIDE:
        logger.critical(
            "Cannot read MPP with OpenSlide because OpenSlide is not available"
        )
        raise CannotReadSpacing()
    slide = openslide.OpenSlide(slide_path)
    mppx: float | None = None
    mppy: float | None = None

    if (
        openslide.PROPERTY_NAME_MPP_X in slide.properties
        and openslide.PROPERTY_NAME_MPP_Y in slide.properties
    ):
        logger.debug(
            "Properties of the OpenSlide object contains keys"
            f" {openslide.PROPERTY_NAME_MPP_X} and {openslide.PROPERTY_NAME_MPP_Y}"
        )
        mppx = slide.properties[openslide.PROPERTY_NAME_MPP_X]
        mppy = slide.properties[openslide.PROPERTY_NAME_MPP_Y]
        logger.debug(
            f"Value of {openslide.PROPERTY_NAME_MPP_X} is {mppx} and value"
            f" of {openslide.PROPERTY_NAME_MPP_Y} is {mppy}"
        )
        if mppx is not None and mppy is not None:
            try:
                logger.debug("Attempting to convert these MPP strings to floats")
                mppx = float(mppx)
                mppy = float(mppy)
                return mppx, mppy
            except Exception as err:
                logger.debug(f"Exception caught while converting to float: {err}")
    elif (
        "tiff.ResolutionUnit" in slide.properties
        and "tiff.XResolution" in slide.properties
        and "tiff.YResolution" in slide.properties
    ):
        logger.debug("Attempting to read spacing using openslide and tiff tags")
        resunit = slide.properties["tiff.ResolutionUnit"].lower()
        if resunit not in {"millimeter", "centimeter", "cm", "inch"}:
            raise CannotReadSpacing(f"unknown resolution unit: '{resunit}'")
        scale = {
            "inch": 25400.0,
            "centimeter": 10000.0,
            "cm": 10000.0,
            "millimeter": 1000.0,
        }.get(resunit, None)

        x_resolution = float(slide.properties["tiff.XResolution"])
        y_resolution = float(slide.properties["tiff.YResolution"])

        if scale is not None:
            try:
                mpp_x = scale / x_resolution
                mpp_y = scale / y_resolution
                return mpp_x, mpp_y
            except ArithmeticError as err:
                raise CannotReadSpacing(
                    f"error in math {scale} / {x_resolution}"
                    f" or {scale} / {y_resolution}"
                ) from err
        else:
            raise CannotReadSpacing()

    else:
        logger.debug(
            "Properties of the OpenSlide object does not contain keys"
            f" {openslide.PROPERTY_NAME_MPP_X} and {openslide.PROPERTY_NAME_MPP_Y}"
        )
    raise CannotReadSpacing()


def _get_mpp_tiffslide(
    slide_path: str | Path,
) -> tuple[float, float]:
    """Read MPP using TiffSlide."""
    logger.debug("Attempting to read MPP using TiffSlide")

    if not HAS_TIFFSLIDE:
        logger.critical(
            "Cannot read MPP with TiffSlide because TiffSlide is not available"
        )
        raise CannotReadSpacing()

    slide = tiffslide.TiffSlide(slide_path)
    mppx: float | None = None
    mppy: float | None = None
    if (
        tiffslide.PROPERTY_NAME_MPP_X in slide.properties
        and tiffslide.PROPERTY_NAME_MPP_Y in slide.properties
    ):
        mppx = slide.properties[tiffslide.PROPERTY_NAME_MPP_X]
        mppy = slide.properties[tiffslide.PROPERTY_NAME_MPP_Y]
        if mppx is None or mppy is None:
            raise CannotReadSpacing()
        else:
            try:
                mppx = float(mppx)
                mppy = float(mppy)
                return mppx, mppy
            except Exception as err:
                raise CannotReadSpacing() from err
    raise CannotReadSpacing()


# Modified from
# https://github.com/bayer-science-for-a-better-life/tiffslide/blob/8bea5a4c8e1429071ade6d4c40169ce153786d19/tiffslide/tiffslide.py#L712-L745
def _get_mpp_tifffile(slide_path: str | Path) -> tuple[float, float]:
    """Read MPP using Tifffile."""
    logger.debug("Attempting to read MPP using tifffile")
    with tifffile.TiffFile(slide_path) as tif:
        series0 = tif.series[0]
        page0 = series0[0]
        if not isinstance(page0, tifffile.TiffPage):
            raise CannotReadSpacing("not a tifffile.TiffPage instance")
        try:
            resolution_unit = page0.tags["ResolutionUnit"].value
            x_resolution = Fraction(*page0.tags["XResolution"].value)
            y_resolution = Fraction(*page0.tags["YResolution"].value)
        except KeyError as err:
            raise CannotReadSpacing() from err

        # tifffile moved the RESUNIT enum to the module level in 2022.7.28 and
        # removed ``tifffile.TIFF.RESUNIT`` in newer releases; prefer the
        # module-level name and fall back to the legacy location.
        RESUNIT = getattr(tifffile, "RESUNIT", None) or tifffile.TIFF.RESUNIT
        scale = {
            RESUNIT.INCH: 25400.0,
            RESUNIT.CENTIMETER: 10000.0,
            RESUNIT.MILLIMETER: 1000.0,
            RESUNIT.MICROMETER: 1.0,
            RESUNIT.NONE: None,
        }.get(resolution_unit, None)
        if scale is not None:
            try:
                mpp_x = scale / x_resolution
                mpp_y = scale / y_resolution
                return mpp_x, mpp_y
            except ArithmeticError as err:
                raise CannotReadSpacing() from err
    raise CannotReadSpacing()


def _get_appmag_openslide(slide_path: str | Path) -> float | None:
    """Return Aperio nominal magnification via OpenSlide, or ``None``.

    Reads ``aperio.AppMag`` (Aperio-specific) and falls back to OpenSlide's
    generic ``openslide.objective-power`` property. All errors are swallowed
    and reported as ``None`` so this can be used as a non-fatal fallback.
    """
    if not HAS_OPENSLIDE:
        return None
    try:
        slide = openslide.OpenSlide(slide_path)
        v = slide.properties.get("aperio.AppMag") or slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER
        )
        return float(v) if v is not None else None
    except Exception as err:
        logger.debug(f"OpenSlide AppMag read failed for {slide_path}: {err}")
        return None


def _get_appmag_tiffslide(slide_path: str | Path) -> float | None:
    """Return Aperio nominal magnification via TiffSlide, or ``None``."""
    if not HAS_TIFFSLIDE:
        return None
    try:
        slide = tiffslide.TiffSlide(slide_path)
        v = slide.properties.get("aperio.AppMag") or slide.properties.get(
            "tiffslide.objective-power"
        )
        return float(v) if v is not None else None
    except Exception as err:
        logger.debug(f"TiffSlide AppMag read failed for {slide_path}: {err}")
        return None


def get_avg_mpp(slide_path: Path | str, default_mpp: float | None = None) -> float:
    """Return the average MPP of a whole slide image.

    The value is in units of micrometers per pixel and is
    the average of the X and Y dimensions.

    ``default_mpp`` is a user-supplied fallback (um/px) used **only** when the
    spacing cannot be read from the slide metadata. Slide metadata is always
    preferred; the fallback exists for slides that carry no MPP at all.

    Raises
    ------
    CannotReadSpacing if the spacing cannot be read and no ``default_mpp`` is given.
    """

    mppx: float
    mppy: float

    if _BACKEND == "openslide":
        try:
            mppx, mppy = _get_mpp_openslide(slide_path)
            return (mppx + mppy) / 2
        except CannotReadSpacing:
            pass
    if _BACKEND == "tiffslide":
        try:
            mppx, mppy = _get_mpp_tiffslide(slide_path)
            return (mppx + mppy) / 2
        except CannotReadSpacing:
            pass

    logger.debug(f"Failed to read MPP using {_BACKEND}.")
    logger.debug("Trying to read MPP with tifffile as last resort.")

    # If tiffslide/openslide don't work, try tifffile.
    try:
        mppx, mppy = _get_mpp_tifffile(slide_path)
        return (mppx + mppy) / 2
    except CannotReadSpacing:
        pass

    # Final fallback: infer MPP from Aperio nominal magnification (AppMag).
    # Many TCGA SVS files have AppMag in their ImageDescription even when MPP
    # tags are missing or unreadable. Vendor-typical values are used; a
    # warning is logged so users can verify against the expected resolution.
    appmag = _get_appmag_openslide(slide_path) or _get_appmag_tiffslide(slide_path)
    if appmag is not None:
        key = int(round(appmag))
        if key in _MPP_FROM_APPMAG:
            mpp = _MPP_FROM_APPMAG[key]
            logger.warning(
                "%s: MPP missing — falling back to AppMag=%g (assumed %.3f um/px). "
                "Verify this matches your scanner.",
                str(slide_path), appmag, mpp,
            )
            return mpp
        logger.warning(
            "%s: MPP missing and AppMag=%g not in fallback table %s.",
            str(slide_path), appmag, sorted(_MPP_FROM_APPMAG),
        )

    # User-supplied fallback: used only when nothing could be read from the slide.
    if default_mpp is not None and default_mpp > 0:
        logger.warning(
            "%s: MPP could not be read from the slide; using the supplied "
            "--spacing-um-px=%g um/px fallback. Verify this matches the scan.",
            str(slide_path), default_mpp,
        )
        return float(default_mpp)

    raise CannotReadSpacing(slide_path)


def _validate_wsi_directory(wsi_dir: str | Path) -> None:
    """Validate that slide stems are unique within ``wsi_dir``."""
    wsi_dir = URIPath(wsi_dir)
    maybe_slides = [p for p in wsi_dir.iterdir() if wsi_dir.scheme == "image-list" or p.is_file()]
    uniq_stems = set(p.stem for p in maybe_slides)
    if len(uniq_stems) != len(maybe_slides):
        raise DuplicateFilePrefixesFound(
            "A slide with the same prefix but different extensions has been found"
            " (like slide.svs and slide.tif). Slides must have unique prefixes."
        )
