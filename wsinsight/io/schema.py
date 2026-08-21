"""Column-prefix conventions for object-CSV registration.

This module owns the single source of truth for the ``region_*`` and
``object_*`` column-name conventions emitted by ``wsinsight reg``.  It
validates user-supplied tags, builds bare prefixes, resolves auto-bumped
suffixes for no-tag collisions (Word "Untitled-1" semantics), and
discovers every probability-prefix group present in a set of primary
CSVs for downstream GeoJSON / OME-CSV export.
"""

from __future__ import annotations

import re
from typing import Iterable
from typing import Sequence

import pandas as pd

__all__ = [
    "make_region_prefix",
    "make_object_prefix",
    "resolve_no_tag_prefix",
    "discover_prob_prefixes",
]

# A tag, when supplied, is a non-empty sequence of lower-case letters,
# digits and underscores.  The empty string is also accepted and yields
# the bare default prefix.
_TAG_RE = re.compile(r"^[a-z0-9_]+$")


def _validate_tag(tag: str) -> None:
    if tag == "":
        return
    if not _TAG_RE.match(tag):
        raise ValueError(
            f"--tag must match [a-z0-9_]+ (got {tag!r}); "
            "lower-case letters, digits and underscores only."
        )


def make_region_prefix(tag: str) -> str:
    """Return the bare ``region`` column prefix (always trailing ``_``).

    ``tag=""`` → ``"region_"`` (back-compat default).
    ``tag="foo"`` → ``"region_foo_"``.
    """
    _validate_tag(tag)
    return "region_" if tag == "" else f"region_{tag}_"


def make_object_prefix(tag: str) -> str:
    """Return the bare ``object`` column prefix (always trailing ``_``).

    ``tag=""`` → ``"object_"``.
    ``tag="foo"`` → ``"object_foo_"``.
    """
    _validate_tag(tag)
    return "object_" if tag == "" else f"object_{tag}_"


# Pattern for the auto-bump scan: matches ``<kind>_prob_<class>`` (no integer
# suffix → bump-key 0) and ``<kind>_<int>_prob_<class>`` (bump-key <int>).
_BUMP_RE_TEMPLATE = r"^{kind}(?:_(\d+))?_prob_[^_].*$"


def resolve_no_tag_prefix(kind: str, existing_cols: Iterable[str]) -> str:
    """Pick the bare prefix to use for a no-tag invocation against existing columns.

    Scans *existing_cols* for column names matching ``<kind>_prob_*`` (bump
    key 0) or ``<kind>_<N>_prob_*`` for any positive integer ``N``.  If no
    such columns exist, returns ``"<kind>_"``.  Otherwise returns
    ``"<kind>_<M>_"`` where ``M`` is the **smallest free positive
    integer** (gap-fill).

    Word "Untitled-1" semantics: a no-tag invocation never overwrites an
    existing namespace; it lands on the next free slot.
    """
    if kind not in ("region", "object"):
        raise ValueError(f"kind must be 'region' or 'object' (got {kind!r})")

    pat = re.compile(_BUMP_RE_TEMPLATE.format(kind=kind))
    used_keys: set[int] = set()
    for col in existing_cols:
        m = pat.match(col)
        if m is None:
            continue
        used_keys.add(int(m.group(1)) if m.group(1) is not None else 0)

    if not used_keys:
        return f"{kind}_"
    # Smallest free positive integer (gap-fill); 0 is the bare default.
    n = 1
    while n in used_keys:
        n += 1
    return f"{kind}_{n}_"


# Pattern used by export discovery.  Groups the leading
# ``prob`` / ``region(_<tag>)?_prob`` / ``object(_<tag>)?_prob`` token,
# which is the GeoJSON / OME-CSV "prefix" string.
_GROUP_RE = re.compile(
    r"^(prob|region(?:_[a-z0-9_]+?)?_prob|object(?:_[a-z0-9_]+?)?_prob)_[^_].*$"
)


def discover_prob_prefixes(csv_paths: Sequence) -> list[str]:
    """Enumerate every ``prob`` / ``region_*_prob`` / ``object_*_prob`` group
    present in the headers of *csv_paths*.

    Each CSV is opened with ``nrows=0`` so only the header is read.  The
    return value is a sorted list of distinct prefix strings (without
    trailing underscore), suitable for passing to
    ``write_geojsons(..., prefix=p)`` or ``write_omecsvs(..., prefix=p)``.
    """
    prefixes: set[str] = set()
    for p in csv_paths:
        try:
            cols = pd.read_csv(p, nrows=0).columns
        except Exception:
            continue
        for c in cols:
            m = _GROUP_RE.match(str(c))
            if m is not None:
                prefixes.add(m.group(1))
    return sorted(prefixes)
