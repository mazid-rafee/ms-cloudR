"""Shared SEN12MS-CR season discovery, ROI grouping, and path helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

from src.utils.io_utils import map_seasons


SEASON_ALIASES = {
    "winter": "ROIs2017_winter",
    "spring": "ROIs1158_spring",
    "summer": "ROIs1868_summer",
    "fall": "ROIs1970_fall",
    "autumn": "ROIs1970_fall",
}

SEASON_SHORT = {
    "ROIs2017_winter": "winter",
    "ROIs1158_spring": "spring",
    "ROIs1868_summer": "summer",
    "ROIs1970_fall": "fall",
}

BAND_ORDER = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
B10_INDEX = 10

# Official SEN12MS-CR loader uses 1-based GDAL band indices; B10=cirrus=11
# which is 0-based channel index 10 after rasterio.read() of all bands.
OFFICIAL_BAND_ORDER_NOTE = (
    "data/SEN12MS-CR/sen12ms_cr_dataLoader.py S2Bands.ALL = "
    "[B01..B09, B10, B11, B12] with B10=cirrus=11 (1-based). "
    "SEN12MSCRDataset reads all 13 channels via rasterio in file order, "
    "yielding 0-based index 10 = B10."
)


def resolve_season_tokens(seasons) -> list[str]:
    """Resolve CLI season tokens to dataset folder prefixes (without _s2)."""
    if seasons is None:
        tokens = ["winter", "spring", "summer", "fall"]
    elif isinstance(seasons, str):
        tokens = [s.strip() for s in seasons.replace(",", " ").split() if s.strip()]
    else:
        tokens = []
        for item in seasons:
            tokens.extend(
                [s.strip() for s in str(item).replace(",", " ").split() if s.strip()]
            )
    resolved = []
    for token in tokens:
        key = token.strip()
        if key in SEASON_ALIASES:
            resolved.append(SEASON_ALIASES[key])
        elif key in SEASON_SHORT:
            resolved.append(key)
        else:
            # Allow already-prefixed names or map_seasons fallback.
            mapped = map_seasons(key)
            if mapped:
                resolved.append(mapped[0])
            else:
                raise ValueError(f"Unknown season token: {token}")
    # Preserve order, drop duplicates.
    out = []
    seen = set()
    for season in resolved:
        if season not in seen:
            out.append(season)
            seen.add(season)
    return out


def season_short_name(season_prefix: str) -> str:
    return SEASON_SHORT.get(season_prefix, season_prefix)


def discover_season_dirs(data_dir: str | Path, seasons: list[str]) -> dict:
    """Verify required modality folders exist for each requested season."""
    root = Path(data_dir)
    result = {}
    missing = []
    for season in seasons:
        dirs = {
            "s2_cloudy": root / f"{season}_s2_cloudy",
            "s2": root / f"{season}_s2",
            "s1": root / f"{season}_s1",
        }
        absent = [name for name, path in dirs.items() if not path.is_dir()]
        result[season] = {
            "short_name": season_short_name(season),
            "dirs": {k: str(v) for k, v in dirs.items()},
            "available": len(absent) == 0,
            "missing": absent,
        }
        if absent:
            missing.append((season, absent))
    return {"root": str(root), "seasons": result, "missing": missing}


def parse_roi_and_patch(cloudy_path: str, season_prefix: str) -> dict:
    """Parse ROI / patch identifiers from a cloudy sample path.

    Example:
      .../ROIs2017_winter_s2_cloudy/s2_cloudy_102/ROIs2017_winter_s2_cloudy_102_p100.tif
    -> roi_id=102, patch_id=p100, roi_dir=s2_cloudy_102
    """
    path = Path(cloudy_path)
    fname = path.name
    parent = path.parent.name
    rel_to_season = None
    # ROI directory: s2_cloudy_<id>
    roi_match = re.match(r"^(?:s2_cloudy_|s2_|s1_)(\d+)$", parent)
    roi_id = roi_match.group(1) if roi_match else parent
    patch_match = re.search(r"_(p\d+)\.(?:tif|tiff)$", fname, flags=re.IGNORECASE)
    patch_id = patch_match.group(1) if patch_match else Path(fname).stem
    # Cross-season geographic identity is NOT reliably detectable: ROI numeric
    # IDs overlap across campaigns (ROIs1158 vs ROIs2017) without a shared atlas.
    # Therefore the split group key is season-scoped.
    short = season_short_name(season_prefix)
    group_key = f"{short}:{roi_id}"
    return {
        "season": short,
        "season_prefix": season_prefix,
        "roi_dir": parent,
        "roi_id": str(roi_id),
        "patch_id": patch_id,
        "filename": fname,
        "group_key": group_key,
        "canonical_geographic_group": None,
        "cross_season_geo_identity_reliable": False,
    }


def relative_cloudy_path(cloudy_path: str, data_dir: str | Path) -> str:
    try:
        return os.path.relpath(cloudy_path, str(data_dir))
    except ValueError:
        return cloudy_path
