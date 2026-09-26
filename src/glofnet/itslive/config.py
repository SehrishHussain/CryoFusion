"""
Configuration for the ITS_LIVE data pipeline.
"""

# =============================================================================
# Glacier Selection
# =============================================================================

GLACIER_ID = "RGI2000-v7.0-G-14-08488"

STAC_URL = "https://stac.itslive.cloud"

COLLECTION_NAME = "itslive-cubes"



# =============================================================================
# Output
# =============================================================================

OUTPUT_DIRECTORY = "data/raw/itslive"

COLLECTION_NAME = "itslive-cubes"

START_DATE = "2024-01-01"
END_DATE = "2026-12-31"


MAX_ITEMS = 20


ITSLIVE_VARIABLES = [
    "mapping",
    "img_pair_info",
    "vx",
    "vy",
    "v",
    "v_error",
     
    "interp_mask",
]