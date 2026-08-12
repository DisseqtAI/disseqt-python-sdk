"""Version of the installed ``disseqt-ai-sdk`` distribution.

Both sub-packages ship in the single ``disseqt-ai-sdk`` distribution, so
the installed metadata is the one source of truth. This mirrors
``disseqt_sdk._version`` (duplicated, not imported — the two sub-packages
stay independent) and both resolve the same distribution, so they cannot
drift.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    SDK_VERSION = version("disseqt-ai-sdk")
except PackageNotFoundError:  # running from a source checkout
    SDK_VERSION = "0.0.0-dev"
