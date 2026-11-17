"""Small shims so the plugin works on pytest 8 and 9."""

from pathlib import Path
from typing import Any

import pytest

PYTEST_MAJOR = int(pytest.__version__.split(".")[0])


def node_path(node: Any) -> Path:
    return Path(str(getattr(node, "path", None) or node.fspath))


def has_xdist(config: pytest.Config) -> bool:
    """xdist registers as ``xdist`` when auto-loaded and as ``xdist.plugin`` with ``-p``; its
    ``--numprocesses`` option is the name-independent signal."""
    pm = config.pluginmanager
    return bool(
        pm.hasplugin("xdist")
        or pm.hasplugin("xdist.plugin")
        or hasattr(config.option, "numprocesses")
    )


def is_worker(config: pytest.Config) -> bool:
    """True inside an xdist worker; the controller (and a plain run) is not a worker."""
    return hasattr(config, "workerinput")
