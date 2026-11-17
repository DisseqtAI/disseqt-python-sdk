"""The pytest plugin (``pytest11`` target): collects ``*.agent.yaml|yml|json`` case files as one
item per case and provides fixtures for Python-defined cases.

Silent when unused: with no agent cases and no ``--disseqt-*`` option it prints nothing and
imports only light modules. Disable it with ``-p no:disseqt``.
"""

from .fixtures import (
    disseqt_arun_case,
    disseqt_config,
    disseqt_judge,
    disseqt_limiter,
    disseqt_orchestrator,
    disseqt_run_case,
)
from .nodes import AgentCaseFailed, AgentCaseIncomplete
from .parametrize import parametrize_cases
from .plugin import (
    pytest_addhooks,
    pytest_addoption,
    pytest_collect_file,
    pytest_collection_modifyitems,
    pytest_configure,
)

__all__ = [
    "AgentCaseFailed", "AgentCaseIncomplete", "disseqt_arun_case", "disseqt_config",
    "disseqt_judge", "disseqt_limiter", "disseqt_orchestrator", "disseqt_run_case",
    "parametrize_cases", "pytest_addhooks", "pytest_addoption", "pytest_collect_file",
    "pytest_collection_modifyitems", "pytest_configure",
]  # fmt: skip
