"""Agentic unit testing: run an agent offline with fake tools and check what it did.

Sub-packages: ``models`` (frozen test-case, run and verdict data, re-exported here), ``harness``
(``build_test_orchestrator``, fakes, ``FakeLLM``), ``assertions`` (``assert_agent``, ``evaluate``,
the metric registry), ``judges`` (offline LLM-judge engine and calibration), ``runner``
(``run_case``, ``run_suite``, ``SuiteResult``), ``pytest_plugin`` (the ``pytest11`` entry point)
and ``cli`` (the ``disseqt`` command). Nothing here is imported by ``disseqt_sdk`` or
``disseqt_agentic_sdk``.
"""

from .models import *  # noqa: F401,F403
from .models import __all__  # noqa: F401
