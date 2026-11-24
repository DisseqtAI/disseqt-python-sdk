"""xdist support: registered only when xdist is present. Aggregation happens only on the
controller (from the reports it receives); workers get the resolved seed."""

from typing import Any


class XdistSupport:
    def __init__(self, impl: Any):
        self.impl = impl

    def pytest_configure_node(self, node: Any) -> None:
        node.workerinput["disseqt_seed"] = self.impl.state.seed

    def pytest_testnodedown(self, node: Any, error: Any) -> None:
        if error is not None:
            self.impl.state.warned.add(f"node-down:{getattr(node, 'gateway', None)}")
        from ..runner.models import CollectionErrorRecord
        from .plugin import WORKER_ERRORS_KEY

        # every worker collects every file, so the same record arrives once per worker
        known = self.impl.state.collection_errors
        for data in (getattr(node, "workeroutput", None) or {}).get(WORKER_ERRORS_KEY, ()):
            record = CollectionErrorRecord.from_dict(data)
            if record not in known:
                known.append(record)
