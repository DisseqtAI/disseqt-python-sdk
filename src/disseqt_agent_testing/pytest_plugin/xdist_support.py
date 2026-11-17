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
