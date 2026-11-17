"""Block outbound network access for every agent-testing test (plan 02, T46).

The harness is offline by design, so a test that reaches for the network is a bug. Unix sockets
and ``socketpair`` stay allowed (``asyncio`` uses them internally).
"""

import socket

import pytest


class NetworkBlockedError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def refuse(address: object) -> None:
        raise NetworkBlockedError(
            f"network access is blocked in tests/agent_testing (tried to connect to {address!r})"
        )

    def guarded_connect(self: socket.socket, address: object) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            refuse(address)
        return real_connect(self, address)  # type: ignore[arg-type]

    def guarded_connect_ex(self: socket.socket, address: object) -> int:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            refuse(address)
        return real_connect_ex(self, address)  # type: ignore[arg-type]

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object) -> None:
        refuse(host)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
