"""T46: the suite runs with sockets blocked (see ../conftest.py); prove the block works."""

import socket

import pytest

from tests.agent_testing.conftest import NetworkBlockedError

from .helpers import refund


def test_outbound_connections_fail() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("example.com", 80), timeout=1)
    with pytest.raises(NetworkBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("93.184.216.34", 80))
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("example.com", 80)


def test_local_socketpair_and_asyncio_still_work() -> None:
    a, b = socket.socketpair()
    a.sendall(b"x")
    assert b.recv(1) == b"x"
    a.close(), b.close()
    import asyncio

    asyncio.run(asyncio.sleep(0))


def test_a_full_harness_run_needs_no_network() -> None:
    assert refund().run("refund order A12").status == "finished"
