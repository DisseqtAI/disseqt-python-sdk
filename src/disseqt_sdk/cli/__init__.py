"""``disseqt`` command-line interface.

The click group `cli` is the entry point wired in pyproject.toml's
`[project.scripts]`. Sub-commands live in sibling modules and are
attached below to keep the top-level shell trivial to reason about.
"""

from __future__ import annotations

import click

from .._version import SDK_VERSION
from .policy import policy as policy_group
from .redteam import redteam as redteam_group
from .run import run as run_cmd
from .scan import scan as scan_cmd
from .validate import validate as validate_cmd


@click.group(name="disseqt")
@click.version_option(version=SDK_VERSION, prog_name="disseqt")
def cli() -> None:
    """Disseqt AI SDK command-line interface.

    Runtime validation, red-team, policy, and scan tools. Auth flows
    from environment variables — see ``disseqt validate --help`` for
    the required set.
    """


cli.add_command(validate_cmd)
cli.add_command(run_cmd)
cli.add_command(redteam_group)
cli.add_command(policy_group)
cli.add_command(scan_cmd)


if __name__ == "__main__":  # pragma: no cover
    cli()
