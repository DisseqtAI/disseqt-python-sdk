"""``disseqt`` command-line interface.

The click group `cli` is the entry point wired in pyproject.toml's
`[project.scripts]`. Sub-commands live in sibling modules and are
attached below to keep the top-level shell trivial to reason about.
"""

from __future__ import annotations

import click

from .._version import SDK_VERSION
from . import resources as _resources
from .login import login as login_cmd
from .login import logout as logout_cmd
from .policy import policy as policy_group
from .redteam import redteam as redteam_group
from .scan import scan as scan_cmd
from .whoami import whoami as whoami_cmd

byov_group = _resources.byov
mcp_target_group = _resources.mcp_target
output_validation_group = _resources.output_validation
pack_group = _resources.pack
plan_group = _resources.plan
plan_run_group = _resources.plan_run
pp_run_group = _resources.pp_run
rag_target_group = _resources.rag_target
rag_validation_group = _resources.rag_validation
session_group = _resources.session
target_group = _resources.target
vulnerability_group = _resources.vulnerability


@click.group(name="disseqt")
@click.version_option(version=SDK_VERSION, prog_name="disseqt")
def cli() -> None:
    """Disseqt AI SDK command-line interface.

    Red-team, GRC policy, prompt-pack, and scan tools. Auth flows from
    environment variables — see ``disseqt login --help`` for the
    required set.
    """


cli.add_command(redteam_group)
cli.add_command(policy_group)
cli.add_command(scan_cmd)
cli.add_command(target_group)
cli.add_command(rag_target_group)
cli.add_command(mcp_target_group)
cli.add_command(pack_group)
cli.add_command(pp_run_group)
cli.add_command(output_validation_group)
cli.add_command(rag_validation_group)
cli.add_command(session_group)
cli.add_command(byov_group)
cli.add_command(vulnerability_group)
cli.add_command(whoami_cmd)
cli.add_command(plan_group)
cli.add_command(plan_run_group)
cli.add_command(login_cmd)
cli.add_command(logout_cmd)


if __name__ == "__main__":  # pragma: no cover
    cli()
