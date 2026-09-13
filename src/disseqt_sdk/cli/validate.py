"""``disseqt validate`` — one-shot policy evaluation (Phase 0a)."""

from __future__ import annotations

import click

from ..models.input_validation import InputValidationRequest
from ..policy import BlockedError, any_blocking
from ._common import build_client, echo_json, print_error, resolve_input


@click.command("validate")
@click.option("--input", "input_arg", help="The input text to evaluate.")
@click.option(
    "--input-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Read the input from a file (use '-' for stdin).",
)
@click.option(
    "--policy",
    "policies",
    multiple=True,
    required=True,
    help="Published policy id to evaluate against. Repeat for multiple policies.",
)
@click.option(
    "--response",
    help="Optional model response text (evaluates policies that inspect the response).",
)
@click.option(
    "--context",
    help="Optional RAG context text.",
)
@click.option(
    "--application-name",
    help=(
        "Override the application_name reported to the Decisions ledger. "
        "Falls back to $DISSEQT_APPLICATION_NAME then 'disseqt-cli'."
    ),
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the full result envelope as JSON. Without this flag, only exit code is set.",
)
def validate(
    input_arg: str | None,
    input_file: str | None,
    policies: tuple[str, ...],
    response: str | None,
    context: str | None,
    application_name: str | None,
    as_json: bool,
) -> None:
    """Evaluate INPUT against one or more realtime policies.

    Exit code is 0 on PASS, 1 on BLOCK, 2 on usage/config error.
    """
    prompt = resolve_input(input_arg, input_file)
    client = build_client(application_name=application_name)
    request = InputValidationRequest(
        prompt=prompt,
        context=context,
        response=response,
    )
    try:
        result = client.validate(request, policies=list(policies))
    except Exception as exc:  # noqa: BLE001 — surface any HTTP/ValueError
        print_error(exc)
        raise SystemExit(2) from exc
    if as_json:
        echo_json(result)
    blocked = any_blocking(result)
    if blocked:
        click.secho("BLOCK", fg="red", err=True)
        raise SystemExit(1)
    click.secho("PASS", fg="green", err=True)


__all__ = ["validate", "BlockedError"]
