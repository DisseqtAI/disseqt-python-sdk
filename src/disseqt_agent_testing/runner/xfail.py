"""``run: {xfail: reason}``: an expected failure is settled the same way in every mode."""

import dataclasses

from .models import CaseResult

EXPECTED_FAILURE = "expected_failure"
UNEXPECTED_PASS = "unexpected_pass"


def settle_xfail(result: CaseResult, reason: str | None) -> CaseResult:
    """A failed case marked xfail becomes ``xfailed`` (green); one that passes becomes ``failed``
    (strict, so a fixed bug cannot stay marked). Anything else (incomplete, error, skipped) is
    unchanged: a verdict that could not be measured is never an expected failure."""
    if not reason or result.reason_code in (EXPECTED_FAILURE, UNEXPECTED_PASS):
        return result
    if result.status == "failed":
        return dataclasses.replace(
            result, status="xfailed", reason_code=EXPECTED_FAILURE, reason=f"expected failure: {reason}"
        )  # fmt: skip
    if result.status == "passed":
        return dataclasses.replace(
            result, status="failed", reason_code=UNEXPECTED_PASS,
            reason=f"agent case '{result.case_id}' FAILED: it is marked xfail ({reason}) but passed; "
            "remove the xfail marker",
        )  # fmt: skip
    return result
