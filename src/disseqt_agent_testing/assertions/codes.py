"""The reason-code registry. Gating reads a code's *class*, never a hard-coded list.

Classes: ``check_failure`` (a real defect), ``benign_skip`` (nothing to check),
``unverifiable`` (could not measure: fails closed), ``instrument_error`` (the check itself broke)
and ``uncalibrated`` (a judge not yet trusted to gate).

Registered codes (kept in sync with ``_seed`` by a test; ``<x>:*`` is a family):

    code                         class             meaning
    answer.forbidden_text        check_failure     forbidden text present in the answer
    answer.missing_text          check_failure     expected text missing from the answer
    answer.no_output             check_failure     the agent produced no answer
    blocked_by:*                 benign_skip       not checked: blocked by another failure
    budget.exceeded              check_failure     a budget was exceeded
    clarify.acted_before_asking  check_failure     the agent acted before asking
    clarify.irrelevant_question  check_failure     the question does not target missing information
    clarify.not_asked            check_failure     the agent did not ask a clarifying question
    clarify.slot_missing         check_failure     the question does not mention the missing information
    clarify.too_many_questions   check_failure     the agent asked too many questions
    data_unavailable             unverifiable      data needed for this check was not recorded
    decision.empty               check_failure     the agent produced no output
    decision.mismatch            check_failure     decision differs from the expected one
    decision.turn_missing        check_failure     the expected turn did not happen
    forbidden.called             check_failure     a forbidden call happened
    harness.not_finished         check_failure     agent run did not finish
    harness.violation            check_failure     harness violation
    intent.mismatch              check_failure     wrong intent
    intent.unresolved            check_failure     intent could not be determined
    judge_failed                 instrument_error  the judge failed
    judge_unavailable            unverifiable      needs a judge and none is configured
    judge_uncalibrated           uncalibrated      the judge is not calibrated yet
    judge_unsure                 unverifiable      the judge was unsure
    loop.consecutive             check_failure     too many consecutive identical calls
    loop.cycle                   check_failure     the agent repeated a cycle of calls
    loop.identical               check_failure     too many identical calls
    metric_error                 instrument_error  the check raised an exception
    missing_required             check_failure     required argument {path} is missing
    not_applicable               benign_skip       nothing to check
    not_configured               benign_skip       not configured
    order.edge_same_batch        check_failure     ordered calls ran in the same batch
    order.edge_unverifiable      check_failure     an ordering could not be checked: a call is missing
    order.edge_violated          check_failure     a required ordering was violated
    order.exact                  check_failure     calls do not match the exact sequence
    order.same_step              check_failure     calls were not in the same step
    order.sequence               check_failure     calls happened in the wrong order
    plan.constraint_failed       check_failure     a plan constraint failed
    plan_quality.flawed          check_failure     the plan judge found flaws in the plan
    answer_quality.no_output     check_failure     the agent produced no answer to judge
    answer_quality.below_bar     check_failure     too few answer-quality criteria were met
    groundedness.no_output       check_failure     the agent produced no answer to check
    groundedness.unsupported     check_failure     the answer has claims no source supports
    groundedness.contradicted    check_failure     the answer contradicts its sources
    predicate_error              instrument_error  a predicate raised or returned a bad value
    ref_unresolvable             instrument_error  a Ref could not be resolved
    refuse.no_alternative        check_failure     the refusal offers no alternative
    refuse.not_refused           check_failure     the agent did not refuse
    refuse.phrase_missing        check_failure     the refusal does not mention the expected phrase
    refuse.unexpected_alternative check_failure     the refusal offers an alternative
    route.invalid_target         check_failure     route to a target that was not offered
    route.mismatch               check_failure     wrong route
    route.unexpected             check_failure     unexpected route
    scope.over_refusal           check_failure     refused an in-scope request
    scope.under_refusal          check_failure     did not refuse an out-of-scope request
    threshold.below              check_failure     a score is below its min_scores threshold
    tool.hallucinated            check_failure     a call to a tool that was not offered
    tool.missing                 check_failure     a required call did not happen
    tool.not_allowed             check_failure     a call outside allowed_tools
    tool.unexpected              check_failure     an unexpected call happened
    type_error:*                 check_failure     {path} has the wrong type
    unexpected_param             check_failure     unexpected argument {path}
    ungrounded_value             check_failure     {path} does not come from the expected source
    value_error:*                check_failure     {path} has the wrong value
    wrong_count                  check_failure     wrong number of calls
"""

import threading
from dataclasses import dataclass

CHECK_FAILURE = "check_failure"
BENIGN_SKIP = "benign_skip"
UNVERIFIABLE = "unverifiable"
INSTRUMENT_ERROR = "instrument_error"
UNCALIBRATED = "uncalibrated"
SEVERITY_CLASSES = (CHECK_FAILURE, BENIGN_SKIP, UNVERIFIABLE, INSTRUMENT_ERROR, UNCALIBRATED)

_LOCK = threading.RLock()


@dataclass(slots=True, frozen=True)
class CodeSpec:
    code: str
    severity_class: str
    template: str


_CODES: dict[str, CodeSpec] = {}


def register_code(
    code: str, severity_class: str, template: str, *, override: bool = False
) -> CodeSpec:
    """Register a reason code. Duplicates raise unless ``override=True``."""
    if severity_class not in SEVERITY_CLASSES:
        raise ValueError(f"severity_class must be one of {list(SEVERITY_CLASSES)}")
    if not code or not isinstance(code, str):
        raise ValueError("code must be a non-empty string")
    spec = CodeSpec(code, severity_class, template)
    with _LOCK:
        if code in _CODES and not override:
            raise ValueError(f"reason code {code!r} is already registered")
        _CODES[code] = spec
    return spec


def unregister_code(code: str) -> None:
    with _LOCK:
        _CODES.pop(code, None)


def lookup(code: str) -> CodeSpec | None:
    """Exact match, else the family before ``:`` (``blocked_by:run_status`` -> ``blocked_by``)."""
    with _LOCK:
        spec = _CODES.get(code)
        if spec is None and ":" in code:
            spec = _CODES.get(code.split(":", 1)[0] + ":*")
        if spec is None and "." in code:  # dotted family, e.g. "myplugin.*"
            spec = _CODES.get(code.split(".", 1)[0] + ".*")
        return spec


def code_class(code: str | None) -> str:
    """Unknown codes fail closed: they are treated as ``unverifiable`` when skipped."""
    if code is None:
        return CHECK_FAILURE
    spec = lookup(code)
    return spec.severity_class if spec else UNVERIFIABLE


def known_codes() -> tuple[str, ...]:
    with _LOCK:
        return tuple(sorted(_CODES))


def _seed() -> None:
    f, b, u, i = CHECK_FAILURE, BENIGN_SKIP, UNVERIFIABLE, INSTRUMENT_ERROR
    rows = [
        # argument codes
        ("missing_required", f, "required argument {path} is missing"),
        ("unexpected_param", f, "unexpected argument {path}"),
        ("type_error:*", f, "{path} has the wrong type"),
        ("value_error:*", f, "{path} has the wrong value"),
        ("ungrounded_value", f, "{path} does not come from the expected source"),
        ("wrong_count", f, "wrong number of calls"),
        # check codes
        ("harness.not_finished", f, "agent run did not finish"),
        ("harness.violation", f, "harness violation"),
        ("decision.mismatch", f, "decision differs from the expected one"),
        ("decision.empty", f, "the agent produced no output"),
        ("decision.turn_missing", f, "the expected turn did not happen"),
        ("tool.missing", f, "a required call did not happen"),
        ("tool.unexpected", f, "an unexpected call happened"),
        ("tool.hallucinated", f, "a call to a tool that was not offered"),
        ("tool.not_allowed", f, "a call outside allowed_tools"),
        ("order.sequence", f, "calls happened in the wrong order"),
        ("order.exact", f, "calls do not match the exact sequence"),
        ("order.edge_violated", f, "a required ordering was violated"),
        ("order.edge_same_batch", f, "ordered calls ran in the same batch"),
        ("order.edge_unverifiable", f, "an ordering could not be checked: a call is missing"),
        ("order.same_step", f, "calls were not in the same step"),
        ("forbidden.called", f, "a forbidden call happened"),
        ("route.mismatch", f, "wrong route"),
        ("route.unexpected", f, "unexpected route"),
        ("route.invalid_target", f, "route to a target that was not offered"),
        ("loop.identical", f, "too many identical calls"),
        ("loop.consecutive", f, "too many consecutive identical calls"),
        ("loop.cycle", f, "the agent repeated a cycle of calls"),
        ("budget.exceeded", f, "a budget was exceeded"),
        ("threshold.below", f, "a score is below its min_scores threshold"),
        ("clarify.not_asked", f, "the agent did not ask a clarifying question"),
        ("clarify.acted_before_asking", f, "the agent acted before asking"),
        ("clarify.slot_missing", f, "the question does not mention the missing information"),
        ("clarify.too_many_questions", f, "the agent asked too many questions"),
        ("clarify.irrelevant_question", f, "the question does not target missing information"),
        ("refuse.not_refused", f, "the agent did not refuse"),
        ("refuse.phrase_missing", f, "the refusal does not mention the expected phrase"),
        ("refuse.no_alternative", f, "the refusal offers no alternative"),
        ("refuse.unexpected_alternative", f, "the refusal offers an alternative"),
        ("scope.over_refusal", f, "refused an in-scope request"),
        ("scope.under_refusal", f, "did not refuse an out-of-scope request"),
        ("intent.mismatch", f, "wrong intent"),
        ("intent.unresolved", f, "intent could not be determined"),
        ("answer.no_output", f, "the agent produced no answer"),
        ("answer.missing_text", f, "expected text missing from the answer"),
        ("answer.forbidden_text", f, "forbidden text present in the answer"),
        ("plan.constraint_failed", f, "a plan constraint failed"),
        ("plan_quality.flawed", f, "the plan judge found flaws in the plan"),
        ("answer_quality.no_output", f, "the agent produced no answer to judge"),
        ("answer_quality.below_bar", f, "too few answer-quality criteria were met"),
        ("groundedness.no_output", f, "the agent produced no answer to check"),
        ("groundedness.unsupported", f, "the answer has claims no source supports"),
        ("groundedness.contradicted", f, "the answer contradicts its sources"),
        # skip reasons
        ("not_applicable", b, "nothing to check"),
        ("not_configured", b, "not configured"),
        ("blocked_by:*", b, "not checked: blocked by another failure"),
        ("data_unavailable", u, "data needed for this check was not recorded"),
        ("judge_unavailable", u, "needs a judge and none is configured"),
        ("judge_unsure", u, "the judge was unsure"),
        ("judge_uncalibrated", UNCALIBRATED, "the judge is not calibrated yet"),
        # error reasons
        ("metric_error", i, "the check raised an exception"),
        ("predicate_error", i, "a predicate raised or returned a bad value"),
        ("judge_failed", i, "the judge failed"),
        ("ref_unresolvable", i, "a Ref could not be resolved"),
    ]
    for code, severity, template in rows:
        register_code(code, severity, template, override=True)


_seed()
