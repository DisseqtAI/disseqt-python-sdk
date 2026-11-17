"""The turn classifier: label each turn ``call | ask | refuse | answer | empty``.

Fixed precedence (plan 03 section 11): declared decision, write call, handoff, empty output,
structured output, then lexicons over the cleaned text. Heuristic by nature: low confidence is
reported, and a judge may be asked when one is configured.
"""

import re
import unicodedata
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any

from ..models import AgentRun, Step
from .config import EvalConfig
from .judge import JudgeTask, interpret_verdict
from .lexicons import LexiconPack, get_lexicon

if TYPE_CHECKING:
    from .metrics.base import EvalContext

LABELS = ("call", "ask", "refuse", "answer")
_FENCE = re.compile(r"```.*?```", re.S)
_INLINE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"https?://\S+")
_QUOTED = re.compile(r"^\s*>.*$", re.M)
_SENTENCE = re.compile(r"[^.!?。！？؟\n]*[.!?。！？؟]+|[^.!?。！？؟\n]+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'"})
SUBSTANTIVE_WORDS = 40


@dataclass(slots=True, frozen=True)
class Classification:
    label: str
    confidence: float
    rule: str
    evidence: tuple[str, ...] = ()
    judged: bool = False
    judge_status: str | None = None  # "judge_failed" / "judge_unsure" when escalation failed
    low_confidence: bool = False
    advisory_label: str | None = None  # an uncalibrated judge's opinion (never used to decide)


def clean_text(text: str) -> str:
    """Strip code, quotes and URLs, then normalize (NFKC, straight apostrophes)."""
    text = _FENCE.sub(" ", text)
    text = _INLINE.sub(" ", text)
    text = _QUOTED.sub(" ", text)
    text = _URL.sub(" ", text)
    return unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).strip()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]


def turn_output(run: AgentRun, turn: int) -> Any:
    if turn < len(run.turn_outputs):
        return run.turn_outputs[turn]
    if turn == 0 and not run.turn_outputs:
        return run.final_answer if run.final_answer is not None else run.final_text
    return None


def turn_calls(run: AgentRun, turn: int) -> list[Step]:
    return [s for s in run.steps if s.kind in ("tool", "subagent") and s.turn == turn]


def turn_count(run: AgentRun) -> int:
    last_step = max((s.turn for s in run.steps), default=-1)
    return max(len(run.turn_outputs), last_step + 1, 1)


def _is_handoff(step: Step, cfg: EvalConfig) -> bool:
    if step.kind == "subagent" or step.name == "transfer_to_agent":
        return True
    return any(fnmatchcase(step.name, p) for p in cfg.route_patterns)


def is_refusal(text: str, pack: LexiconPack, cfg: EvalConfig, raw: str | None = None) -> str | None:
    """Return the matched phrase or ``None``. Only the opening (two sentences / 200 chars) counts;
    an apology followed by a lot of substantive text, or by code, is not a refusal."""
    head_sents = sentences(text)[:2]
    head = " ".join(head_sents)[:200].lower()
    lowered = text.lower()
    extra = tuple(p.lower() for p in cfg.lexicons.refusal_extra)
    excluded = tuple(p.lower() for p in cfg.lexicons.refusal_exclude)
    if any(p in lowered for p in excluded):
        return None
    hit: str | None = None
    for prefix in (*pack.refusal_prefixes, *extra):
        if head.startswith(prefix):
            hit = prefix
            break
    if hit is None:
        for prefix in pack.modal_prefixes:
            if head.startswith(prefix):
                rest = head[len(prefix) :].split()
                if rest and rest[0].strip(".,;:!?") in pack.action_verbs:
                    hit = prefix
                    break
    if hit is None:
        for pattern in pack.refusal_patterns:
            match = re.search(pattern, head)
            if match:
                hit = match.group()
                break
    if hit is None:
        return None
    if raw is not None and "```" in raw:
        return None  # substantive content (code) follows
    first = head_sents[0] if head_sents else ""
    rest_words = len(text.split()) - len(first.split())
    if rest_words > SUBSTANTIVE_WORDS:
        return None
    return hit


def _wrapped_inverted(sentence: str) -> bool:
    return sentence.startswith("¿") and sentence.rstrip().endswith("?")


def is_clarifying_question(text: str, pack: LexiconPack, cfg: EvalConfig) -> str | None:
    """Return the question sentence or ``None``. A closing pleasantry is not a question; a
    question buried early in a long answer is not clarifying."""
    sents = sentences(text)
    if not sents:
        return None
    closers = (*pack.closers, *(c.lower() for c in cfg.lexicons.closers_extra))
    cues = (*pack.question_cues, *(c.lower() for c in cfg.lexicons.question_cues_extra))
    last_two = sents[-2:]
    short = not any(len(s.split()) > 12 for s in sents)
    for sentence in sents:
        low = sentence.lower().strip()
        is_q = (
            low.endswith(("?", "？", "؟"))
            or _wrapped_inverted(sentence)
            or any(cue in low for cue in cues)
        )
        if not is_q:
            continue
        if any(closer in low for closer in closers):
            continue
        if sentence in last_two or short:
            return sentence
    return None


def classify_turn(
    run: AgentRun, turn: int, cfg: EvalConfig, ctx: "EvalContext | None" = None
) -> Classification:
    output = turn_output(run, turn)
    calls = turn_calls(run, turn)
    pack = get_lexicon(cfg.lang)
    # 1: a declared decision
    if cfg.decision_source and isinstance(output, dict):
        parts = cfg.decision_source.split(".")
        node: Any = output
        for part in parts[1:] if parts and parts[0] == "answer" else parts:
            node = node.get(part) if isinstance(node, dict) else None
        if node in LABELS:
            return Classification(node, 1.0, "declared", (cfg.decision_source,))
    # 2: any write-effect call
    writes = [s for s in calls if s.kind == "tool" and s.effect == "write"]
    if writes:
        return Classification("call", 1.0, "write_call", tuple(s.name for s in writes))
    # 3: a handoff
    handoffs = [s for s in calls if _is_handoff(s, cfg)]
    if handoffs:
        return Classification("call", 1.0, "handoff", tuple(s.name for s in handoffs))
    # 4: empty output
    if output is None or (isinstance(output, str) and not output.strip()):
        if calls:
            return Classification("call", 0.9, "read_calls_no_text", tuple(s.name for s in calls))
        return Classification("empty", 1.0, "empty_output")
    # 5: structured output is never lexicon-classified
    if not isinstance(output, str):
        if calls:
            return Classification("call", 0.9, "read_calls_structured")
        return Classification("answer", 0.9, "structured_answer")
    # 6: lexicons over the cleaned text
    text = clean_text(output)
    refusal = is_refusal(text, pack, cfg, raw=output)
    question = is_clarifying_question(text, pack, cfg)
    if refusal and question:
        result = Classification(
            "refuse", 0.6, "refusal_and_question", (refusal, question), low_confidence=True
        )
    elif refusal:
        result = Classification("refuse", 0.85, "refusal_phrase", (refusal,))
    elif question:
        result = Classification("ask", 0.85, "clarifying_question", (question,))
    elif calls:
        result = Classification("call", 0.9, "read_calls_no_question", tuple(s.name for s in calls))
    else:
        result = Classification("answer", 0.8, "plain_answer")
    low = result.confidence < cfg.judge_confidence_threshold
    if low and ctx is not None:
        return _escalate(result, output, ctx, turn)
    return Classification(
        result.label, result.confidence, result.rule, result.evidence, low_confidence=low
    )


def _escalate(base: Classification, text: str, ctx: "EvalContext", turn: int) -> Classification:
    from .judge_inputs import conversation, tool_calls

    verdict = ctx.request_judge(
        JudgeTask(
            "decision",
            inputs={
                "turn": turn,
                "conversation": conversation(ctx.run, turn),
                "agent_text": text,
                "tool_calls": tool_calls(ctx.run, turn),
                "available_tools": [t.name for t in ctx.run.available_tools],
            },
        ),
        "decision_match",
    )
    if verdict is None:
        return Classification(
            base.label, base.confidence, base.rule, base.evidence, low_confidence=True
        )
    interp = interpret_verdict(verdict, positive=LABELS, negative=())
    if interp.status == "error":
        return Classification(
            base.label, base.confidence, base.rule, base.evidence, True, "judge_failed", True
        )
    if interp.reason_code == "judge_unsure":
        return Classification(
            base.label, base.confidence, base.rule, base.evidence, True, "judge_unsure", True
        )
    if interp.reason_code == "judge_uncalibrated":  # advisory: keep the heuristic label
        return Classification(
            base.label, base.confidence, base.rule, (*base.evidence, "judge advisory"), False,
            None, True, str(interp.details.get("advisory_label")),
        )  # fmt: skip
    return Classification(
        str(verdict.label), 1.0, "judge", base.evidence, judged=True, low_confidence=False
    )


def count_questions(text: str, pack: LexiconPack, cfg: EvalConfig) -> int:
    """How many sentences of ``text`` ask something (closing pleasantries do not count)."""
    closers = (*pack.closers, *(c.lower() for c in cfg.lexicons.closers_extra))
    cues = (*pack.question_cues, *(c.lower() for c in cfg.lexicons.question_cues_extra))
    n = 0
    for sentence in sentences(text):
        low = sentence.lower().strip()
        asks = (
            low.endswith(("?", "\uff1f", "\u061f"))
            or _wrapped_inverted(sentence)
            or any(cue in low for cue in cues)
        )
        if asks and not any(closer in low for closer in closers):
            n += 1
    return n
