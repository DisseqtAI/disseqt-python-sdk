"""Language packs for refusal and question detection. Punctuation rules apply to every language;
only the word lists are per pack. Add a language with ``register_lexicon``."""

import threading
from dataclasses import dataclass

from ..models import thaw  # noqa: F401  (kept for symmetry with other modules)

_LOCK = threading.RLock()


@dataclass(slots=True, frozen=True)
class LexiconPack:
    lang: str
    # always a refusal when they open the answer (statements of inability)
    refusal_prefixes: tuple[str, ...]
    # only a refusal when followed by one of ``action_verbs`` ("I cannot help" yes, "I cannot
    # stress enough" no)
    modal_prefixes: tuple[str, ...]
    action_verbs: tuple[str, ...]
    refusal_patterns: tuple[str, ...]  # regexes, matched with word boundaries
    interrogatives: tuple[str, ...]
    question_cues: tuple[str, ...]  # clarification phrases that need no question mark
    closers: tuple[str, ...]  # "anything else?" style sign-offs: never a clarifying question
    # openings that are a refusal only when an inability phrase follows ("I'm sorry, I can't
    # do that" yes, "I'm sorry to hear that. I've issued the refund." no)
    apology_prefixes: tuple[str, ...] = ()


_PACKS: dict[str, LexiconPack] = {}


def register_lexicon(pack: LexiconPack, *, override: bool = False) -> None:
    with _LOCK:
        if pack.lang in _PACKS and not override:
            raise ValueError(f"lexicon {pack.lang!r} is already registered")
        _PACKS[pack.lang] = pack


def unregister_lexicon(lang: str) -> None:
    with _LOCK:
        _PACKS.pop(lang, None)


def get_lexicon(lang: str) -> LexiconPack:
    with _LOCK:
        pack = _PACKS.get(lang) or _PACKS.get(lang.split("-")[0])
        known = sorted(_PACKS)
    if pack is None:
        raise ValueError(f"unknown lexicon {lang!r}; registered: {known}")
    return pack


EN = LexiconPack(
    lang="en",
    apology_prefixes=(
        "i'm sorry",
        "i am sorry",
        "i apologize",
        "i apologise",
        "my apologies",
        "sorry,",
        "sorry.",
    ),
    refusal_prefixes=(
        "i'm unable to",
        "i am unable to",
        "unfortunately, i can't",
        "unfortunately, i cannot",
        "i'm not able to",
        "i am not able to",
        "i'm afraid i can't",
        "i'm afraid i cannot",
    ),
    modal_prefixes=(
        "i cannot",
        "i can't",
        "i can not",
        "i won't",
        "i will not",
        "i'm not going to",
    ),
    action_verbs=(
        "help", "assist", "provide", "do", "share", "comply", "answer", "support", "fulfill",
        "discuss", "give", "create", "generate", "access", "process", "complete", "perform",
        "proceed", "make", "issue", "refund", "cancel", "change", "delete", "modify", "disclose",
        "recommend", "advise", "write", "send", "approve", "grant", "authorize", "verify",
    ),  # fmt: skip
    refusal_patterns=(
        r"\b(?:i|we) (?:can'?t|cannot) help with\b",
        r"\b(?:i|we)(?:'m|'re| am| are) not able to (?:assist|help)\b",
        r"\b(?:i|we) (?:won'?t|will not|cannot|can'?t) be able to\b",
        r"\boutside (?:of )?my (?:scope|capabilities)\b",
        r"\bi (?:can'?t|cannot|won'?t) (?:help|assist|provide|do)\b",
        r"\b(?:is|are) not eligible\b",
        r"\bnot eligible for\b",
        r"\b(?:i|we)(?:'m|'re| am| are) unable to (?:help|assist|process|complete)\b",
    ),
    interrogatives=(
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "can", "could",
        "would", "should", "do", "does", "did", "is", "are", "was", "were", "will", "shall",
        "may", "might", "have", "has",
    ),  # fmt: skip
    question_cues=(
        "could you clarify",
        "can you clarify",
        "do you mean",
        "which one",
        "please specify",
        "please confirm",
        "could you tell me",
        "can you tell me",
        "could you provide",
        "can you provide",
        "please provide",
        "what is your",
        "which order",
    ),
    closers=(
        "anything else",
        "let me know if you have any other questions",
        "let me know if you need",
        "let me know if there",
        "does that help",
        "does that make sense",
        "hope that helps",
        "how else can i help",
        "how can i help",
        "is there anything",
        "feel free to ask",
    ),
)

register_lexicon(EN, override=True)
