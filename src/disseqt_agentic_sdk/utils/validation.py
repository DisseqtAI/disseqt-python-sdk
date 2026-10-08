"""
Shared HTTP header-value validation.

Lives in ``utils`` (not ``client`` or ``transport``) specifically so both
``client.client`` and ``transport.http`` can import the same function
without a circular import -- ``client`` already imports ``HTTPTransport``
from ``transport``, and ``transport`` needs this same check at the true
choke point every value passes through before becoming a header
(``_send_group``), independent of which upstream class produced it.
"""

from __future__ import annotations

# The two things that actually break sending a value as an HTTP header,
# checked directly against the real failure modes rather than an
# assumed character blacklist:
#
#   1. Embedded \r / \n — `requests` raises InvalidHeader for these
#      (header-injection risk); everything else in the C0-control /
#      DEL range (tab, NUL, bell, ...) is, in practice, sent over the
#      wire by `requests` without complaint, so it isn't checked here.
#   2. Anything outside the Latin-1 range — `http.client.putheader`
#      encodes header values as Latin-1 and raises an uncaught
#      UnicodeEncodeError for anything outside it (an emoji, most
#      non-Latin scripts, a copy-pasted smart quote). That exception
#      isn't a `requests.exceptions.RequestException`, so nothing
#      downstream catches it — it can kill the background flush thread
#      outright. Checked directly via an encode attempt rather than an
#      enumerated character set, so it can't drift out of sync with
#      what actually breaks. TP-2128 round-3 senior review P1 #1.1
#      (the original C0-control blacklist rejected harmless characters
#      like tab while letting the actual crash-risk class through).
#
# Originally application_id-only (TP-2128 round-2 P2 #2.3), generalized
# to api_key (now also travels as a header — see transport/http.py).
# Validated at both client.client.DisseqtAgenticClient.__init__ and
# transport.http.HTTPTransport.__init__, so a directly-constructed
# HTTPTransport can't bypass the check.
_HEADER_VALUE_DISALLOWED_LINE_BREAKS = frozenset("\r\n")


def validate_header_value(
    value: str, field_name: str, example_hint: str = "a plain ASCII token (typically a UUID)"
) -> None:
    """
    Raise ``ValueError`` immediately when ``value`` is unsafe to send as
    an HTTP header.

    ``field_name``'s value is opaque to us but must not carry characters
    that would break sending it as an HTTP header on every send.
    Combined with retain-on-failure (TP-2128 round-2 P1 #1.2) an
    un-caught failure here would otherwise retry forever against a
    value that will never succeed. Fail loudly here instead. TP-2128
    round-2 P2 #2.3, tightened by round-3 P1 #1.1 to match what
    ``requests``/``http.client`` actually reject rather than an assumed
    control-character list. Caller is responsible for the emptiness
    check — an empty string always passes (no-op): a required-field
    check is a separate, more actionable error than this one.
    """
    if _HEADER_VALUE_DISALLOWED_LINE_BREAKS & set(value):
        raise ValueError(
            f"{field_name} contains a carriage return or newline character, "
            f"which requests rejects as a header-injection risk on every send. "
            f"Use {example_hint}."
        )
    try:
        value.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"{field_name} contains a character outside the Latin-1 range "
            f"({exc}), which would crash HTTP header encoding on every send. "
            f"Use {example_hint}."
        ) from exc
