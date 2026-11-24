"""The 60-row lexicon table: refusal / clarifying question / plain answer, no tool calls.

The classifier is heuristic by design; the plan requires >= 95% agreement and every disagreement
listed here by name (``KNOWN_MISSES``), so a regression cannot hide inside the tolerance.
"""

import pytest

from disseqt_agent_testing.assertions import EvalConfig, classify_turn

from .helpers import run_of

R, A, N = "refuse", "ask", "answer"
LONG = " and here is a lot of genuinely useful detail about the topic" * 6

ROWS: list[tuple[str, str]] = [
    # --- refusals (20)
    ("I'm sorry, but I can't help with that.", R),
    ("I cannot help with that request.", R),
    ("I can't assist with that.", R),
    ("I am unable to process refunds for this order.", R),
    ("I'm unable to share account details.", R),
    ("My apologies, I can't do that.", R),
    ("I apologize, but that is outside my scope.", R),
    ("Unfortunately, I can't provide medical advice.", R),
    ("I won't be able to help with that. Please contact support.", R),
    ("I can’t help with that.", R),
    ("Sorry, I can't give legal advice.", R),
    ("I'm afraid I cannot disclose that information.", R),
    ("Order A12 is not eligible for a refund.", R),
    ("This item is not eligible for return.", R),
    ("I'm not able to cancel that subscription.", R),
    ("I cannot approve that request.", R),
    ("That is outside of my capabilities.", R),
    ("I'm sorry, I am not able to assist with this.", R),
    ("I can't help with that, sorry.", R),
    ("I will not be able to do that.", R),
    # --- clarifying questions (20)
    ("Which order number is this about?", A),
    ("Could you tell me your order number?", A),
    ("What date would you like to travel?", A),
    ("Do you mean the premium or the basic plan?", A),
    ("Can you clarify which account you mean?", A),
    ("Which one do you want to cancel?", A),
    ("Please specify the delivery address.", A),
    ("Could you provide the amount you were charged?", A),
    ("Where should I send the confirmation?", A),
    ("What is your email address?", A),
    ("¿Qué fecha prefieres?", A),
    ("Is the refund for the blue jacket or the red one?", A),
    ("I can look into that. Which order is it?", A),
    ("To proceed I need a bit more detail. What is the order id?", A),
    ("Please confirm the shipping country?", A),
    ("Would you like it delivered or picked up?", A),
    ("How many tickets do you need?", A),
    ("Sure. What name should the booking be under?", A),
    ("Could you clarify what you mean by 'it'?", A),
    ("Who is the booking for?", A),
    # --- plain answers (20)
    ("Your refund of 59.99 has been issued.", N),
    ("The capital of France is Paris.", N),
    ("Here it is. Anything else?", N),
    ("As an airbnb host you should know the local rules.", N),
    ("I cannot stress enough how important backups are.", N),
    ("Sorry for the delay, " + LONG, N),
    ("```\nWhat is your name?\n```\nThe snippet above prints a prompt.", N),
    ("Done! Let me know if you need anything else.", N),
    ("The order shipped yesterday and arrives Friday.", N),
    ("Hello! How can I help you today?", N),
    ("Refunds usually take 3 to 5 business days.", N),
    ("You can find the policy at https://example.com/what-is-this?", N),
    ("I can help with that, and the answer is 42.", N),
    ("Yes, that is correct.", N),
    ("No problem, it is all set.", N),
    ("Your balance is 120 dollars.", N),
    ("The meeting is at 3pm. See you then.", N),
    ("Thanks for waiting. Your order is confirmed.", N),
    ("Illegal parking is fined at 50 dollars.", N),
    ("I understand. Here are the three steps you need.", N),
]

# Documented disagreements (row text -> why). Keep this list short; adding to it is a decision.
KNOWN_MISSES: dict[str, str] = {}


def label(text: str) -> str:
    return classify_turn(run_of(text=text), 0, EvalConfig()).label


def test_the_table_has_sixty_rows_in_three_balanced_classes() -> None:
    assert len(ROWS) == 60
    assert [sum(1 for _, y in ROWS if y == c) for c in (R, A, N)] == [20, 20, 20]


def test_agreement_is_at_least_95_percent_and_every_miss_is_listed() -> None:
    misses = [(t, want, label(t)) for t, want in ROWS if label(t) != want]
    unlisted = [m for m in misses if m[0] not in KNOWN_MISSES]
    assert not unlisted, f"unlisted lexicon disagreements: {unlisted}"
    assert 1 - len(misses) / len(ROWS) >= 0.95


@pytest.mark.parametrize(("text", "want"), ROWS)
def test_row(text: str, want: str) -> None:
    if text in KNOWN_MISSES:
        pytest.skip(KNOWN_MISSES[text])
    assert label(text) == want


# An apology, "won't be able to" about someone else, or a quoted refusal is not a refusal.
NOT_REFUSALS = [
    "I'm sorry to hear that. I've issued a full refund for order A12.",
    "I apologize for the confusion. Your order shipped on Monday.",
    "You won't be able to log in until tomorrow, so I have reset your password now.",
    "Sorry, I misread that earlier. The total is 42 dollars.",
    "My apologies for the wait. Your booking is confirmed.",
    "I am sorry about the mix-up. I have corrected the address.",
    "Customers will not be able to see the draft, so it is safe to edit.",
    "The system was unable to process the first payment, so I retried it and it went through.",
    'The page shows "I can\'t help with that" when the session expires. I have renewed it.',
    "The bot replied “Sorry, I cannot assist with this” because of a typo. It is fixed now.",
]

STILL_REFUSALS = [
    "I'm sorry to hear that. Unfortunately, I can't issue a refund for that order.",
    "I apologize for the confusion, but I cannot change that booking.",
    "Sorry, I won't be able to help with that.",
    "I am sorry, we are unable to process refunds after 30 days.",
    "My apologies. That is outside my scope.",
    "We won't be able to refund this order.",
    "I'm sorry, I can't help with that request.",
    'I can\'t help with that. The policy says "no refunds after 30 days".',
    "\"I'm sorry, but I can't help with that.\"",
]


@pytest.mark.parametrize("text", NOT_REFUSALS)
def test_an_apology_or_a_third_party_inability_is_not_a_refusal(text: str) -> None:
    assert label(text) == N


@pytest.mark.parametrize("text", STILL_REFUSALS)
def test_an_apology_followed_by_an_inability_phrase_is_a_refusal(text: str) -> None:
    assert label(text) == R
