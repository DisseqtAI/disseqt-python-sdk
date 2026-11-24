"""The first-shippable demo suite, written as files into a directory."""

from pathlib import Path

AGENTS = "tests.agent_testing._support.demo_agents"

HEADER = """version: 1
defaults:
  repeats: 5
cases:
"""
FAKES = """    fakes:
      get_order:
        effect: read
        schema: {type: object, properties: {order_id: {type: string}}, required: [order_id]}
        responses:
          - {when: {order_id: A12}, returns: {order_id: A12, total: 59.99}}
      check_policy: {effect: read, default: {returns: {eligible: true}}}
      issue_refund: {default: {returns: {amount: 59.99, status: ok}}}
"""
REFUND_EXPECT = """    expect:
      decision: call
      tools:
        - {name: get_order, id: lookup}
        - name: issue_refund
          id: refund
          args: {order_id: {"$m": "ref", source: "call:lookup.result.order_id"}}
          after: [lookup]
      order: partial
"""
YAML = HEADER + f"""  - id: refund_happy
    input: "Refund order A12, it arrived broken"
    run: {{agent: "{AGENTS}:make_demo_agent"}}
{FAKES}{REFUND_EXPECT}  - id: refund_missing_order
    input: "I want a refund for my broken headphones"
    run: {{agent: "{AGENTS}:make_demo_agent"}}
{FAKES}    expect:
      decision: ask
      forbidden: [{{name: issue_refund}}]
  - id: refund_wrong_order
    input: "Refund order A12, it arrived broken"
    run: {{agent: "{AGENTS}:make_demo_wrong_order_agent"}}
{FAKES}{REFUND_EXPECT}"""


def write_demo(root: Path) -> Path:
    cases = root / "tests" / "agent_cases"
    cases.mkdir(parents=True, exist_ok=True)
    (cases / "refund.agent.yaml").write_text(YAML)
    return cases
