"""AgenticBehaviourRequest carries the union of agentic + LLM text fields.

A realtime policy can mix agentic validators (tool_call_accuracy,
topic_adherence, …) with text validators (factual_consistency,
data_leakage, …). validate(request, policies=[...]) sends ONE input bag,
so the carrier must hold every field the policy's validators read —
otherwise the text rules skip with missing_input and the policy silently
under-evaluates.
"""

from disseqt_sdk.models.agentic_behaviour import AgenticBehaviourRequest


class TestAgenticRequestUnion:
    def test_full_union_serializes_all_domains(self):
        req = AgenticBehaviourRequest(
            prompt="What is the capital of France?",
            context="France is a country in Europe.",
            response="The capital of France is Paris.",
            conversation_history=["user: hi", "agent: hello"],
            tool_calls=[{"name": "lookup_capital", "args": {"country": "France"}}],
            agent_responses=["Looking that up."],
            reference_data={"expected": "Paris"},
        )
        bag = req.to_input_data()
        assert bag["llm_input_query"] == "What is the capital of France?"
        assert bag["llm_input_context"] == "France is a country in Europe."
        assert bag["llm_output"] == "The capital of France is Paris."
        assert bag["conversation_history"] == ["user: hi", "agent: hello"]
        assert bag["tool_calls"][0]["name"] == "lookup_capital"
        assert bag["agent_responses"] == ["Looking that up."]
        assert bag["reference_data"] == {"expected": "Paris"}

    def test_text_fields_renamed_to_wire_shape(self):
        bag = AgenticBehaviourRequest(prompt="q", tool_calls=[{"name": "t"}]).to_input_data()
        assert bag == {"llm_input_query": "q", "tool_calls": [{"name": "t"}]}
        # SDK names never leak onto the wire
        assert "prompt" not in bag and "response" not in bag and "context" not in bag

    def test_none_fields_omitted(self):
        bag = AgenticBehaviourRequest(tool_calls=[{"name": "t"}]).to_input_data()
        assert "llm_input_query" not in bag
        assert "llm_input_context" not in bag
        assert "llm_output" not in bag

    def test_agentic_only_unchanged(self):
        # Pre-0.7.x call sites that pass agentic fields only are untouched.
        bag = AgenticBehaviourRequest(
            conversation_history=["user: hi"],
            agent_responses=["hello"],
        ).to_input_data()
        assert bag == {
            "conversation_history": ["user: hi"],
            "agent_responses": ["hello"],
        }

    def test_empty_request_serializes_empty(self):
        # An empty bag still triggers the validate() empty-input guard.
        assert AgenticBehaviourRequest().to_input_data() == {}
