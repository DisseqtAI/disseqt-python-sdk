#!/usr/bin/env python3
"""Example usage of the Disseqt Prompt Packs API via DisseqtAPIClient.

This script demonstrates the full prompt-pack lifecycle:
1. Generate a prompt pack
2. Create a run against the pack
3. List & inspect runs
4. Export pack prompts to CSV (by pack ID)
5. Trigger output validations
6. Retrieve validation results & summaries
7. Clean up resources

For Create Run to succeed, the script uses OPENAI_API_KEY from the environment,
or the default below for local runs. Prefer env var in production; do not commit keys.
"""

import os
import sys
import time

# Ensure local src/ is used over any installed package so local changes are picked up
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Default for local example runs; override with OPENAI_API_KEY env var
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-your-openai-api-key-here")

from disseqt_sdk import DisseqtAPIClient  # noqa: E402 — sys.path tweaked above
from disseqt_sdk.client import HTTPError  # noqa: E402
from disseqt_sdk.models.prompt_packs import (  # noqa: E402
    CreateRunRequest,
    GeneratePromptPackRequest,
    MetricEvaluation,
    OutputValidationMetric,
    PaginationParams,
    PromptPackCategory,
    PromptPackOutputValidationCategory,
    PromptPackOutputValidationRequest,
)


def _print_full_http_error(e: HTTPError, context: str = "Request") -> None:
    """Print full HTTP error: status code, message, and response body."""
    print(f"Error {context}: HTTP {e.status_code}: {e.message}")
    if e.response_body:
        print(f"Response body: {e.response_body}")
    print()


def main():
    """Demonstrate Prompt Packs API usage."""

    # ------------------------------------------------------------------
    # 1. Initialize the API client (Kong gateway on localhost:8000)
    # ------------------------------------------------------------------
    client = DisseqtAPIClient(
        project_id="121c8136-5458-494b-a8be-ad46440f4330",
        api_key="99499ed3-f956-4881-bec9-64d5cea0edec",
        base_url="http://localhost:8000",
        timeout=30,
    )

    # ------------------------------------------------------------------
    # 2. Generate a prompt pack
    # ------------------------------------------------------------------
    print("=== Generate Prompt Pack ===")
    try:
        pack = client.generate_prompt_pack(
            GeneratePromptPackRequest(
                pack_name="AI Generated Security Pack",
                pack_short_desc="AI-generated prompts for security testing",
                author="AI Generator",
                domain="Security",
                generation_type="AI",
                categories=[
                    PromptPackCategory(
                        main_category="reliability_and_safety",
                        subcategory="hate_speech",
                        prompts_count=5,
                    ),
                    PromptPackCategory(
                        main_category="privacy_and_security",
                        subcategory="pii_handling",
                        prompts_count=3,
                    ),
                ],
            )
        )
        print(f"Pack created: {pack}")
        pack_id = pack.get("id") or pack.get("pack_id") or pack.get("data", {}).get("id")
        print(f"Pack ID: {pack_id}")
    except HTTPError as e:
        _print_full_http_error(e, "generating pack")
        return
    except Exception as e:
        print(f"Error generating pack: {type(e).__name__}: {e}")
        return

    # ------------------------------------------------------------------
    # 3. Create a run (backend validates the LLM api_key against the provider)
    # Pack generation is async; if pack has no prompts yet we wait and retry.
    # ------------------------------------------------------------------
    print("\n=== Create Run ===")
    llm_api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
        or DEFAULT_OPENAI_API_KEY
    )
    if not llm_api_key:
        print(
            "Skipping Create Run: set OPENAI_API_KEY (or LLM_API_KEY) in the environment. "
            "The backend validates the key against the provider (e.g. OpenAI)."
        )
        run_id = None
    else:
        run_request = CreateRunRequest(
            run_name="GPT-4 Evaluation Run",
            run_type="evaluation",
            api_key=llm_api_key,
            model_name="gpt-4",
            provider="openai",
        )
        max_wait_sec = 90
        poll_interval_sec = 15
        run_id = None
        for attempt in range(max_wait_sec // poll_interval_sec + 1):
            try:
                run = client.create_run(pack_id, run_request)
                print(f"Run created: {run}")
                run_id = run.get("id") or run.get("run_id") or run.get("data", {}).get("id")
                print(f"Run ID: {run_id}")
                break
            except HTTPError as e:
                _print_full_http_error(e, "Create Run")
                if (
                    e.status_code == 400
                    and "no prompts to evaluate" in (e.response_body or "").lower()
                ):
                    if attempt < max_wait_sec // poll_interval_sec:
                        if attempt == 0:
                            print(
                                "Pack is still generating prompts; waiting for generation to complete..."
                            )
                        print(f"  Waiting {poll_interval_sec}s before retry ({attempt + 1})...")
                        time.sleep(poll_interval_sec)
                    else:
                        print(
                            "Hint: Pack generation may still be in progress. Try again in a minute."
                        )
                        return
                else:
                    if "invalid API key" in (e.response_body or "").lower():
                        print(
                            "Hint: Ensure OPENAI_API_KEY (or LLM_API_KEY) is a valid key for the provider."
                        )
                    return
            except Exception as e:
                print(f"Error creating run: {type(e).__name__}: {e}")
                return

    # ------------------------------------------------------------------
    # 4. List runs for the pack
    # ------------------------------------------------------------------
    print("\n=== List Runs ===")
    try:
        runs = client.list_runs(pack_id, pagination=PaginationParams(limit=10, offset=0))
        print(f"Runs: {runs}")
    except Exception as e:
        print(f"Error listing runs: {e}")

    # ------------------------------------------------------------------
    # 5. Export pack prompts to CSV (by pack ID)
    # ------------------------------------------------------------------
    print("\n=== Export Pack Prompts to CSV ===")
    try:
        out = client.download_pack_csv(pack_id)
        if isinstance(out, str):
            path = "pack_prompts_export.csv"
            with open(path, "w") as f:
                f.write(out)
            print(f"Exported {len(out.splitlines())} lines to {path}")
        else:
            print(f"Download response (e.g. URL): {out}")
    except HTTPError as e:
        _print_full_http_error(e, "download pack CSV")
    except Exception as e:
        print(f"Error exporting pack CSV: {e}")

    # ------------------------------------------------------------------
    # 6. Get run details (with outputs)
    # ------------------------------------------------------------------
    print("\n=== Get Run Details ===")
    try:
        run_details = client.get_run(run_id, include_outputs=True)
        print(f"Run details: {run_details}")
    except Exception as e:
        print(f"Error getting run: {e}")

    # ------------------------------------------------------------------
    # 7. Get run outputs
    # ------------------------------------------------------------------
    print("\n=== Get Run Outputs ===")
    try:
        outputs = client.get_run_outputs(run_id)
        print(f"Outputs: {outputs}")
    except Exception as e:
        print(f"Error getting outputs: {e}")

    # ------------------------------------------------------------------
    # 8. Create output validation
    # ------------------------------------------------------------------
    print("\n=== Create Output Validation ===")
    try:
        validation = client.create_output_validation(
            run_id,
            PromptPackOutputValidationRequest(
                prompt_pack_output_validation_run_name="SDK Test Validation",
                metric_evaluations=[
                    MetricEvaluation(
                        metric_name=OutputValidationMetric.HATE_SPEECH.value,
                        category=PromptPackOutputValidationCategory.OUTPUT_VALIDATION.value,
                    ),
                    MetricEvaluation(
                        metric_name=OutputValidationMetric.TOXICITY.value,
                        category=PromptPackOutputValidationCategory.OUTPUT_VALIDATION.value,
                    ),
                ],
            ),
        )
        print(f"Validation created: {validation}")
        validation_id = (
            validation.get("id")
            or validation.get("validation_id")
            or validation.get("data", {}).get("id")
        )
        print(f"Validation ID: {validation_id}")
    except Exception as e:
        print(f"Error creating validation: {e}")
        return

    # ------------------------------------------------------------------
    # 9. List output validations for the run
    # ------------------------------------------------------------------
    print("\n=== List Run Output Validations ===")
    try:
        validations = client.list_run_output_validations(run_id)
        print(f"Validations: {validations}")
    except Exception as e:
        print(f"Error listing validations: {e}")

    # ------------------------------------------------------------------
    # 10. Get output validation details
    # ------------------------------------------------------------------
    print("\n=== Get Output Validation Details ===")
    try:
        val_details = client.get_output_validation(validation_id)
        print(f"Validation details: {val_details}")
    except Exception as e:
        print(f"Error getting validation: {e}")

    # ------------------------------------------------------------------
    # 11. Get output validation summary
    # ------------------------------------------------------------------
    print("\n=== Get Output Validation Summary ===")
    try:
        summary = client.get_output_validation_summary(validation_id)
        print(f"Summary: {summary}")
    except Exception as e:
        print(f"Error getting summary: {e}")

    # ------------------------------------------------------------------
    # 12. Get output validation results (paginated)
    # ------------------------------------------------------------------
    print("\n=== Get Output Validation Results ===")
    try:
        results = client.get_output_validation_results(
            validation_id,
            pagination=PaginationParams(limit=10, offset=0),
        )
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error getting results: {e}")

    # ------------------------------------------------------------------
    # 13. Get grouped outputs
    # ------------------------------------------------------------------
    print("\n=== Get Grouped Outputs ===")
    try:
        grouped = client.get_output_validation_grouped_outputs(validation_id)
        print(f"Grouped outputs: {grouped}")
    except Exception as e:
        print(f"Error getting grouped outputs: {e}")

    # ------------------------------------------------------------------
    # 14. List all output validations for the pack
    # ------------------------------------------------------------------
    print("\n=== List Pack Output Validations ===")
    try:
        pack_validations = client.list_pack_output_validations(pack_id)
        print(f"Pack validations: {pack_validations}")
    except Exception as e:
        print(f"Error listing pack validations: {e}")

    # ------------------------------------------------------------------
    # 15. Clean up: delete validation, then delete run
    # ------------------------------------------------------------------
    print("\n=== Cleanup ===")
    try:
        del_val = client.delete_output_validation(validation_id)
        print(f"Deleted validation: {del_val}")
    except Exception as e:
        print(f"Error deleting validation: {e}")

    try:
        del_run = client.delete_run(run_id)
        print(f"Deleted run: {del_run}")
    except Exception as e:
        print(f"Error deleting run: {e}")

    print("\n=== Prompt Packs Demo Complete ===")


if __name__ == "__main__":
    main()
