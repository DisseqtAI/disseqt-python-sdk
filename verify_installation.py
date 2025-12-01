#!/usr/bin/env python3
"""
Quick verification script to test disseqt-sdk installation.
Run this after installing the package to verify everything works.
"""

def main():
    print("🔍 Verifying disseqt-sdk installation...\n")

    # Test 1: Import package
    try:
        import disseqt_sdk
        print("✅ Package import successful")
    except ImportError as e:
        print(f"❌ Failed to import disseqt_sdk: {e}")
        return False

    # Test 2: Import Client
    try:
        from disseqt_sdk import Client, SDKConfigInput
        print("✅ Client import successful")
    except ImportError as e:
        print(f"❌ Failed to import Client: {e}")
        return False

    # Test 3: Import models
    try:
        from disseqt_sdk.models.agentic_behaviour import AgenticBehaviourRequest  # noqa: F401
        from disseqt_sdk.models.input_validation import InputValidationRequest
        from disseqt_sdk.models.output_validation import OutputValidationRequest  # noqa: F401
        print("✅ Models import successful")
    except ImportError as e:
        print(f"❌ Failed to import models: {e}")
        return False

    # Test 4: Import validators
    try:
        from disseqt_sdk.validators.agentic_behavior.reliability import (
            TopicAdherenceValidator,  # noqa: F401
        )
        from disseqt_sdk.validators.input.safety import ToxicityValidator
        from disseqt_sdk.validators.output.accuracy import FactualConsistencyValidator  # noqa: F401
        print("✅ Validators import successful")
    except ImportError as e:
        print(f"❌ Failed to import validators: {e}")
        return False

    # Test 5: Create client instance
    try:
        Client(project_id="test_project", api_key="test_key")
        print("✅ Client instantiation successful")
    except Exception as e:
        print(f"❌ Failed to create Client instance: {e}")
        return False

    # Test 6: Create validator instance
    try:
        ToxicityValidator(
            data=InputValidationRequest(prompt="Test prompt"),
            config=SDKConfigInput(threshold=0.5)
        )
        print("✅ Validator instantiation successful")
    except Exception as e:
        print(f"❌ Failed to create validator instance: {e}")
        return False

    # Test 7: Check package version
    try:
        version = getattr(disseqt_sdk, '__version__', '0.1.0')
        print(f"✅ Package version: {version}")
    except Exception as e:
        print(f"⚠️  Could not retrieve version: {e}")

    print("\n🎉 All verification tests passed!")
    print("\nYou can now use disseqt-sdk in your projects.")
    print("\nQuick example:")
    print("""
from disseqt_sdk import Client, SDKConfigInput
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.validators.input.safety import ToxicityValidator

client = Client(project_id="your_project_id", api_key="your_api_key")
validator = ToxicityValidator(
    data=InputValidationRequest(prompt="Your text here"),
    config=SDKConfigInput(threshold=0.5)
)
result = client.validate(validator)
print(result)
    """)

    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

