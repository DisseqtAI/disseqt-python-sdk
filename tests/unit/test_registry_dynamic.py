"""Test dynamic registry system."""

import pytest

from disseqt_sdk.enums import InputValidation, OutputValidation, ValidatorDomain
from disseqt_sdk.registry import (
    _VALIDATOR_REGISTRY,
    get_validator_metadata,
    list_registered_validators,
    register_validator,
)


class TestValidatorRegistry:
    """Test validator registry functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        _VALIDATOR_REGISTRY.clear()

    def test_register_validator_decorator(self):
        """Test validator registration via decorator."""
        @register_validator(
            domain=ValidatorDomain.INPUT_VALIDATION,
            slug=InputValidation.TOXICITY.value,
            path_template="/api/v1/sdk/validators/{domain}/{validator}",
        )
        class TestValidator:
            pass

        # Check that validator was registered
        key = f"{ValidatorDomain.INPUT_VALIDATION.value}:{InputValidation.TOXICITY.value}"
        assert key in _VALIDATOR_REGISTRY

        metadata = _VALIDATOR_REGISTRY[key]
        assert metadata["class"] == TestValidator
        assert metadata["domain"] == ValidatorDomain.INPUT_VALIDATION
        assert metadata["slug"] == InputValidation.TOXICITY.value
        assert metadata["path_template"] == "/api/v1/sdk/validators/{domain}/{validator}"

    def test_register_multiple_validators(self):
        """Test registering multiple validators."""
        @register_validator(
            domain=ValidatorDomain.INPUT_VALIDATION,
            slug=InputValidation.TOXICITY.value,
        )
        class ToxicityValidator:
            pass

        @register_validator(
            domain=ValidatorDomain.OUTPUT_VALIDATION,
            slug=OutputValidation.FACTUAL_CONSISTENCY.value,
        )
        class FactualValidator:
            pass

        # Check both validators are registered
        assert len(_VALIDATOR_REGISTRY) == 2

        toxicity_key = f"{ValidatorDomain.INPUT_VALIDATION.value}:{InputValidation.TOXICITY.value}"
        factual_key = f"{ValidatorDomain.OUTPUT_VALIDATION.value}:{OutputValidation.FACTUAL_CONSISTENCY.value}"

        assert toxicity_key in _VALIDATOR_REGISTRY
        assert factual_key in _VALIDATOR_REGISTRY

        assert _VALIDATOR_REGISTRY[toxicity_key]["class"] == ToxicityValidator
        assert _VALIDATOR_REGISTRY[factual_key]["class"] == FactualValidator

    def test_get_validator_metadata_success(self):
        """Test successful retrieval of validator metadata."""
        @register_validator(
            domain=ValidatorDomain.INPUT_VALIDATION,
            slug=InputValidation.BIAS.value,
            path_template="/custom/path/{domain}/{validator}",
        )
        class BiasValidator:
            pass

        metadata = get_validator_metadata(
            ValidatorDomain.INPUT_VALIDATION,
            InputValidation.BIAS.value
        )

        assert metadata["class"] == BiasValidator
        assert metadata["domain"] == ValidatorDomain.INPUT_VALIDATION
        assert metadata["slug"] == InputValidation.BIAS.value
        assert metadata["path_template"] == "/custom/path/{domain}/{validator}"

    def test_get_validator_metadata_not_found(self):
        """Test error when validator metadata not found."""
        with pytest.raises(KeyError, match="Validator not registered"):
            get_validator_metadata(
                ValidatorDomain.INPUT_VALIDATION,
                "nonexistent-validator"
            )

    def test_list_registered_validators_empty(self):
        """Test listing validators when registry is empty."""
        validators = list_registered_validators()
        assert validators == {}

    def test_list_registered_validators_with_data(self):
        """Test listing validators with registered data."""
        @register_validator(
            domain=ValidatorDomain.INPUT_VALIDATION,
            slug=InputValidation.TOXICITY.value,
        )
        class ToxicityValidator:
            pass

        @register_validator(
            domain=ValidatorDomain.OUTPUT_VALIDATION,
            slug=OutputValidation.FACTUAL_CONSISTENCY.value,
        )
        class FactualValidator:
            pass

        validators = list_registered_validators()

        assert len(validators) == 2

        toxicity_key = f"{ValidatorDomain.INPUT_VALIDATION.value}:{InputValidation.TOXICITY.value}"
        factual_key = f"{ValidatorDomain.OUTPUT_VALIDATION.value}:{OutputValidation.FACTUAL_CONSISTENCY.value}"

        assert toxicity_key in validators
        assert factual_key in validators

        # Verify it returns a copy (not the original registry)
        validators.clear()
        assert len(_VALIDATOR_REGISTRY) == 2

    def test_decorator_returns_original_class(self):
        """Test that decorator returns the original class unchanged."""
        class OriginalValidator:
            def test_method(self):
                return "test"

        decorated_class = register_validator(
            domain=ValidatorDomain.INPUT_VALIDATION,
            slug=InputValidation.TOXICITY.value,
        )(OriginalValidator)

        # Should be the same class
        assert decorated_class is OriginalValidator
        assert decorated_class().test_method() == "test"

    def test_default_path_template(self):
        """Test default path template is used when not specified."""
        @register_validator(
            domain=ValidatorDomain.INPUT_VALIDATION,
            slug=InputValidation.TOXICITY.value,
        )
        class TestValidator:
            pass

        metadata = get_validator_metadata(
            ValidatorDomain.INPUT_VALIDATION,
            InputValidation.TOXICITY.value
        )

        assert metadata["path_template"] == "/api/v1/sdk/validators/{domain}/{validator}"

    def test_custom_path_template(self):
        """Test custom path template is preserved."""
        custom_template = "/custom/api/{domain}/validate/{validator}"

        @register_validator(
            domain=ValidatorDomain.INPUT_VALIDATION,
            slug=InputValidation.TOXICITY.value,
            path_template=custom_template,
        )
        class TestValidator:
            pass

        metadata = get_validator_metadata(
            ValidatorDomain.INPUT_VALIDATION,
            InputValidation.TOXICITY.value
        )

        assert metadata["path_template"] == custom_template

    def test_registry_key_format(self):
        """Test that registry keys follow expected format."""
        @register_validator(
            domain=ValidatorDomain.INPUT_VALIDATION,
            slug=InputValidation.TOXICITY.value,
        )
        class TestValidator:
            pass

        expected_key = "input-validation:toxicity"
        assert expected_key in _VALIDATOR_REGISTRY

        # Verify the key format matches domain.value:slug pattern
        domain_value = ValidatorDomain.INPUT_VALIDATION.value
        slug_value = InputValidation.TOXICITY.value
        assert expected_key == f"{domain_value}:{slug_value}"
