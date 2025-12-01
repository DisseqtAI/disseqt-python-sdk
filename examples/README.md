# Examples

This directory contains example scripts demonstrating how to use the Disseqt SDK.

## Available Examples

### 1. `example.py` - Basic Validator Examples

Demonstrates the core functionality of the SDK with various validator types:

- **Input Validation**: Toxicity detection, bias detection
- **Output Validation**: Factual consistency checking
- **Agentic Behavior**: Topic adherence validation
- **MCP Security**: Prompt injection detection
- **RAG Grounding**: Context relevance validation

**Usage:**
```bash
python examples/example.py
```

### 2. `example_composite_score.py` - Composite Score Evaluator

Shows how to use the Composite Score Evaluator for comprehensive multi-metric evaluation:

- Full composite evaluation with custom weights
- Configurable label thresholds for each category
- Evaluation of 18 metrics across 3 main categories:
  - Factual/Semantic Alignment (9 metrics)
  - Language Quality (3 metrics)
  - Safety/Security/Integrity (6 metrics)
- Simple evaluation with default settings

**Usage:**
```bash
python examples/example_composite_score.py
```

**Features Demonstrated:**
- Custom weight configuration
- Label threshold overrides
- Overall confidence scoring
- Detailed breakdown analysis
- Credit tracking

### 3. `verify_installation.py` - Installation Verification

A utility script to verify that the Disseqt SDK is correctly installed and all components can be imported.

**Usage:**
```bash
python examples/verify_installation.py
```

**What it checks:**
- Package import
- Client and model imports
- Validator imports
- Client instantiation
- Validator instantiation
- Package version

## Prerequisites

Before running these examples, ensure you have:

1. **Installed the SDK:**
   ```bash
   pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git
   ```

2. **Obtained API credentials:**
   - Project ID
   - API Key

3. **Set up your credentials** in the example files or environment variables.

## Configuration

Most examples require you to configure your API credentials. Look for these lines in each script:

```python
client = Client(
    project_id="your_project_id",  # Replace with your project ID
    api_key="your_api_key",         # Replace with your API key
)
```

## Running Examples

### From the project root:
```bash
python examples/example.py
python examples/example_composite_score.py
python examples/verify_installation.py
```

### From the examples directory:
```bash
cd examples
python example.py
python example_composite_score.py
python verify_installation.py
```

## Example Output

Each script provides detailed console output showing:
- Request configuration
- API responses
- Validation results
- Error handling (if applicable)

## Need Help?

- **Documentation**: See the main [README.md](../README.md)
- **API Reference**: Check the [SDK documentation](../README.md#available-validators)
- **Support**: Contact support@disseqt.ai

## Contributing

If you have additional example use cases, feel free to contribute by:
1. Creating a new example script
2. Adding it to this directory
3. Documenting it in this README
4. Submitting a pull request

