# Installation Guide

## Install from GitHub (Recommended for Development)

### For Public Repository

Install the latest version from the main branch:

```bash
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git
```

### Install Specific Version

Install from a specific branch:

```bash
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git@main
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git@develop
```

Install from a specific tag (when available):

```bash
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git@v0.1.0
```

Install from a specific commit:

```bash
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git@<commit-hash>
```

### For Private Repository

If the repository is private, you'll need to authenticate:

#### Option 1: Using Personal Access Token (Recommended)

```bash
pip install git+https://<username>:<token>@github.com/DisseqtAI/disseqt-python-sdk.git
```

#### Option 2: Using SSH (if configured)

```bash
pip install git+ssh://git@github.com/DisseqtAI/disseqt-python-sdk.git
```

#### Option 3: Using GitHub CLI

```bash
gh auth login
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git
```

## Install from PyPI (When Published)

Once published to PyPI:

```bash
pip install disseqt-sdk
```

## Installation in Virtual Environment (Recommended)

### Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install package
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git
```

### Using uv (Faster Alternative)

```bash
# Create virtual environment and install
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

uv pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git
```

## Installation in requirements.txt

Add to your `requirements.txt`:

```text
# Latest from main branch
git+https://github.com/DisseqtAI/disseqt-python-sdk.git

# Specific branch
git+https://github.com/DisseqtAI/disseqt-python-sdk.git@main

# Specific tag
git+https://github.com/DisseqtAI/disseqt-python-sdk.git@v0.1.0

# Specific commit
git+https://github.com/DisseqtAI/disseqt-python-sdk.git@abc1234567890
```

Then install:

```bash
pip install -r requirements.txt
```

## Installation in pyproject.toml

For projects using modern Python packaging:

```toml
[project]
dependencies = [
    "disseqt-sdk @ git+https://github.com/DisseqtAI/disseqt-python-sdk.git",
]
```

Or with uv:

```bash
uv add git+https://github.com/DisseqtAI/disseqt-python-sdk.git
```

## Editable Installation (For Development)

If you want to contribute or develop on the SDK:

```bash
# Clone the repository
git clone https://github.com/DisseqtAI/disseqt-python-sdk.git
cd disseqt-python-sdk

# Install in editable mode with uv (recommended)
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

## Verifying Installation

After installation, verify it works:

```python
import disseqt_sdk
from disseqt_sdk import Client

print("disseqt-sdk installed successfully!")

# Test basic import
client = Client(project_id="test", api_key="test")
print(f"Client initialized: {client}")
```

Or from command line:

```bash
python -c "import disseqt_sdk; print('✓ disseqt-sdk installed successfully')"
```

## Prerequisites

- Python 3.10.14 or higher
- Git (for git-based installation)
- pip or uv package manager

## Troubleshooting

### Git Not Found

If you get "git not found" error:

- **macOS**: `brew install git`
- **Ubuntu/Debian**: `sudo apt-get install git`
- **Windows**: Download from [git-scm.com](https://git-scm.com)

### Authentication Issues

For private repositories:

1. Use SSH keys: Set up GitHub SSH keys
2. Use Personal Access Token: Create token with `repo` scope
3. Use GitHub CLI: Run `gh auth login`

### SSL Certificate Errors

If you encounter SSL errors:

```bash
pip install --trusted-host github.com git+https://github.com/DisseqtAI/disseqt-python-sdk.git
```

### Permission Errors

Use `--user` flag for user-level installation:

```bash
pip install --user git+https://github.com/DisseqtAI/disseqt-python-sdk.git
```

## Upgrade/Reinstall

To upgrade to the latest version:

```bash
pip install --upgrade --force-reinstall git+https://github.com/DisseqtAI/disseqt-python-sdk.git
```

## Uninstall

```bash
pip uninstall disseqt-sdk
```

## Support

For installation issues:
- GitHub Issues: https://github.com/DisseqtAI/disseqt-python-sdk/issues
- Email: support@disseqt.ai

