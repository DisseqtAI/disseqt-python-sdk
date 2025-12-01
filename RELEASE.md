# Release Process

This document outlines the release process for the Disseqt SDK.

## Release Checklist

### Pre-Release

- [ ] All tests pass (`uv run pytest`)
- [ ] Code quality checks pass (`uv run pre-commit run --all-files`)
- [ ] Coverage is ≥90% (`uv run pytest --cov=disseqt_sdk --cov-report=term-missing`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated with new version
- [ ] Version is bumped in `pyproject.toml`

### Release Steps

1. **Update Version**
   ```bash
   # Update version in pyproject.toml
   # Update CHANGELOG.md with release date
   ```

2. **Lock Dependencies**
   ```bash
   uv lock
   ```

3. **Final Testing**
   ```bash
   uv run pytest -q --cov=disseqt_sdk --cov-report=term-missing
   uv run pre-commit run --all-files
   ```

4. **Commit and Tag**
   ```bash
   git add .
   git commit -m "Release v0.1.0"
   git tag v0.1.0
   git push origin main --tags
   ```

5. **Build Package**
   ```bash
   uv build
   ```

6. **Publish** (when ready for public distribution)
   ```bash
   # This will be configured when publishing is enabled
   # uv publish
   ```

### Post-Release

- [ ] Verify release artifacts
- [ ] Update documentation site (if applicable)
- [ ] Announce release to stakeholders
- [ ] Monitor for issues

## Version Scheme

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

## Release Types

### Major Release (1.0.0, 2.0.0, etc.)
- Breaking changes to public API
- Significant new features
- Architecture changes

### Minor Release (0.1.0, 0.2.0, etc.)
- New validators or domains
- New features (backwards compatible)
- Performance improvements

### Patch Release (0.1.1, 0.1.2, etc.)
- Bug fixes
- Security patches
- Documentation updates
- Dependency updates

## Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Individual feature branches
- `hotfix/*`: Critical fixes for production

## Automation

Future releases may include:
- Automated testing on multiple Python versions
- Automated security scanning
- Automated dependency updates
- Automated changelog generation

## Support Policy

- **Current Major Version**: Full support
- **Previous Major Version**: Security fixes only
- **Older Versions**: No support

## Contact

For release-related questions:
- Email: support@disseqt.ai
- Internal: Release team
