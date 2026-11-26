# Release Process

This document outlines the release process for the Disseqt SDK.

## How to Release

Releases are automated in two steps. `main` is protected by the
`security-enforcement` ruleset (PR, 1 approval and `lint` / `test` / `trivy`
required, no bypass), so nothing pushes to `main` directly.

1. **Open the release PR.** Actions → **Release** → *Run workflow* on `main`,
   choose `patch` / `minor` / `major`. The workflow bumps `pyproject.toml`,
   moves `## [Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] - <date>`, runs
   `uv lock` and the unit + agentic tests, pushes `release-pr/vX.Y.Z` and
   opens a **Release vX.Y.Z** PR. It also starts CI on that branch, since a
   PR opened by `GITHUB_TOKEN` does not trigger `pull_request` workflows.
2. **Review and merge the PR.** On merge, **Tag release** tags the merge
   commit `vX.Y.Z`, creates the GitHub release from the CHANGELOG section and
   starts **Publish to PyPI** at that tag.

Then check that **Publish to PyPI** passed (approve the `pypi` environment if
it asks) and that https://pypi.org/p/disseqt-ai-sdk shows the new version.
The backend picks up the new version through registry polling: no Vault bump
or restart.

### Rules

- Feature PRs add entries under `## [Unreleased]` only. Never edit the
  version by hand: the Release workflow refuses to bump past an untagged
  version.
- Only one release PR at a time. Merge or close it (and delete its branch)
  before running **Release** again.
- If **Tag release** fails after the merge, re-run it, or run it manually:
  it tags whatever version `main` declares.
- If PyPI publishing fails, re-run only **Publish to PyPI** at the tag. Do
  not run **Release** again; it would bump the version a second time.
- Repo setting needed: Settings → Actions → General → *Allow GitHub Actions
  to create and approve pull requests* must be on.

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
