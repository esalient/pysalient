# GitHub Actions Workflows

This directory contains the CI/CD workflows for pySALIENT.

## Workflows

### ci.yml - Continuous Integration

**Triggers**: Push to main, Pull Requests, Version tags

**Jobs**:
- `build`: Runs linting and tests using pixi
- `update-version`: Updates version in files when a tag is pushed

### cd.yml - Continuous Deployment

**Triggers**: Version tags (v*.*.*), Manual dispatch

**Jobs**:

1. **publish-pypi**: Builds and publishes to PyPI
   - Uses pixi for environment management
   - Builds wheel and sdist with `python -m build`
   - Publishes to PyPI using trusted publishing (no tokens needed)
   - Optional: Publishes to Test PyPI for manual runs

2. **build-conda**: Builds conda packages
   - Builds on Linux, macOS, and Windows
   - Uses `pixi build` to create conda packages
   - Uploads packages as GitHub artifacts

3. **create-release**: Creates GitHub Release
   - Attaches conda packages
   - Auto-generates release notes

## Setup Requirements

### PyPI Trusted Publishing

Configure at https://pypi.org/manage/account/publishing/:
- **Project**: pysalient
- **Owner**: esalient
- **Repository**: pysalient
- **Workflow**: cd.yml

### Optional: Test PyPI Token

Add as repository secret `TEST_PYPI_API_TOKEN` for testing releases.

## Usage

### Publishing a Release

```bash
# 1. Update version in pyproject.toml
sed -i 's/version = ".*"/version = "0.0.2"/' pyproject.toml

# 2. Commit and tag
git add pyproject.toml
git commit -m "Bump version to 0.0.2"
git push

# 3. Create and push tag
git tag v0.0.2
git push origin v0.0.2
```

### Manual Testing

Go to Actions → "Publish to PyPI and Build Conda Package" → "Run workflow" → Enable dry-run mode

## Security Notes

- No API tokens are stored in the repository
- PyPI publishing uses OIDC trusted publishing
- All secrets are managed through GitHub Secrets
- Workflows follow GitHub security best practices

For detailed publishing instructions, see [PUBLISHING.md](../../PUBLISHING.md).
