# Scripts

Utility scripts for development and release management.

## bump_version.sh

Automates version bumping across all relevant files.

**Usage**:
```bash
./scripts/bump_version.sh 0.0.2
```

**What it does**:
- Updates version in `pyproject.toml`
- Updates version in `docs/source/conf.py`
- Updates version in `recipe.yaml`
- Provides next steps for committing and tagging

**Full release workflow**:
```bash
# Bump version
./scripts/bump_version.sh 0.0.2

# Review changes
git diff

# Commit and push
git add -A
git commit -m "Bump version to 0.0.2"
git push

# Create and push tag (triggers CD workflow)
git tag v0.0.2
git push origin v0.0.2
```

## Manual Build Testing

### Test PyPI wheel build locally:
```bash
pixi run build-wheel
# Output in dist/

# Test the wheel
pip install dist/pysalient-*.whl
python -c "import pysalient; print(pysalient.__version__)"
```

### Test conda package build locally:
```bash
pixi run build-conda
# Output in output/

# Test with pixi
pixi run python -c "import pysalient; print(pysalient.__version__)"
```
