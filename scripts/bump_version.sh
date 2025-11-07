#!/bin/bash
# Script to bump version across all relevant files

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <new_version>"
    echo "Example: $0 0.0.2"
    exit 1
fi

NEW_VERSION=$1

echo "Bumping version to $NEW_VERSION..."

# Update pyproject.toml
echo "Updating pyproject.toml..."
sed -i "s/^version = .*/version = \"$NEW_VERSION\"/" pyproject.toml

# Update docs/source/conf.py if it exists
if [ -f "docs/source/conf.py" ]; then
    echo "Updating docs/source/conf.py..."
    sed -i "s/^release = .*/release = '$NEW_VERSION'/" docs/source/conf.py
fi

# Update recipe.yaml
echo "Updating recipe.yaml..."
sed -i "s/version: .*/version: \"$NEW_VERSION\"/" recipe.yaml

echo ""
echo "Version bumped to $NEW_VERSION in:"
echo "  - pyproject.toml"
echo "  - docs/source/conf.py (if exists)"
echo "  - recipe.yaml"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Commit changes: git add -A && git commit -m 'Bump version to $NEW_VERSION'"
echo "  3. Push changes: git push"
echo "  4. Create tag: git tag v$NEW_VERSION && git push origin v$NEW_VERSION"
