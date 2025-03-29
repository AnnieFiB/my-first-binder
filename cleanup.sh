#!/bin/bash

echo "############################################"
echo "#       Project Cleanup Utility (Unix)      #"
echo "############################################"

DIR_COUNT=0
FILE_COUNT=0

echo "[1/7] Cleaning cache directories..."
find . -type d \( -name "__pycache__" -o -name ".ipynb_checkpoints" -o -name ".pytest_cache" \) -exec rm -rf {} + && echo "  ✔ Removed cache directories"

echo "[2/7] Removing Python cache files (.pyc, .pyo)..."
FILE_COUNT=$(find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete -print | wc -l)
echo "  ✔ Removed $FILE_COUNT Python cache files"

echo "[3/7] Removing temporary files (.tmp, ~$*)..."
TMP_COUNT=$(find . -type f \( -name "*.tmp" -o -name "*~" \) -delete -print | wc -l)
echo "  ✔ Removed $TMP_COUNT temporary files"

echo "[4/7] Clearing Jupyter checkpoints..."
CHECKPOINT_COUNT=$(find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + -print | wc -l)
echo "  ✔ Cleared $CHECKPOINT_COUNT Jupyter checkpoints"

echo "[5/7] Removing .devcontainer folder..."
if [ -d ".devcontainer" ]; then
    rm -rf .devcontainer
    echo "  ✔ Removed .devcontainer"
else
    echo "  ℹ️  No .devcontainer folder found"
fi

echo "[6/7] Cleaning .code-workspace and VS Code remote settings..."
if compgen -G "*.code-workspace" > /dev/null; then
    for file in *.code-workspace; do
        sed -i '/remoteAuthority/d' "$file" && echo "  ✔ Cleaned $file"
    done
else
    echo "  ℹ️  No workspace files to clean"
fi

if [ -f ".vscode/settings.json" ]; then
    if ! grep -q "remote.containers.enabled" .vscode/settings.json; then
        sed -i '$s/}/,\n  "remote.containers.enabled": false\n}/' .vscode/settings.json
        echo "  ✔ Added remote.containers.enabled=false to settings.json"
    fi
else
    mkdir -p .vscode
    echo '{ "remote.containers.enabled": false }' > .vscode/settings.json
    echo "  ✔ Created settings.json to disable dev containers"
fi

echo "[7/7] Clearing pip cache..."
if command -v pip &> /dev/null; then
    pip cache purge
    echo "  ✔ Pip cache cleared"
else
    echo "  ⚠️ pip not found — skipping cache purge"
fi

echo
echo "✅ Cleanup complete!"
