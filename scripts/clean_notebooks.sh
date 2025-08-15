#!/bin/bash

# Check that the script is run in the root of the Git working tree
# TODO: This may not be a kosher way to check. Needs to  be revisited during review
if [ ! -d .git ]; then
    echo "This script must be run in the root of a Git repository."
    exit 1
fi

# Clean all metadata from the Jupyter notebooks committed in the repository
jupyter nbconvert \
  --inplace \
  --ClearMetadataPreprocessor.enabled=True \
  $(git ls-tree --full-tree --name-only -r  HEAD | grep  ".ipynb$")

