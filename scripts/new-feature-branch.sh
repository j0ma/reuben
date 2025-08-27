#!/usr/bin/env bash

set -euo pipefail

should_checkout=${should_checkout:-"yes"}

last_feature_branch=$(git last-feature-branch  | sed 's/^* //')
last_feature_branch_idx=$(echo ${last_feature_branch} | cut -f1 -d-)
last_feature_branch_idx_int=$(echo ${last_feature_branch_idx} | sed 's/^0*//')
n_digits=$(echo ${last_feature_branch_idx} | sed 's/[0-9]/\n&/g' | tail -n +2 | wc -l)
new_idx=$(( ${last_feature_branch_idx_int} + 1 ))
new_idx_padded=$(python -c "print('${new_idx}'.zfill(${n_digits}))")

if [ "$#" -lt 1 ]
then
    new_feature_branch_slug=$(gum input --placeholder "${new_idx_padded}-<slug-goes-here>")
else
    new_feature_branch_slug=$1
fi

new_feature_branch_name="${new_idx_padded}-${new_feature_branch_slug}"

gum confirm "About to create branch '${new_feature_branch_name}'. Confirm?" && (
    echo ""

    if [ "${should_checkout}" = "yes" ]
    then
        git checkout -b ${new_feature_branch_name}
    else
        git branch --verbose ${new_feature_branch_name}
    fi
) 
