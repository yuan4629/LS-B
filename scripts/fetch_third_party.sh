#!/usr/bin/env bash
# Fetch the third-party code that this repository does not redistribute.
#
#   baselines/BiLoRA/  BiLoRA (no license file upstream), pinned commit
#   brainnet/          brainnet/ package of Brain Decodes Deep Nets (CC BY-NC),
#                      pinned commit, plus third_party/brainnet_plmodel.patch
#
# Usage (from anywhere):  bash scripts/fetch_third_party.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BILORA_URL="https://github.com/yifeiacc/BiLoRA"
BILORA_COMMIT="78ff950f44644bf248dc76531cda73d5f6b1ad57"
BDD_URL="https://github.com/huzeyann/BrainDecodesDeepNets"
BDD_COMMIT="8f16e48cbfb8acb041b3e984ba76bca08027b5e1"

fetch() {  # url commit dir
    local url="$1" commit="$2" dir="$3"
    if [ ! -d "$dir/.git" ]; then
        git clone --quiet --filter=blob:none "$url" "$dir"
    fi
    git -C "$dir" checkout --quiet "$commit"
    echo "$(basename "$dir") at $(git -C "$dir" rev-parse HEAD)"
}

fetch "$BILORA_URL" "$BILORA_COMMIT" "$ROOT/baselines/BiLoRA"
fetch "$BDD_URL" "$BDD_COMMIT" "$ROOT/third_party/BrainDecodesDeepNets"

rm -rf "$ROOT/brainnet"
cp -R "$ROOT/third_party/BrainDecodesDeepNets/brainnet" "$ROOT/brainnet"
(cd "$ROOT" && git apply third_party/brainnet_plmodel.patch)
echo "brainnet/ copied and patched"
