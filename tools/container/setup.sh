#!/bin/bash
# Inside the container, with the network: a clone of the repository on the
# volume, set up as a rented pod is. Run by tools/container_check.py.
#
#     setup.sh COMMIT [FILE ...]
#
# /host/repo is the repository on the host, read-only. The clone is put at
# its HEAD, which has to be COMMIT. Each FILE is then copied over the clone
# from the host's working tree, to try a change before it is committed. The
# development corpus is copied in as it is copied to a pod, and
# tools/pod_bootstrap.sh does the rest, without the base model.
set -euo pipefail
commit="$1"
shift
clone=/workspace/ieee-cis-dsqa

git config --global --add safe.directory '*'
git config --global advice.detachedHead false
if [ ! -d "$clone/.git" ]; then
    git clone --quiet --no-checkout /host/repo "$clone"
fi
cd "$clone"
git fetch --quiet /host/repo HEAD
git checkout --quiet --force --detach FETCH_HEAD
git clean --quiet -fd
if [ "$(git rev-parse HEAD)" != "$commit" ]; then
    echo "[setup] the clone is at $(git rev-parse HEAD), not at $commit"
    exit 1
fi
for file in "$@"; do
    mkdir -p "$(dirname "$file")"
    cp "/host/repo/$file" "$file"
    echo "[setup] from the host's working tree: $file"
done

mkdir -p data/corpus
cp -n /host/repo/data/corpus/*.pdf data/corpus/
echo "[setup] $(find data/corpus -name '*.pdf' | wc -l) PDFs in data/corpus"

DSQA_BASE_MODEL=none bash tools/pod_bootstrap.sh
