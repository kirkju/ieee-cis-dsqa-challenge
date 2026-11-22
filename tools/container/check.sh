#!/bin/bash
# Inside the container, with the network cut and the disk read-only: the
# checks. Run by tools/container_check.py, from the clone on the volume.
#
#     check.sh [--allow-dirty]
#
# Only /tmp and /out can be written, and both belong to the runner: the one
# for the zip it unpacks and the folder it runs in, the other for the answers.
# The zip that is run over the whole corpus is the one the host built, at
# /host/zip/entry.zip. --allow-dirty says that the clone has files of the
# host's working tree copied over it, and the zip with them.
set -uo pipefail
cd /workspace/ieee-cis-dsqa
python=.venv/bin/python
small="${DSQA_SMALL_MODEL:-unsloth/Qwen3.5-0.8B}"
dirty=()
if [ "${1:-}" = "--allow-dirty" ]; then
    dirty=(--allow-dirty)
fi
failed=()

step() {
    local name="$1"
    shift
    echo
    echo "[check] $name"
    "$@" || failed+=("$name")
}

# The two conditions are checked before anything is concluded from them: a
# container started without them would pass every step below and prove nothing.
no_network() {
    ! "$python" -c "import socket; socket.create_connection(('1.1.1.1', 53), 3)" 2> /dev/null
}
read_only() {
    local place
    for place in / "$HOME" /workspace "$PWD" "$PWD/03_dsqa" "$PWD/data/corpus" "$HF_HOME"; do
        if touch "$place/written-by-the-check" 2> /dev/null; then
            echo "[check] $place can be written"
            return 1
        fi
    done
}
whole_corpus() {
    local out=/tmp/whole.log zip=/host/zip/entry.zip theirs mine
    # The zip that is run is the host's own: the file that would be uploaded.
    if [ ! -f "$zip" ]; then
        echo "[check] no zip from the host at $zip"
        return 1
    fi
    # One with uncommitted files is said to be unfit to upload, which it is,
    # and is run all the same.
    if [ "${#dirty[@]}" -eq 0 ]; then
        "$python" -m tools.build_entry --check "$zip" || return 1
    fi
    # Built here from the same commit, it has to be the same file.
    "$python" -m tools.build_entry "${dirty[@]}" --out /tmp/entry.zip > /tmp/build.log 2>&1
    theirs="$(sha256sum "$zip" | cut -d ' ' -f 1)"
    mine="$(sha256sum /tmp/entry.zip 2> /dev/null | cut -d ' ' -f 1)"
    if [ "$theirs" != "$mine" ]; then
        echo "[check] the zip built here is not the host's: ${mine:-none} and $theirs"
        return 1
    fi
    echo "[check] the host's zip, and the same zip when built here: sha256 $theirs"
    "$python" -m tools.run_local "$zip" --model-id "$small" \
        --corpus-dir data/corpus --ids 3,19,25 --out /out 2>&1 | tee "$out"
    [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
    # Nothing fell back: the embedder and the model loaded, and each question
    # went to the model.
    if grep -E "no embedder|could not be loaded|no index|no time left|not read:" "$out"; then
        return 1
    fi
    grep -q "3 answered, 0 empty, 0 not a string, 0 raised" "$out"
}

echo "[check] $(git rev-parse --short HEAD), $("$python" --version), $(uname -sr)"
step "the network is cut" no_network
step "the disk is read-only" read_only
step "the fast tests" "$python" -m pytest -q -p no:cacheprovider
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "[check] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"
    step "the entry from its zip, with its embedder and $small (the slow test)" \
        env DSQA_STAND_IN_MODEL="$small" "$python" -m pytest -q -p no:cacheprovider -m slow \
        tests/slow/test_entry.py
    step "the same over the whole development corpus" whole_corpus
else
    echo "[check] no GPU in the container: the entry was not run with a real model"
    failed+=("no GPU")
fi

echo
if [ "${#failed[@]}" -gt 0 ]; then
    echo "[check] FAILED:"
    printf '[check]   - %s\n' "${failed[@]}"
    exit 1
fi
echo "[check] all of it passed"
