#!/bin/bash
# Sets a rented GPU machine up for a session with the base model.
#
# Run it from a clone of the repository on the machine's volume:
#
#     cd /workspace/ieee-cis-dsqa && bash tools/pod_bootstrap.sh
#
# What it leaves ready, all of it on the volume the clone is on, because a
# pod's own disk is erased when the pod stops:
#
#     .venv/                 Python 3.10 with the evaluation's pinned packages,
#                            and pytest and ruff
#     ../hf/                 the Hugging Face cache: the base model, and a
#                            small model for the slow tests
#     03_dsqa/embed_model    what the entry and the RAG example need next to
#     02_rag/embed_model     their main.py, none of it in the repository
#     02_rag/docling_models
#     data/corpus/           the development corpus: what is open is
#                            downloaded, what is there is checked by its hash
#     ../dsqa_env.sh         the environment of a session, for every shell
#                            that runs the model: source ../dsqa_env.sh
#     ../logs/               what the downloads and the checks printed
#
# It can be run again at any time: each step looks at what is there first.
# After a pod restart only uv itself has to be fetched again.
#
# The base model is 64.5 GB. It downloads in the background while the rest
# is set up, and the script waits for it at the end. The fast tests are run,
# and the slow ones when there is a GPU: the organizers' examples and the
# entry through transformers with the small model, which fails in a minute
# what would otherwise fail after the base model has loaded.
#
# The documents marked "manual" in the manifest cannot be downloaded. They
# are copied from the laptop, and so can the whole corpus be, before or
# after this script:
#
#     scp -P <port> data/corpus/*.pdf root@<host>:/workspace/ieee-cis-dsqa/data/corpus/
#
# On a machine paid by the minute, three settings take out what the first
# session with the base model lost its time to: the packages written to a
# network volume, the tests of a commit that had passed them elsewhere, and
# a corpus fetched from the publishers while the laptop was sending it too.
#
#     DSQA_LOCAL=/root/dsqa DSQA_TESTS=none DSQA_CORPUS=wait bash tools/pod_bootstrap.sh
#
# Settings, all optional:
#
#     DSQA_BASE_MODEL    the model to download; "none" for a rehearsal on a
#                        machine without the room for it
#     DSQA_SMALL_MODEL   the small model of the slow tests
#     DSQA_WORK          where everything goes (the folder the clone is in)
#     DSQA_LOCAL         a folder on the machine's own disk for the virtualenv
#                        and its Python, with a link to the first at .venv. A
#                        pod's volume is a network filesystem, where a small
#                        file takes a hundred times what it takes on the
#                        machine's disk, and the packages are a hundred
#                        thousand small files. The folder is gone when the
#                        pod stops, and this script makes it again
#     DSQA_TESTS         all (the default), fast or none
#     DSQA_CORPUS        download (the default), or wait: the documents are
#                        being copied from the laptop, none is downloaded,
#                        and the script waits until all of them are there
#     DSQA_CORPUS_WAIT   the longest it waits for them, in seconds (1800)
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
work="${DSQA_WORK:-$(dirname "$repo")}"
base_model="${DSQA_BASE_MODEL:-allenai/Olmo-3.1-32B-Instruct}"
small_model="${DSQA_SMALL_MODEL:-unsloth/Qwen3.5-0.8B}"
own="${DSQA_LOCAL:-}"
tests="${DSQA_TESTS:-all}"
corpus="${DSQA_CORPUS:-download}"
corpus_wait="${DSQA_CORPUS_WAIT:-1800}"
logs="$work/logs"
python="$repo/.venv/bin/python"
# What the base model needs on the disk, in GB, with room to spare.
base_model_gb=75

case "$tests" in all | fast | none) ;; *)
    echo "[pod] DSQA_TESTS is all, fast or none, not '$tests'"
    exit 2 ;;
esac
case "$corpus" in download | wait) ;; *)
    echo "[pod] DSQA_CORPUS is download or wait, not '$corpus'"
    exit 2 ;;
esac

export HF_HOME="${HF_HOME:-$work/hf}"
export UV_CACHE_DIR="$work/.uv/cache"
export UV_PYTHON_INSTALL_DIR="$work/.uv/python"
# uv's own Python, kept on the volume: one found on the pod's disk would be
# gone, and the virtualenv with it, when the pod is replaced by another.
export UV_PYTHON_PREFERENCE=only-managed
export PATH="$work/.uv/bin:$PATH"
if [ -n "$own" ]; then
    # Asked for on the machine's own disk, the two go there together. uv's
    # cache stays out of it: it is on the volume, and nothing in it can be
    # linked into another disk.
    export UV_PYTHON_INSTALL_DIR="$own/python"
    export UV_NO_CACHE=1
fi

problems=()
say() { echo "[pod] $*"; }

mkdir -p "$logs" "$HF_HOME"
cd "$repo"

# --- the machine --------------------------------------------------------------
gpu=""
if command -v nvidia-smi > /dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2> /dev/null | head -n 1 || true)"
fi
# A pod is a container: it sees the CPUs and the memory of its host, and what
# it may use of them is in its control group. One that reports 96 CPUs and
# 504 GB was given the time of 7.65 CPUs and 47 GB. The group is read in
# the layout of either version: both were met on rented machines.
group=/sys/fs/cgroup
cpus="$(nproc)"
quota="" period=""
if [ -r "$group/cpu.max" ]; then
    read -r quota period < "$group/cpu.max"
elif [ -r "$group/cpu/cpu.cfs_quota_us" ]; then
    quota="$(cat "$group/cpu/cpu.cfs_quota_us")" period="$(cat "$group/cpu/cpu.cfs_period_us")"
fi
case "$quota" in '' | max | -*) ;; *)
    cpus="$(awk -v q="$quota" -v p="$period" 'BEGIN {printf "%.4g", q / p}')" ;;
esac
memory="$(awk '/MemTotal/ {printf "%.0f", $2 / 1048576}' /proc/meminfo)"
limit="$(cat "$group/memory.max" 2> /dev/null || cat "$group/memory/memory.limit_in_bytes" 2> /dev/null || true)"
# A group with no limit says "max", or a number larger than any machine.
case "$limit" in '' | max | ???????????????*) ;; *)
    memory="$(awk -v m="$limit" 'BEGIN {printf "%.0f", m / 1073741824}')" ;;
esac
free_gb="$(df -BG --output=avail "$work" | tail -n 1 | tr -dc '0-9')"
say "GPU: ${gpu:-none}"
say "${cpus} CPUs, ${memory} GB of memory, ${free_gb} GB free in $work"
say "commit $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

# --- Python and the pinned packages -------------------------------------------
if ! command -v uv > /dev/null 2>&1; then
    say "installing uv in $work/.uv/bin"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$work/.uv/bin" sh
fi
if [ -n "$own" ]; then
    # The virtualenv on the machine's own disk, and a link to it where every
    # tool looks for it. git does not look inside a virtualenv, which says so
    # itself, but it would list the link as a new file and the tree would no
    # longer be clean: the link is kept out of its sight, in this clone alone.
    if [ ! -x "$own/venv/bin/python" ]; then
        uv venv --clear --python 3.10 "$own/venv"
    fi
    if [ "$(readlink "$repo/.venv" 2> /dev/null || true)" != "$own/venv" ]; then
        rm -rf "$repo/.venv"
        ln -s "$own/venv" "$repo/.venv"
    fi
    exclude="$(git rev-parse --git-path info/exclude)"
    mkdir -p "$(dirname "$exclude")"
    grep -qx '/.venv' "$exclude" 2> /dev/null || echo '/.venv' >> "$exclude"
    say "the virtualenv is at $own/venv"
elif [ ! -x "$python" ]; then
    uv venv --clear --python 3.10 "$repo/.venv"
fi
say "installing the pinned packages: the list in $logs/install.log"
uv pip install --python "$python" \
    -r "$repo/requirements-participant.txt" -r "$repo/requirements-dev.txt" 2>&1 \
    | tee "$logs/install.log" \
    | { grep -E "^(Resolved|Prepared|Installed|Checked|Audited|warning|error)" || true; }
say "$("$python" --version), $("$python" -c 'import torch; print("torch", torch.__version__)')"

# --- the models ---------------------------------------------------------------
cached() { [ -d "$HF_HOME/hub/models--${1//\//--}" ]; }

base_pid=""
if [ "$base_model" = none ]; then
    say "no base model asked for"
elif ! cached "$base_model" && [ "$free_gb" -lt "$base_model_gb" ]; then
    problems+=("$base_model was not downloaded: it needs $base_model_gb GB and $work has $free_gb")
else
    say "downloading $base_model in the background: tail -f $logs/base_model.log"
    "$repo/.venv/bin/hf" download "$base_model" > "$logs/base_model.log" 2>&1 &
    base_pid=$!
fi

if [ "$tests" = all ]; then
    say "the small model of the slow tests, $small_model"
    "$repo/.venv/bin/hf" download "$small_model" > "$logs/small_model.log" 2>&1 \
        || problems+=("$small_model did not download: see $logs/small_model.log")
fi

if [ ! -f 03_dsqa/embed_model/modules.json ]; then
    say "the entry's embedding model"
    "$python" -c "from tools import build_entry; build_entry.fetch_embedder()" \
        > "$logs/embedder.log" 2>&1 \
        || problems+=("the embedding model did not download: see $logs/embedder.log")
fi
if [ ! -d 02_rag/embed_model ] && [ -d 03_dsqa/embed_model ]; then
    # The RAG example's embedder is the same model, saved the same way.
    cp -r 03_dsqa/embed_model 02_rag/embed_model
fi
if [ ! -d 02_rag/docling_models ]; then
    say "the RAG example's layout model"
    "$repo/.venv/bin/docling-tools" models download layout -o 02_rag/docling_models \
        > "$logs/docling.log" 2>&1 \
        || problems+=("Docling's layout model did not download: see $logs/docling.log")
fi

# --- the development corpus ---------------------------------------------------
# The documents the manifest has a hash for are the corpus. One of them that
# is not there was not downloaded, or is one of those copied by hand.
lacking() {
    "$python" -c '
from tools import corpus
rows = corpus.load_manifest()
print(" ".join(row["doc_id"] for row in rows if row["sha256"] and not corpus.pdf_path(row).exists()))
'
}
fetch=()
if [ "$corpus" = wait ]; then
    # A document being copied may be there before all of it is: the wait is
    # over when none is lacking and each is the file the manifest describes.
    say "the development corpus: waiting for it to be copied here, ${corpus_wait}s at most"
    began=$SECONDS
    until { [ -z "$(lacking)" ] && "$python" -m tools.corpus --no-download > /dev/null 2>&1; } \
        || [ $((SECONDS - began)) -ge "$corpus_wait" ]; do
        sleep 10
    done
    say "waited $((SECONDS - began))s for it"
    fetch=(--no-download)
else
    say "the development corpus"
fi
"$python" -m tools.corpus "${fetch[@]}" 2>&1 | tee "$logs/corpus.log" \
    || problems+=("the corpus: a download failed or a file is not the manifest's")
missing="$(lacking)"
if [ -n "$missing" ]; then
    problems+=("the corpus lacks, to be copied from the laptop: $missing")
fi

# --- the checks ---------------------------------------------------------------
if [ "$tests" = none ]; then
    say "no tests: none asked for"
else
    say "the fast tests"
    "$python" -m pytest -q 2>&1 | tee "$logs/tests_fast.log" | tail -n 15 \
        || problems+=("the fast tests failed: see $logs/tests_fast.log")
fi
if [ "$tests" != all ]; then
    [ "$tests" = none ] || say "the slow tests are left out: not asked for"
elif [ -n "$gpu" ]; then
    say "the slow tests, with $small_model on the GPU"
    DSQA_STAND_IN_MODEL="$small_model" "$python" -m pytest -q -m slow 2>&1 \
        | tee "$logs/tests_slow.log" | tail -n 15 \
        || problems+=("the slow tests failed: see $logs/tests_slow.log")
else
    say "no GPU: the slow tests are left out"
fi

# --- the base model, and what a session needs ---------------------------------
if [ -n "$base_pid" ]; then
    say "waiting for $base_model"
    if wait "$base_pid"; then
        say "$base_model: $(du -sh "$HF_HOME/hub/models--${base_model//\//--}" | cut -f 1)"
    else
        problems+=("$base_model did not download: see $logs/base_model.log")
    fi
fi

cat > "$work/dsqa_env.sh" << EOF
# The environment of a session with the base model: source $work/dsqa_env.sh
export HF_HOME="$HF_HOME"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR"
export UV_PYTHON_PREFERENCE=only-managed
export PATH="$work/.uv/bin:\$PATH"
EOF

if [ "${#problems[@]}" -gt 0 ]; then
    say "NOT READY:"
    for problem in "${problems[@]}"; do
        say "  - $problem"
    done
    exit 1
fi
say "ready. The runs of a session: source $work/dsqa_env.sh && bash tools/pod_session.sh"
