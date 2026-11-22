#!/bin/bash
# The runs of a session with the base model, one after another, unattended.
#
# On a machine set up by tools/pod_bootstrap.sh, from the clone:
#
#     source ../dsqa_env.sh
#     nohup bash tools/pod_session.sh > ../logs/session.log 2>&1 &
#     tail -f ../logs/session.log
#
# Every run is a process of its own, as an evaluation is: it loads the model,
# pays for that in its first call and leaves the GPU empty for the next. A
# run that fails or hangs does not stop the ones after it. Each is recorded
# in the ledger on the commit the clone is at, with its answers under
# results/runs, so the tree has to be clean when the session starts: a change
# outside results/ and docs/ would make every row one that cannot be quoted.
#
# With names, only those runs, in the order given:
#
#     bash tools/pod_session.sh probe entry-sample
#
#     retrieval      the entry's retrieval over the tuning set, no model: the
#                    laptop's row measured again on this machine
#     probe          what a prompt costs the model, by its length
#     entry-sample   our entry, from its zip, over the 25 sample questions
#     zero-shot      the organizers' zero-shot example over the same
#     rag            the organizers' RAG example over the same
#     entry-tuning   our entry over the 98 questions of the tuning set
#
# For a trial before the session proper, which shows in a few minutes what
# would otherwise show after the model has loaded six times:
#
#     DSQA_LIMIT=2 bash tools/pod_session.sh entry-sample
#
#     DSQA_LIMIT         only the first N questions of a run
#     DSQA_PROBE_SIZES   the sizes of context the probe tries, comma-separated
#
# The rows and the answers are then committed on a branch of their own, to
# be fetched from the laptop and merged there:
#
#     git switch -c pod/$(date -u +%F) && git add results && git commit
#
# The answers are marked afterwards, on the laptop, with tools/judge.py.
set -uo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
work="${DSQA_WORK:-$(dirname "$repo")}"
logs="$work/logs"
python="$repo/.venv/bin/python"
model="${DSQA_BASE_MODEL:-allenai/Olmo-3.1-32B-Instruct}"
# The longest a run may take before it is given up, in seconds.
longest=7200

# As in the evaluation: nothing is fetched, and what is missing fails here.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$work/hf}"

say() { echo "[session] $(date -u +%H:%M:%S) $*"; }

mkdir -p "$logs"
cd "$repo"

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -n 1 || true)"
if [ -z "$gpu" ]; then
    say "no GPU: nothing to run"
    exit 1
fi
# The CPUs a timing depends on are the ones the machine lets us use: a pod
# sees every CPU of its host and is given the time of a few of them, which
# its control group says, in the layout of one version or the other.
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
note="The base model on a rented machine: $gpu, $cpus CPUs where the evaluation's has 32."
sample=(--dataset sample-25 --out results/runs --note "$note")
tuning=(--questions dev/questions/tuning_qa.json --dataset tuning-98)
tuning+=(--out results/runs --note "$note")
limit=()
if [ -n "${DSQA_LIMIT:-}" ]; then
    limit=(--limit "$DSQA_LIMIT")
fi
sizes=()
if [ -n "${DSQA_PROBE_SIZES:-}" ]; then
    sizes=(--sizes "$DSQA_PROBE_SIZES")
fi

run() {
    case "$1" in
        retrieval)
            "$python" -m tools.eval_retrieval --model-id "$model" --system 03_dsqa --record \
                --questions dev/questions/tuning_qa.json \
                --evidence dev/questions/tuning_evidence.json \
                --dataset tuning-98 --corpus dev-150 --note "On a rented machine: $gpu."
            ;;
        probe)
            "$python" -m tools.probe_engine --model-id "$model" --corpus-dir data/corpus \
                --record --corpus dev-150 "${sample[@]}" "${sizes[@]}"
            ;;
        entry-sample)
            "$python" -m tools.run_local 03_dsqa_submission.zip --model-id "$model" \
                --corpus-dir data/corpus --record --corpus dev-150 "${sample[@]}" "${limit[@]}"
            ;;
        zero-shot)
            "$python" -m tools.run_local 01_zero_shot --model-id "$model" \
                --record "${sample[@]}" "${limit[@]}"
            ;;
        rag)
            "$python" -m tools.run_local 02_rag --model-id "$model" \
                --corpus-dir data/corpus --record --corpus dev-150 "${sample[@]}" "${limit[@]}"
            ;;
        entry-tuning)
            "$python" -m tools.run_local 03_dsqa_submission.zip --model-id "$model" \
                --corpus-dir data/corpus --record --corpus dev-150 "${tuning[@]}" "${limit[@]}"
            ;;
        *)
            echo "not a run of the session: $1"
            return 2
            ;;
    esac
}

# The script asked for one run by itself, below: timeout takes a program,
# not a function.
if [ "${1:-}" = "--one" ]; then
    run "$2"
    exit $?
fi

names=("$@")
if [ "${#names[@]}" -eq 0 ]; then
    names=(retrieval probe entry-sample zero-shot rag entry-tuning)
fi

say "$gpu, commit $(git rev-parse --short HEAD), $model"
if ! "$python" -m tools.build_entry > "$logs/build_entry.log" 2>&1; then
    say "the entry's zip could not be built from a clean tree: see $logs/build_entry.log"
    exit 1
fi
say "$(grep sha256 "$logs/build_entry.log" | head -n 1)"

failed=()
for name in "${names[@]}"; do
    say "$name: started, its output in $logs/$name.log"
    began=$SECONDS
    if timeout -k 30 "$longest" bash "$repo/tools/pod_session.sh" --one "$name" \
        > "$logs/$name.log" 2>&1; then
        say "$name: done in $((SECONDS - began))s"
    else
        say "$name: FAILED after $((SECONDS - began))s"
        failed+=("$name")
    fi
    grep -E "^\[(runner|probe)\] .*(answered|projected|peak|loaded in|indexed in)|in_context" \
        "$logs/$name.log" | tail -n 8
done

say "the session is over: ${#names[@]} runs, ${#failed[@]} failed (${failed[*]:-none})"
"$python" -m tools.ledger --last "${#names[@]}"
[ "${#failed[@]}" -eq 0 ]
