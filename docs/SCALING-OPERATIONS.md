# Scaling validation operations

Qualification criteria remain in `SCALING-VALIDATION.md`, frozen in the run's
manifest with all algorithm and input hashes. Operational changes described here
do not alter those criteria.

Final confirmation can run two evaluation children concurrently. Each retains a
3 GiB hard address-space limit; the host reserve remains 2 GiB. Training remains
two independent children with 4 GiB each. Confirmation uses the same checkpoints,
opponents, seed 8200000, 2,000 blocks and six populations. Training CPU overlap
and throughput measurements exclude evaluation batches.

Reports are locked and validated before publication. A completed report is reused
only after provenance validation. Interrupted pending output is preserved under
an attempt-specific filename before retry. A failed child stops its peer; inspect
both logs and lifetime metrics before any resume. Memory, node or table failures
require investigation rather than unchanged retries.

Trainer process exit zero alone does not mean its target was reached: resource
stops also save a checkpoint and return zero. The pair supervisor now checks each
exited trainer's checkpoint stop reason immediately, stops its peer on an early
resource stop, and excludes that pair from successful throughput batches. Wrapper
metrics describe process completion; the checkpoint describes training completion.

The original active supervisor loaded sequential confirmation code at launch.
Leave healthy training running. After both 4,200 checkpoints and all screening
reports are complete, a monitor may reload the supervisor to use parallel
confirmation if useful for the finite deadline. First verify the invocation PID,
command line, status `confirming`, and absence of training children. Send SIGTERM
only to that verified supervisor, wait for it and its evaluation child to exit,
and preserve pending outputs. Verify frozen inputs and complete checkpoints before
running `.venv/bin/python -m scripts.scaling_validation --resume`. Never launch a
second supervisor alongside the first. An investigated natural interruption may
also resume with the updated orchestration.

The invocation deadline remains 2026-11-15T00:00:00Z. If confirmation cannot finish
inside it, report qualification as incomplete; do not extend it or change gates to
obtain favorable statistics. Operational scaling and learning gains have separate
outcomes, and neither establishes competitive strength against strong opponents.
