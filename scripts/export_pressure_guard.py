"""Export an opt-in pressure-guard candidate; never edit the source checkpoint."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from poker.equity_policy import PressureGuard
from poker.history_cfr import load
from scripts.live_gate import digest


def run(checkpoint, out, guard=PressureGuard.VERSION):
    if guard not in PressureGuard.VERSIONS:
        raise ValueError('unknown history decision guard')
    out = Path(out)
    if out.exists():
        raise ValueError('preserve existing output')
    source_hash = digest(checkpoint)
    saved = load(checkpoint)
    if saved.get('decision_guard') is not None:
        raise ValueError('source already has a guard')
    saved.update(decision_guard=guard,
                 guard_source_sha256=source_hash)
    if digest(checkpoint) != source_hash:
        raise RuntimeError('input changed')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=out.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(saved, f, allow_nan=False)
        os.link(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink()
    print(json.dumps(dict(checkpoint=str(out), sha256=digest(out),
                          source_sha256=source_hash, guard=guard)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--guard', choices=PressureGuard.VERSIONS, default=PressureGuard.VERSION)
    a = p.parse_args()
    run(a.checkpoint, a.out, a.guard)
