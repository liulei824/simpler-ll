#!/usr/bin/env python3
"""Summarize device wall time per rank from a scene-test STRACE log.

The runtime emits one ``chip.run.runner_run.device_wall`` span per device per
round. Round 0 (``inv=1``) is the warm-up and is dropped.
"""

import re
import statistics
import sys

SPAN = re.compile(
    r"inv=(?P<inv>\d+).*?name=chip\.run\.runner_run\.device_wall "
    r"ts=\d+ dur=(?P<dur>\d+).*?dev_id=(?P<dev>\d+)"
)


def rows(path):
    per_dev = {}
    with open(path) as fh:
        for line in fh:
            m = SPAN.search(line)
            if m is None:
                continue
            inv = int(m.group("inv"))
            if inv <= 1:
                continue
            per_dev.setdefault(int(m.group("dev")), []).append(int(m.group("dur")))
    return per_dev


def main():
    for path in sys.argv[1:]:
        per_dev = rows(path)
        for dev in sorted(per_dev):
            vals = sorted(per_dev[dev])
            if not vals:
                continue
            p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
            print(
                f"{path},dev{dev},n={len(vals)},"
                f"median_us={statistics.median(vals) / 1000:.1f},"
                f"p95_us={p95 / 1000:.1f},min_us={vals[0] / 1000:.1f}"
            )


if __name__ == "__main__":
    main()
