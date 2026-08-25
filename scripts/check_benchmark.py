"""Verify docs/benchmark.html still agrees with the measured results.

Run:  python scripts/check_benchmark.py

The benchmark page is hand-designed, so unlike ``docs/RESULTS.md`` it cannot simply be
regenerated — but it carries the same figures, and a figure typed once is a figure that
goes stale the next time the evaluation runs. This reads the numbers back out of the page
and compares them to ``data/results/eval.json`` and ``data/results/sweep.json``.

It checks the figures the page argues from: the headline statistics, every row of the
results table, the calibration gate and the sweep bounds. It is not a full parse of the
page, and it is not meant to be — it is a tripwire on the numbers a reader would quote.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

EVAL = Path("data/results/eval.json")
SWEEP = Path("data/results/sweep.json")
PAGE = Path("docs/benchmark.html")


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{path} is missing.")
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def stat_block(html: str, label: str) -> tuple[str, str, str]:
    """The value, delta and baseline a headline stat displays."""
    pattern = (
        rf'<div class="label">{re.escape(label)}</div>\s*'
        r'<div class="value">([^<]+)</div>\s*'
        r'<div class="delta"><b class="up">([^<]+)</b> vs baseline ([^<]+)</div>'
    )
    match = re.search(pattern, html)
    if match is None:
        raise SystemExit(f"could not find the '{label}' statistic in {PAGE}")
    return match.group(1), match.group(2), match.group(3)


def main() -> int:
    ev, sw = load(EVAL), load(SWEEP)
    html = PAGE.read_text(encoding="utf-8")

    policies = {p["policy"]: p for p in ev["policies"]}
    base = policies[ev["baseline"]]
    ours = policies["bandit_greedy"]
    summary = sw["summary"]

    failures: list[str] = []

    def expect(what: str, actual: str, wanted: str) -> None:
        if actual.strip() != wanted:
            failures.append(f"{what}: page says {actual.strip()!r}, results say {wanted!r}")

    def expect_present(what: str, wanted: str) -> None:
        if wanted not in html:
            failures.append(f"{what}: {wanted!r} does not appear in the page")

    # --- headline statistics ----------------------------------------------------------
    value, delta, baseline = stat_block(html, "Payments recovered")
    expect("recovered", value, f"{ours['recovery_rate'] * 100:.1f}%")
    expect(
        "recovered delta",
        delta,
        f"+{(ours['recovery_rate'] - base['recovery_rate']) * 100:.1f} pts",
    )
    expect("recovered baseline", baseline, f"{base['recovery_rate'] * 100:.1f}%")

    value, delta, baseline = stat_block(html, "Revenue recovered")
    expect("revenue", value, f"{ours['revenue_recovery_rate'] * 100:.1f}%")
    expect(
        "revenue delta",
        delta,
        f"+{(ours['revenue_recovery_rate'] - base['revenue_recovery_rate']) * 100:.1f} pts",
    )
    expect("revenue baseline", baseline, f"{base['revenue_recovery_rate'] * 100:.1f}%")

    value, delta, baseline = stat_block(html, "Attempts per payment")
    expect("attempts", value, f"{ours['attempts_per_episode']:.2f}")
    expect("attempts baseline", baseline, f"{base['attempts_per_episode']:.2f}")

    value, delta, baseline = stat_block(html, "Wasted attempts")
    expect("wasted", value, f"{ours['wasted_attempt_rate'] * 100:.1f}%")
    expect("wasted baseline", baseline, f"{base['wasted_attempt_rate'] * 100:.1f}%")

    # --- results table ----------------------------------------------------------------
    for name in ("fixed_1_3_5_7", "outreach_only", "aggressive", "taxonomy_aware", "bandit_greedy"):
        p = policies[name]
        row = (
            f"<td>{p['recovery_rate'] * 100:.1f}%</td>"
            f"<td>{p['revenue_recovery_rate'] * 100:.1f}%</td>"
            f"<td>{p['attempts_per_episode']:.2f}</td>"
            f"<td>{p['wasted_attempt_rate'] * 100:.1f}%</td>"
            f"<td>{p['contacts_per_episode']:.2f}</td>"
            f"<td>{p['opt_out_rate'] * 100:.1f}%</td>"
        )
        expect_present(f"results row for {name}", row)

    # --- calibration gate and sweep bounds ---------------------------------------------
    cal = ev["calibration"]
    expect_present(
        "calibration reproduced value",
        f'<div class="gval" style="color:var(--accent)">'
        f"{cal['observed_recovery'] * 100:.1f}%</div>",
    )
    configurations = summary["n_configurations"]
    expect_present("sweep configuration count", f'<div class="gval">{configurations}</div>')
    expect_present("sweep worst uplift", f"+{summary['uplift_min_pp']:.1f}")
    expect_present("sweep best uplift", f"+{summary['uplift_max_pp']:.1f}")

    # --- ablations --------------------------------------------------------------------
    for name in ("ablation_session_blind", "ablation_rail_blind", "ablation_remedy_routed"):
        rate = sw["ablations"][name]["recovery_rate"]
        expect_present(f"ablation figure for {name}", f"<td>{rate * 100:.1f}%</td>")

    if failures:
        print(f"{PAGE} disagrees with the measured results:\n", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        print(
            "\nUpdate the page from data/results/, or re-run the evaluation if the "
            "results are the stale half.",
            file=sys.stderr,
        )
        return 1

    print(f"{PAGE} agrees with data/results/ on every figure checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
