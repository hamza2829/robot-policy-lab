"""Publish only measurements downloaded from this workflow's completed jobs."""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path("artifacts")
evaluation = json.loads((root / "evaluation.json").read_text())
coverage = json.loads((root / "coverage.json").read_text())["totals"]
suites = ET.parse(root / "junit.xml").getroot().iter("testsuite")
counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
for suite in suites:
    for key in counts:
        counts[key] += int(suite.attrib.get(key, 0))
assert counts["tests"] and not any(counts[key] for key in ("failures", "errors", "skipped"))
url = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
summary = dict(
    workflow_url=url, source_commit=os.environ["GITHUB_SHA"], tests=counts,
    coverage=coverage, evaluation=evaluation,
)
reports = Path("reports")
reports.mkdir(exist_ok=True)
(reports / "final.json").write_text(json.dumps(summary, indent=2) + "\n")
(reports / "environment.txt").write_bytes((root / "environment.txt").read_bytes())
(reports / "manifest.json").write_bytes((root / "manifest.json").read_bytes())
lines = [
    f"Measured by [GitHub Actions]({url}) on source commit `{os.environ['GITHUB_SHA']}`.",
    "",
    f"Tests: **{counts['tests']} passed**, no skipped tests. "
    f"Coverage: **{coverage['percent_covered']:.2f}%** (statements and branches combined).",
    "",
    "| Policy | Successes / episodes | Success rate | 95% Wilson interval | Mean final error |",
    "| :--- | ---: | ---: | ---: | ---: |",
]
for name, item in evaluation["policies"].items():
    low, high = item["wilson_95"]
    lines.append(
        f"| {name} | {item['successes']} / {item['episodes']} | "
        f"{item['success_rate']:.2%} | {low:.1%} to {high:.1%} | "
        f"{item['mean_final_distance']:.4f} m |"
    )
lines += ["", "Observed failures (same evaluation seeds for every policy):", ""]
for name, item in evaluation["policies"].items():
    failed = [row for row in item["outcomes"] if not row["success"]]
    examples = ", ".join(f"{row['seed']} ({row['final_distance']:.3f} m)" for row in failed[:5])
    if failed:
        lines.append(f"- `{name}`: {item['failure_counts']}. Example seeds: {examples}.")
    else:
        lines.append(f"- `{name}`: no failures in these {item['episodes']} episodes.")
lines += [
    "",
    "Full per-seed outcomes, training splits, losses, and coverage totals are in "
    "[reports/final.json](reports/final.json). The workflow artifacts also contain "
    "checkpoints, JUnit XML, HTML coverage, and the exact dependency environment.",
    "",
    "This section and the reports are committed by the workflow after lint, tests, "
    "and the benchmark pass. That documentation commit does not trigger another run. "
    "The source commit above identifies the code actually measured.",
]
readme = Path("README.md")
before, tail = readme.read_text().split("<!-- RESULTS:START -->")
_, after = tail.split("<!-- RESULTS:END -->")
readme.write_text(before + "<!-- RESULTS:START -->\n" + "\n".join(lines)
                  + "\n<!-- RESULTS:END -->" + after)
with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
    output.write("\n".join(lines) + "\n")
