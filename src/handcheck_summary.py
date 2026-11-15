"""Combine the reviewers' hand-check exports into one summary.

    python -m src.handcheck_summary

Reads results/milestone/handcheck/handcheck_<reviewer>.json (exported from the gallery) and the
fixed sample, counts accept/unsure/reject overall, per axis, per question type and per reviewer,
and writes results/milestone/handcheck/summary.json. Each question counts once, for the reviewer
it was assigned to in the sample.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path("results/milestone")
SAMPLE = ROOT / "data" / "handcheck_sample.jsonl"
EXPORTS = sorted((ROOT / "handcheck").glob("handcheck_*.json"))
VERDICTS = ("accept", "unsure", "reject")


def main() -> None:
    sample = [json.loads(line) for line in open(SAMPLE, encoding="utf-8")]
    by_id = {q["id"]: q for q in sample}
    position = {q["id"]: i + 1 for i, q in enumerate(sample)}

    reviews, hashes = {}, set()
    for path in EXPORTS:
        export = json.load(open(path, encoding="utf-8"))
        hashes.add(export["sample_sha256"])
        reviewer = path.stem.split("_")[-1]
        for r in export["reviews"]:
            if r["verdict"] in VERDICTS and by_id[r["id"]]["reviewer"] == reviewer:
                reviews[r["id"]] = r
    assert len(hashes) == 1, "exports come from different samples"
    missing = [q["id"] for q in sample if q["id"] not in reviews]
    assert not missing, f"{len(missing)} questions have no verdict from their assigned reviewer"

    def count(rows: list[dict]) -> dict:
        c = Counter(reviews[q["id"]]["verdict"] for q in rows)
        return {"n": len(rows), **{v: c[v] for v in VERDICTS}, "accept_rate": round(c["accept"] / len(rows), 3)}

    groups = {"all": sample}
    for key in ("axis", "qtype", "reviewer"):
        for value in sorted({q[key] for q in sample}):
            groups[f"{key}/{value}"] = [q for q in sample if q[key] == value]
    summary = {name: count(rows) for name, rows in groups.items()}
    summary["flagged"] = [
        {"position": position[q["id"]], "id": q["id"], "reviewer": q["reviewer"], "axis": q["axis"],
         "qtype": q["qtype"], "question": q["question"], "answer": q["answer"],
         "verdict": reviews[q["id"]]["verdict"], "corrected_label": reviews[q["id"]]["corrected_label"],
         "notes": reviews[q["id"]]["notes"]}
        for q in sample if reviews[q["id"]]["verdict"] != "accept"
    ]

    out = ROOT / "handcheck" / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"{'group':<18}{'n':>5}{'accept':>8}{'unsure':>8}{'reject':>8}{'accept %':>10}")
    for name, s in summary.items():
        if name != "flagged":
            print(f"{name:<18}{s['n']:>5}{s['accept']:>8}{s['unsure']:>8}{s['reject']:>8}{100 * s['accept_rate']:>9.1f}")
    print(f"\n{len(summary['flagged'])} flagged questions -> {out}")


if __name__ == "__main__":
    main()
