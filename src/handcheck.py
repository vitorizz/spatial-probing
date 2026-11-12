"""Build an offline, self-contained gallery for reviewing spatial training labels."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from src.common import REPO_ROOT, load_config
from src.data._common import read_jsonl, required_text

ASSETS = REPO_ROOT / "handcheck"


def _object(value: Any, width: int, height: int) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("objects a/b must be dictionaries with name and bbox")
    name = required_text(value, "name")
    box = value.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("bbox must be COCO pixel coordinates [x, y, width, height]")
    if any(type(number) not in (int, float) or not math.isfinite(number) for number in box):
        raise ValueError("bbox coordinates must be finite numbers")
    x, y, w, h = box
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError(f"bbox {box} lies outside the {width}x{height} image")
    # Preserve extra evidence fields such as depth; reject non-JSON values/NaN.
    json.dumps(value, allow_nan=False)
    return {**value, "name": name, "bbox": box}


class _Evidence:
    """Join B's saved pair/depth artifacts without importing the generator runtime."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.splits: dict[str, tuple] = {}

    def for_question(self, row: dict[str, Any], width: int, height: int) -> dict[str, Any]:
        image_path = Path(required_text(row, "image_path"))
        split = image_path.parent.name
        if split not in {"train2017", "val2017"}:
            raise ValueError("evidence join requires image paths containing train2017 or val2017")
        if split not in self.splits:
            pairs = {}
            for _, pair in read_jsonl(self.root / f"pairs_{split}.jsonl"):
                key = (pair["coco_image_id"], pair["file_name"],
                       tuple(sorted((pair["a"]["name"], pair["b"]["name"]))))
                if key in pairs:
                    raise ValueError(f"ambiguous pair evidence: {key}")
                pairs[key] = pair
            depths = {}
            for _, record in read_jsonl(self.root / f"depth_{split}.jsonl"):
                if record["coco_image_id"] in depths:
                    raise ValueError(f"duplicate depth evidence: {record['coco_image_id']}")
                depths[record["coco_image_id"]] = record["depth"]
            pair_cfg = load_config(self.root / f"config_pairs_{split}.yaml")["pairs"]
            depth_cfg = load_config(self.root / f"config_depth_{split}.yaml")["depth"]
            self.splits[split] = pairs, depths, pair_cfg, depth_cfg
        pairs, depths, pair_cfg, depth_cfg = self.splits[split]
        names = (required_text(row, "o1"), required_text(row, "o2"))
        key = (row["coco_image_id"], image_path.name, tuple(sorted(names)))
        if names[0] == names[1] or key not in pairs:
            raise ValueError(f"no unique pair evidence for question {row['id']}")
        pair = pairs[key]
        depth = depths.get(row["coco_image_id"], {})
        objects = {obj["name"]: obj for obj in (pair["a"], pair["b"])}
        joined = {}
        for field, name in zip(("a", "b"), names):
            obj = _object(objects[name], pair["width"], pair["height"])
            sx, sy = width / pair["width"], height / pair["height"]
            # Scale from original COCO coordinates to the downloaded image, whose
            # dimensions may have been rounded independently during resizing.
            box = [value * scale for value, scale in zip(obj["bbox"], (sx, sy, sx, sy))]
            # Avoid a rounding-only overflow at the right/bottom image edge.
            box[2], box[3] = min(box[2], width - box[0]), min(box[3], height - box[1])
            joined[field] = {**obj, "bbox": box, "depth": depth.get(str(obj["ann_id"]))}
        joined["evidence"] = {
            "question_record": {key: row[key] for key in
                                ("axis", "relation", "cx_gap", "depth_gap") if key in row},
            "original_image_size": [pair["width"], pair["height"]],
            "pairs_config": pair_cfg, "depth_config": depth_cfg,
        }
        if row.get("evidence") is not None:
            joined["evidence"]["supplied_evidence"] = row["evidence"]
        return {**row, **joined}


def _card(item: dict[str, Any], index: int) -> str:
    escape = lambda value: html.escape(str(value), quote=True)
    boxes, descriptions = [], []
    for key in ("a", "b"):
        obj = item[key]
        if obj is None:
            continue
        x, y, w, h = obj["bbox"]
        boxes.append(f'<rect class="box-{key}" x="{x}" y="{y}" width="{w}" height="{h}"/>')
        descriptions.append(f'<span class="object-{key}">{key.upper()}: {escape(obj["name"])}</span>')
    evidence = {key.upper(): item[key] for key in ("a", "b") if item[key] is not None}
    if item["evidence"] is not None:
        evidence["label_evidence"] = item["evidence"]
    evidence_html = escape(json.dumps(evidence, indent=2, ensure_ascii=False)) if evidence else "No boxes or label evidence supplied."
    verdicts = "".join(f'<option value="{key}">{name}</option>' for key, name in (
        ("unreviewed", "Unreviewed"), ("accept", "Accept"), ("reject", "Reject"), ("unsure", "Unsure"),
    ))
    corrections = '<option value="">No correction</option>' + "".join(
        f'<option value="{escape(label)}">{escape(label)}</option>' for label in item["options"]
    )
    return f'''<article class="card" data-index="{index}" id="item-{index}">
  <div class="card-heading"><span>#{index + 1}</span><code>{escape(item['id'])}</code><strong>{escape(item['label'])}</strong></div>
  <figure><img loading="lazy" src="{item['image_src']}" alt="Image for question {escape(item['id'])}">
    <svg aria-hidden="true" viewBox="0 0 {item['width']} {item['height']}">{''.join(boxes)}</svg></figure>
  <div class="card-body"><p class="question">{escape(item['question'])}</p>
    <p class="objects">{' &nbsp; '.join(descriptions) or 'No object boxes supplied.'}</p>
    <details><summary>Label evidence</summary><pre>{evidence_html}</pre></details>
    <div class="decisions"><label>Decision<select class="verdict">{verdicts}</select></label>
      <label>Corrected answer<select class="corrected-label">{corrections}</select></label></div>
    <label>Notes<textarea class="notes" rows="2" placeholder="Ambiguity, bad boxes, or reason for rejecting"></textarea></label>
  </div></article>'''


def build_gallery(
    source: str | Path, out: str | Path, *, image_root: str | Path | None = None,
    cfg: dict[str, Any] | None = None, evidence_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Render local JSONL records; relative image paths default to the JSONL folder.

    Required: id, image_path, label or B's answer; question or o1/o2 (or a/b names).
    B's qtype selects yes/no or four-way options; yes/no questions are verbatim.
    Optional a/b use COCO pixel boxes. Evidence is displayed without interpreting
    depth conventions. The sample hash includes original image bytes and evidence.
    """
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve():
        raise ValueError("output must differ from the input annotations")
    cfg = load_config() if cfg is None else cfg
    option_sets = {"4way": cfg["options"]["fourway"], "yesno": cfg["options"]["yesno"]}
    for kind, options in option_sets.items():
        if not options or any(not isinstance(option, str) for option in options) or len(set(options)) != len(options):
            raise ValueError(f"{kind} options must be unique strings")
    root = Path(image_root) if image_root is not None else source.parent
    evidence = _Evidence(evidence_dir) if evidence_dir is not None else None
    items, seen, images = [], set(), {}
    for line_number, row in read_jsonl(source):
        try:
            identifier = required_text(row, "id")
            if identifier in seen:
                raise ValueError(f"duplicate question id: {identifier}")
            kind = row.get("qtype", "4way")
            if not isinstance(kind, str) or kind not in option_sets:
                raise ValueError("qtype must be '4way' or 'yesno'")
            options = option_sets[kind]
            label = required_text(row, "label" if "label" in row else "answer")
            if "label" in row and "answer" in row and row["answer"] != label:
                raise ValueError("label and answer disagree")
            if label not in options:
                raise ValueError(f"label must be one of {options}")
            image_path = Path(required_text(row, "image_path"))
            image_path = (root / image_path).resolve()
            if image_path not in images:
                image_bytes = image_path.read_bytes()
                with Image.open(io.BytesIO(image_bytes)) as image:
                    width, height = image.size
                    thumbnail = image.convert("RGB")
                    thumbnail.thumbnail((1024, 1024))
                    buffer = io.BytesIO()
                    thumbnail.save(buffer, format="JPEG", quality=85)
                images[image_path] = {
                    "width": width, "height": height,
                    "image_src": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                }
            image = images[image_path]
            if evidence is not None:
                row = evidence.for_question(row, image["width"], image["height"])
            a, b = (_object(row.get(key), image["width"], image["height"]) for key in ("a", "b"))
            if "question" in row:
                question = required_text(row, "question")
            else:
                if kind == "yesno":
                    raise ValueError("yesno records require their original question")
                o1, o2 = row.get("o1") or (a or {}).get("name"), row.get("o2") or (b or {}).get("name")
                names = {"o1": o1, "o2": o2}
                question = cfg["prompts"]["fourway"].format(
                    o1=required_text(names, "o1"), o2=required_text(names, "o2"),
                )
            item = {"id": identifier, "question": question, "label": label, "options": options,
                    "a": a, "b": b, "evidence": row.get("evidence"), **image}
            json.dumps(item, allow_nan=False)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc
        seen.add(identifier)
        items.append(item)
    manifest = [{key: value for key, value in item.items() if key != "image_src"} for item in items]
    sample_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    payload = {"schema_version": 1, "sample_sha256": sample_hash,
               "items": [{key: item[key] for key in ("id", "question", "label", "options")} for item in items]}
    # A literal </script> in source text must not terminate the JSON script element.
    payload_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    cards = "\n".join(_card(item, index) for index, item in enumerate(items))
    css = (ASSETS / "gallery.css").read_text(encoding="utf-8")
    js = (ASSETS / "gallery.js").read_text(encoding="utf-8")
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spatial label hand-check</title><style>{css}</style></head><body>
<header><div class="header-inner"><p class="eyebrow">SPATIAL PROBE / TRAINING DATA REVIEW</p>
<h1>Check the answer. Record your decision.</h1>
<p>Check object identity, box placement, and whether the proposed answer is visually clear.
Mark ambiguity as unsure. Inspect depth evidence for front/behind labels.</p>
<div class="toolbar"><label>Reviewer<input id="reviewer" placeholder="Your name" autocomplete="name"></label>
<label>Show<select id="status-filter"><option value="all">All questions</option><option value="unreviewed">Unreviewed</option>
<option value="accept">Accepted</option><option value="reject">Rejected</option><option value="unsure">Unsure</option></select></label>
<label>Find<input id="search" type="search" placeholder="Question ID, text, or relation"></label>
<button id="export" type="button">Export reviews</button>
<label class="import-button">Restore JSON<input id="import" type="file" accept="application/json,.json"></label></div>
<p id="progress" aria-live="polite"></p><p id="message" role="status"></p>
<p class="hint">Export a JSON copy to share or safeguard your work. Restore replaces this session.
Browser storage is a convenience; keep exports. Use one reviewer per session.</p>
<p class="hint">Sample: {html.escape(source.name)} · {len(items)} questions · <code>{sample_hash[:12]}</code></p>
</div></header><main>{cards}<p id="empty" hidden>No questions match these filters.</p></main>
<script id="sample-data" type="application/json">{payload_json}</script><script>{js}</script>
</body></html>'''
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return {"n": len(items), "n_images": len(images), "sample_sha256": sample_hash, "output": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="fixed sample JSONL")
    parser.add_argument("--out", type=Path, default=ASSETS / "gallery.html", help="output HTML file")
    parser.add_argument("--image-root", type=Path, help="base for relative image paths (default: JSONL folder)")
    parser.add_argument("--evidence-dir", type=Path, help="B's saved pairs/depth JSONL and config snapshots")
    args = parser.parse_args()
    try:
        result = build_gallery(args.input, args.out, image_root=args.image_root, evidence_dir=args.evidence_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
