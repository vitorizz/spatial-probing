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


def _card(item: dict[str, Any], index: int, options: list[str]) -> str:
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
        f'<option value="{escape(label)}">{escape(label)}</option>' for label in options
    )
    return f'''<article class="card" data-index="{index}" id="item-{index}">
  <div class="card-heading"><span>#{index + 1}</span><code>{escape(item['id'])}</code><strong>{escape(item['label'])}</strong></div>
  <figure><img loading="lazy" src="{item['image_src']}" alt="Image for question {escape(item['id'])}">
    <svg aria-hidden="true" viewBox="0 0 {item['width']} {item['height']}">{''.join(boxes)}</svg></figure>
  <div class="card-body"><p class="question">{escape(item['question'])}</p>
    <p class="objects">{' &nbsp; '.join(descriptions) or 'No object boxes supplied.'}</p>
    <details><summary>Label evidence</summary><pre>{evidence_html}</pre></details>
    <div class="decisions"><label>Decision<select class="verdict">{verdicts}</select></label>
      <label>Corrected relation<select class="corrected-label">{corrections}</select></label></div>
    <label>Notes<textarea class="notes" rows="2" placeholder="Ambiguity, bad boxes, or reason for rejecting"></textarea></label>
  </div></article>'''


def build_gallery(
    source: str | Path, out: str | Path, *, image_root: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render local JSONL records; relative image paths default to the JSONL folder.

    Required: id, image_path, label; question or o1/o2 (or a.name/b.name).
    Optional a/b use COCO pixel boxes. Evidence is displayed without interpreting
    depth conventions. The sample hash includes original image bytes and evidence.
    """
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve():
        raise ValueError("output must differ from the input annotations")
    cfg = load_config() if cfg is None else cfg
    options = cfg["options"]["fourway"]
    if not options or any(not isinstance(option, str) for option in options) or len(set(options)) != len(options):
        raise ValueError("fourway options must be unique strings")
    root = Path(image_root) if image_root is not None else source.parent
    items, seen, images = [], set(), {}
    for line_number, row in read_jsonl(source):
        try:
            identifier = required_text(row, "id")
            if identifier in seen:
                raise ValueError(f"duplicate question id: {identifier}")
            label = required_text(row, "label")
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
            a, b = (_object(row.get(key), image["width"], image["height"]) for key in ("a", "b"))
            if "question" in row:
                question = required_text(row, "question")
            else:
                o1, o2 = row.get("o1") or (a or {}).get("name"), row.get("o2") or (b or {}).get("name")
                names = {"o1": o1, "o2": o2}
                question = cfg["prompts"]["fourway"].format(
                    o1=required_text(names, "o1"), o2=required_text(names, "o2"),
                )
            item = {"id": identifier, "question": question, "label": label,
                    "a": a, "b": b, "evidence": row.get("evidence"), **image}
            json.dumps(item, allow_nan=False)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc
        seen.add(identifier)
        items.append(item)
    manifest = [{key: value for key, value in item.items() if key != "image_src"} for item in items]
    sample_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    payload = {"schema_version": 1, "sample_sha256": sample_hash, "options": options,
               "items": [{key: item[key] for key in ("id", "question", "label")} for item in items]}
    # A literal </script> in source text must not terminate the JSON script element.
    payload_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    cards = "\n".join(_card(item, index, options) for index, item in enumerate(items))
    css = (ASSETS / "gallery.css").read_text(encoding="utf-8")
    js = (ASSETS / "gallery.js").read_text(encoding="utf-8")
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spatial label hand-check</title><style>{css}</style></head><body>
<header><div class="header-inner"><p class="eyebrow">SPATIAL PROBE / TRAINING DATA REVIEW</p>
<h1>Check the relation. Record your decision.</h1>
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
    args = parser.parse_args()
    try:
        result = build_gallery(args.input, args.out, image_root=args.image_root)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
