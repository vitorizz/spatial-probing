# Training-question hand-check

Build one HTML file containing the fixed sample's images, object boxes, questions,
proposed answers, and any label evidence. Open the generated file in a browser;
it needs no server or network connection.

```bash
source .venv/bin/activate
python -m src.handcheck --input data/coco_spatial/handcheck_sample.jsonl \
  --image-root drive/data --evidence-dir drive/data/coco_spatial
```

The default output is `handcheck/gallery.html` (ignored by Git). `--out` changes
the location. Generated HTML embeds compressed image previews; the source images
are not modified. Regeneration replaces the HTML but does not delete exported
reviews. Gallery generation uses the existing Pillow, NumPy, and PyYAML requirements.

## Sample status

No fixed 200-question sample or reviewer assignments were supplied in Git or
`drive/`. `tasks.pdf` assigns C the gallery and everyone the review, but names no
owner for sample selection or reviewer allocation. Both are deferred pending
team coordination; this tool does not select a sample or assign reviewers.

## Input format

One JSON object per line, with a unique string `id`, `image_path`, and `label`
or B's `answer`. B's `qtype` is `4way` or `yesno`; omitted types default to
four-way for older gallery inputs. Yes/no records require the original
`question` verbatim. For four-way records, supply `question` verbatim or `o1`/`o2`
to use the shared prompt. Names can also come from `a.name` and `b.name`:

```json
{"id":"coco:123:cat:box:left","image_path":"images/000000000123.jpg","label":"left","a":{"name":"cat","bbox":[20,40,100,90]},"b":{"name":"box","bbox":[250,30,130,140]},"evidence":{"method":"box centers","center_gap":0.4}}
```

Final B records are accepted directly, including negative questions:

```json
{"id":"train_00000","coco_image_id":418684,"image_path":"coco_images/train2017/000000418684.jpg","o1":"chair","o2":"person","relation":"front","qtype":"yesno","question":"Is the chair behind the person? Answer yes or no.","answer":"no"}
```

The expected answer is `answer`, not `relation`: in this example the generator
records `front` but asks about `behind`. Correction menus and restored reviews
use each question's answer options, so binary corrections stay yes/no.

Relative image paths resolve from the JSONL file's folder. Use `--image-root` to
resolve them from another folder (for example, the repo root). Absolute paths
are accepted. Images must decode successfully.

`a` and `b` are optional. If supplied, each has `name` and `bbox`, with boxes in
COCO **pixel** coordinates `[x, y, width, height]` for the supplied image. A is
the first object in the question; B is the reference object. A's box is violet
and B's is cyan. Both the boxes and any extra fields, such as depth estimates,
are visible under **Label evidence**. Optional `evidence` can contain the method,
depth conventions, margins, or thresholds. The page does not infer or invent
depth evidence. Missing evidence is visible and can be flagged during review.

For final B records, `--evidence-dir` joins `pairs_<split>.jsonl` and
`depth_<split>.jsonl`, using the split in `image_path`, COCO image ID/filename,
and object names. It also reads `config_pairs_<split>.yaml` and
`config_depth_<split>.yaml` from that directory. A/B boxes and depths follow
the question's `o1`/`o2` order, even when B swapped the candidate pair. Original
COCO boxes are scaled to the downloaded image size. Saved labeling conventions,
thresholds, and question-record gaps are shown as evidence; nothing is relabeled.
Missing or ambiguous pair matches fail rather than show unrelated boxes.
Without the flag, boxes/evidence are optional and come only from the input rows.

Keep all 200 agreed sample records fixed and agree each person's approximately
70 IDs before reviewing. Building a gallery from a full train/validation file
for development does not designate those records as the team's review sample.

## Reviewing and saving

1. Enter your reviewer name. Use one reviewer per session/export.
2. Check object identity and box placement, then the spatial relation. Inspect
   the supplied depth evidence for front/behind. Use **Unsure** when the image
   or evidence does not support a clear answer.
3. Choose **Accept**, **Reject**, or **Unsure**; add notes and an optional corrected
   answer for a rejected item. Search and decision filters help find items.
4. **Export reviews** downloads JSON containing the reviewer, sample fingerprint,
   and decisions keyed by question ID. Keep this file and share it for team review.
5. **Restore JSON** replaces the current session after validating the entire file.
   It rejects unknown/duplicate IDs and exports from a different sample.

The fingerprint covers questions, labels, original image bytes, boxes, and
evidence. Changes to these require a new review session. Machine-specific image
paths do not affect the fingerprint. Browser storage autosaves when available,
but file-origin storage varies by browser; exported JSON is the portable record.
Restoring is a replacement, not a merge of multiple reviewers' work.

## Verification

```bash
python -m unittest discover -s tests -v
node tests/test_handcheck.js
```

The Python suite uses synthetic images to check embedding, box placement data,
escaping, and sample fingerprints. The JavaScript checks use Node's built-in
test runner, with no npm packages; Node is only needed for those development
checks, not gallery generation or use.
