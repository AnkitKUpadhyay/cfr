# Mask generation

The repository contains two extraction versions. Use the released historical
masks for the original CFR evaluation; the updated extractor produces a
different mask version.

| Version | Entry point | Selection |
|---|---|---|
| Original | `scripts/extract_wildlifedatasets_sam_masks.py` | Annotation-box IoU, or confidence for full-image inputs. See [legacy extraction](legacy_mask_generation.md). |
| Updated | `python -m sam3_masks` | Candidate box contains the input center; largest mask-pixel area, then confidence, then candidate order. |

## Installation

From the repository root, use Python 3.10 or newer in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements-sam3.txt
hf auth login
```

The pinned inference requirements identify the Transformers revision and
PyTorch/torchvision versions used by the extractor. Select PyTorch wheels
compatible with your machine. Obtain access to `facebook/sam3` on Hugging Face
before inference, or provide a downloaded model directory with `--model`.
Model weights are not bundled. Input validation and the synthetic tests need
only `pip install -e .`; they do not load SAM3 or require a GPU.

## Inputs and prompts

Supply a CSV or COCO JSON file. CSV rows require an `annotation_id` and `path`;
use `image_id`, `identity`, and `split` when available. Paths are relative to
`--image-root`. Annotation IDs must be unique.

| `input_kind` | Required geometry | SAM input |
|---|---|---|
| `source_box` | `bbox` as `[x,y,width,height]` in current-image coordinates | Crop to the annotation box before inference. |
| `native_crop` | No current-image box | Entire supplied image. |
| `prepared_crop` | Crop already prepared upstream | Entire supplied image; do not crop a second time. |
| `full_image` | No current-image box | Entire original photograph. |

Without `input_kind`, a box selects `source_box`; otherwise the whole image is
used as `native_crop`. One annotation per image does **not** establish centered
framing. Verify that assumption when using the center rule. Multiple annotations
on the same image need separate source boxes or separately prepared crop files.

CSV boxes may instead use `bbox_x,bbox_y,bbox_w,bbox_h`. Source crop endpoints
use Python `round`, then clipping to image bounds; empty crops are errors.
Images are converted to RGB without automatic EXIF rotation. The supplied boxes
must describe that input orientation. Preserve upstream geometry in
`source_image` and `source_bbox_xywh`; these fields do not alter inference.

Prompt precedence is the same for every dataset:

1. Per-record `sam_prompt` or `prompt` string.
2. A JSON prompt-map match using category name, species, then category ID.
3. The default `--prompt` or manifest `prompts`.

For a mixed-species dataset, provide the correct species prompt for each
annotation through a prompt map or `sam_prompt`. Dataset names do not select
special behavior. Species labels alone do not generate a prompt automatically.
See [prepared crops](../examples/prepared_crops.csv),
[source-box annotations](../examples/source_boxes.json), and the
[prompt map](../examples/prompts.json).

Repeated `--prompt` arguments, or a list in the prompt map, run each prompt in
the supplied order. Candidates are pooled and selected once; ties retain prompt
order and then the processor's candidate order. Use multiple prompts only when
they describe the same intended target.

## Run

Edit the paths and prompts in [examples/jobs.jsonl](../examples/jobs.jsonl),
then validate a job before inference:

```bash
python -m sam3_masks --manifest examples/jobs.jsonl --index 0 --dry-run
python -m sam3_masks --manifest examples/jobs.jsonl --index 0 \
  --device cuda:0 --save-candidate-masks
```

Manifest-relative paths resolve beside the manifest. Direct CLI paths resolve
from the working directory; CLI options override manifest settings. Each job
needs metadata, image root, prompts and a fresh output directory. The CLI also
accepts these directly:

```bash
python -m sam3_masks \
  --metadata annotations.csv --image-root data/animals \
  --dataset MyDataset --prompt antelope \
  --output-dir outputs/my_masks --device cuda:0
```

Use `--limit 10` for a small trial, `--dry-run --report outputs/preflight.json`
for a complete input report, and `--local-files-only` for cached/offline weights.
Existing output artifacts and reports are not overwritten.

## Selection procedure

1. Prepare the annotation input and run SAM3 with its resolved prompts. The
   processor uses **score > 0.3** and **mask probability > 0.5** by default.
   These are SAM processing settings, not another downstream acceptance filter.
2. Discard empty/nonbinary/nonfinite masks, nonfinite scores, and invalid boxes.
   Clip valid boxes to input bounds.
3. Keep boxes containing `(W/2,H/2)` under half-open bounds:
   `x1 <= W/2 < x2` and `y1 <= H/2 < y2`. The center pixel need not be foreground.
4. Reject if no valid central candidate remains. Otherwise choose the greatest
   foreground-pixel count; break equal-area ties by confidence, then original
   candidate index. Coverage is `foreground_pixels / (H*W)`.
5. Save the selected mask unchanged at the full input dimensions.

No coverage threshold, hole filling, speck removal, or automatic off-center
fallback is applied. `selection_bbox` is rejected; selection always refers to
the whole SAM input. For StripeSpotter, use `full_image` with prompt `zebra` and
retain the flank box only as source provenance. A later flank analysis can crop
the image and mask together.

Largest central area is a selection heuristic, not a guarantee of correct target
identity or accurate boundaries. Validate mask quality and rejection behavior on
representative examples before interpreting downstream results.

## Outputs

| File | Contents |
|---|---|
| `annotation_masks_rle.json.gz` | One COCO RLE record per annotation. Accepted masks have `present=true`; rejected/error records have null segmentation. |
| `annotation_match_metadata.jsonl` | IDs, input/crop geometry, RGB hash, prompts, candidates, selected index and outcome. |
| `extraction_summary.json` | Counts, configuration, checkpoint revision, dependency versions, code/input hashes and run status. |
| `candidate_masks/*.npz` | Optional full-input candidate masks, required for the confidence fallback below. |

RLE sizes are `[height,width]`; metadata image sizes are `[width,height]`. Set
the evaluator's `annotation_masks_rle` index field to the exported RLE file and
use matching annotation IDs and input geometry. Raw extraction outputs differ
from the curated HF packaging. Local audit files contain user-supplied metadata
and paths; keep them outside the source release.

Normal rejection statuses are `no_candidate`, `no_valid_candidate`, and
`no_central_candidate`. Input errors are recorded separately and return exit
code 2. Inference failures stop the run and leave `.partial` outputs rather
than publishing a completed mask archive.

## Optional confidence fallback

The updated HappyWhale and NOAARightWhale releases use a separate second pass
for `no_central_candidate` records. All primary acceptances and other rejection
types remain unchanged. The fallback selects valid candidates by **confidence,
then foreground-pixel count, then original index**. It performs no new inference.

The source run must be complete, error-free, and include `--save-candidate-masks`:

```bash
python scripts/apply_confidence_fallback.py \
  --source-dir outputs/my_masks --output-dir outputs/my_masks_fallback
```

The destination must not exist. It receives updated RLE, audit and summary files
plus `fallback_manifest.json`. Candidate files are referenced through a relative
symlink, so retain the original run. This policy is explicit and optional;
validate target selection before applying it to another dataset.

## Repeatability

Selection is deterministic for identical candidates. Inference fixes the seed,
uses FP32 eager attention, enables strict deterministic PyTorch operations and
disables TF32/cuDNN benchmarking. One image/prompt is inferred at a time;
`--batch-size` groups host-side preparation only.

Repeat runs require the same inputs, prompts/order, software, hardware and model
revision. Pin `--revision` to the recorded `resolved_revision`; preserve local
model directories unchanged. Cross-hardware bitwise equality is not promised.

The public package removes superseded target-region selection and dataset-name
guards from the research implementation while retaining its standard selection
rule. Existing HF masks remain unchanged; their saved code hashes describe the
original generating runs.

Run the synthetic tests without SAM3 or model downloads:

```bash
python -m unittest discover -s tests -v
```
