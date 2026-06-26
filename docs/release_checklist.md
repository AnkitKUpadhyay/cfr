# Anonymous Release Checklist

Use this checklist before uploading the artifact.

## Include

- This code repository.
- Final result CSVs and LaTeX tables under `results/`.
- An anonymized dataset index with path placeholders or reviewer-accessible
  artifact paths.
- Accepted mask artifacts, if they are uploaded separately and source licenses
  allow sharing of derived masks.
- SAM candidate metadata and prompt metadata, if included as part of the mask
  artifact.

## Do Not Include

- Raw source images unless the original dataset license explicitly allows
  redistribution.
- Absolute local filesystem paths from the authors' machines or clusters.
- Usernames, institution-specific home directories, or private scratch paths.
- Git commit metadata with identifying names or email addresses.
- API tokens, Kaggle credentials, or Hugging Face tokens.

## Recommended Upload Layout

```text
code/
  README.md
  scripts/
  configs/
  slurm/
  results/
  docs/

mask_artifact/
  <dataset>/
    annotation_masks.pkl
    annotation_best_any_iou_masks.pkl
    annotation_match_metadata.jsonl
    image_candidate_metadata.jsonl
    extraction_summary.json
    candidate_masks/
```

The paper should cite the original datasets separately. This release is meant
to reproduce the evaluation layer, not to become a new redistributed image
collection.
