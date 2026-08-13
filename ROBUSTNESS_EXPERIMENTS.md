# CFR Robustness Experiments

This experiment uses the exact annotation crops, accepted masks, neutral fill,
query/gallery validity rule, MiewID and MegaDescriptor checkpoints, and final
CFR/SSI formulas used for Table S3. Clean Table S3 embeddings are read from a
separate immutable directory.

## Perturbation scope

- Mask perturbations affect query and gallery masks. They test sensitivity to
  errors in the released segmentation layer.
- Image perturbations affect queries only. The gallery remains clean, modeling
  degraded field observations queried against the original reference gallery.
- The full-image view is unchanged by mask perturbations.
- The accepted annotation set and its ordering never change.
- Production evaluation requires FAISS, matching Table S3. This matters for
  exact ties that occur in some shortcut-only embeddings.

## Mask perturbations

- `mask_erode`: removes the requested fraction of animal-mask pixels nearest
  the boundary.
- `mask_dilate`: adds the same fraction of pixels nearest the boundary.
- `mask_interior_false_negative`: removes a compact region inside the animal.
- `mask_exterior_false_positive`: adds a compact region separated from the
  animal. This is the controlled "background addition" test: unrelated scene
  pixels are incorrectly labeled as foreground.

Mask severity is measured relative to the accepted mask area: 5%, 10%, and 20%.
The extraction summary records the achieved fraction.

## Query-image perturbations

- Gaussian noise with sigma 0.02, 0.05, or 0.10 on the zero-to-one scale.
- Gaussian blur with radius 1, 2, or 4 pixels before model resize.
- JPEG compression at quality 80, 50, or 20.
- Foreground and background localized CutOut at 5%, 10%, or 20% of mask area,
  using the same ImageNet-mean neutral fill as CFR.

Stochastic regions and noise are deterministic per dataset, annotation,
condition, severity, and seed.

## Outputs

Each condition writes aggregate JSON under:

```text
conditions/<condition-tag>/results/<model>/<dataset>_<model>_robustness.json
```

Condition embeddings may be deleted after successful evaluation to avoid
hundreds of gigabytes of redundant storage. They are reproducible from the
condition manifest and code.
