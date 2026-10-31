# CFR Robustness Experiments

This experiment uses the exact annotation crops, accepted masks, neutral fill,
query/gallery validity rule, MiewID and MegaDescriptor checkpoints, and final
CFR/SSI formulas used for the paper's reference evaluation. Clean embeddings
are read from a separate directory and are not modified.

## Perturbation scope

- Mask perturbations affect query and gallery masks. They test sensitivity to
  errors in the released segmentation layer.
- Image perturbations affect queries only. The gallery remains clean, modeling
  degraded field observations queried against the original reference gallery.
- The full-image view is unchanged by mask perturbations.
- The accepted annotation set and its ordering never change.
- Paper reproduction requires FAISS, matching the reference evaluation. This matters for
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

## Random seeds

Seed 0 is included in `configs/cfr_robustness_conditions.tsv`. We repeat the
five stochastic conditions with seeds 1 and 2 using
`configs/cfr_robustness_extra_seeds.tsv`:

- interior mask false negatives
- exterior mask false positives
- Gaussian noise
- foreground CutOut
- background CutOut

Mask erosion, mask dilation, Gaussian blur, and JPEG compression are
deterministic, so rerunning them with another seed would reproduce the same
input. Launch the two additional seeds with the same submission helper:

```bash
CFR_CONDITIONS_FILE=configs/cfr_robustness_extra_seeds.tsv \
  slurm/submit_cfr_robustness_sweeps.sh low medium high
```

## Outputs

Each condition writes aggregate JSON under:

```text
conditions/<condition-tag>/results/<model>/<dataset>_<model>_robustness.json
```

Condition embeddings may be deleted after successful evaluation to avoid
hundreds of gigabytes of redundant storage. They are reproducible from the
condition manifest and code.

Collect the completed conditions and summarize variation across seeds with:

```bash
python scripts/collect_cfr_robustness.py \
  --baseline_csv /path/to/cfr_summary_reference_k1.csv \
  --results_root /path/to/cfr_robustness \
  --output_dir outputs/cfr_robustness_summary

python scripts/summarize_cfr_robustness_seeds.py \
  --input_dir outputs/cfr_robustness_summary \
  --output_dir outputs/cfr_robustness_summary
```

The output directory contains dataset-level metrics, paired changes from the
clean reference, macro averages and variation across seeds as CSV files.
