# CounterFactual Retrieval (CFR)

This repository contains code for **CounterFactual Retrieval for Evidence-Aware Individual Animal Identification**, accepted at **WACV 2027**. It provides mask generation, CFR evaluation, and experiment configurations for individual animal identification. Foreground masks are available on [Hugging Face](https://huggingface.co/datasets/imageomics/cfr-wildlife-masks).

## Install

Use Python 3.10 or newer in a dedicated environment:

```bash
git clone --depth 1 https://github.com/AnkitKUpadhyay/cfr.git
cd cfr
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

This installs the lightweight input validation, mask selection, and export package. For CFR evaluation, also install `python -m pip install -r requirements.txt`. For SAM3 inference, use the separate pinned environment described in [mask generation](docs/mask_generation.md). Source images and model weights are obtained separately.

## Download masks

Two mask versions are available. **Use the historical masks for the original evaluation; updated masks define a different experiment.**

| Version | Purpose | Extractor |
|---|---|---|
| `paper` | Original 34-dataset evaluation mask release | [Legacy SAM3 script](scripts/extract_wildlifedatasets_sam_masks.py) |
| `standardized` | Updated annotation-input selection procedure | [`sam3_masks`](sam3_masks/) |

Download only the datasets you need:

```bash
python -m pip install huggingface_hub
python scripts/download_masks.py --release paper --datasets NyalaData --output-dir data/masks
python scripts/download_masks.py --release standardized --datasets NyalaData --output-dir data/masks
```

The downloader pins verified HF commits, checks file hashes, and separates the two versions. See [mask versions and file formats](docs/mask_releases.md) for dataset lists, coordinates, and source matching. The historical code and artifacts are also preserved under the GitHub tag `paper-code-v1`.

## Generate masks

**Updated extractor:** supply annotation metadata, an image directory, and a species/body-part prompt. Validate inputs before running inference:

```bash
python -m sam3_masks --metadata annotations.csv --image-root images \
  --dataset MyDataset --prompt zebra --dry-run
python -m sam3_masks --metadata annotations.csv --image-root images \
  --dataset MyDataset --prompt zebra --output-dir outputs/masks --device cuda:0
```

Source boxes are cropped before SAM3; prepared crops and full images are used as supplied. Among valid candidates whose boxes contain the input center, selection uses mask pixel count, then confidence, then candidate order. A single annotation does not establish centered framing. Species routing, multiple prompts, optional confidence fallback, and output files are explained in [the updated extraction guide](docs/mask_generation.md).

**Legacy extractor:** the original script, job launcher, and prompt template remain available. Its box-IoU/highest-confidence rules differ from the updated extractor. Follow [legacy extraction](docs/legacy_mask_generation.md); downloading the historical masks is the direct way to use the frozen release.

## Evaluate CFR

Obtain the source images and matching identity/crop metadata, then configure their paths:

```bash
cp configs/cfr_dataset_index.template.tsv configs/cfr_dataset_index.tsv
```

Edit the index to include your datasets, image roots, metadata, and selected mask-version paths. Annotation IDs and image/crop coordinates must agree; downloading masks alone does not supply identity labels. See [dataset acquisition](docs/dataset_acquisition.md) and [evaluation inputs](docs/artifact_manifest.md).

```bash
python scripts/run_cfr_evaluation.py \
  --central_index configs/cfr_dataset_index.tsv --suite all \
  --models miewid megadescriptor --mode extract_eval \
  --output_root outputs/cfr --ks 1 5 10 --device cuda:0
```

Evaluation writes JSON/CSV metrics and local embedding caches. The reference protocol uses accepted-mask annotations with known identities, excludes the query itself from the gallery, and leaves same-image and same-encounter exclusions off. CFR/SSI definitions and silhouette settings are documented with the [evaluation inputs](docs/artifact_manifest.md).

## Additional experiments

- [Mask perturbations and image-corruption robustness](ROBUSTNESS_EXPERIMENTS.md).
- [Foreground-mixed training checkpoint evaluation](experiments/megadescriptor_mix_ablation/README.md).
- Optional cluster launchers in [`slurm/`](slurm/) use editable account, partition, and path placeholders.

Run the lightweight regression suite with `python -m unittest discover -s tests -v`. It checks selection, input handling, exports, fallback, and mask downloads without downloading model weights. GPU inference and retrieval require the separate model dependencies and source data.

## Citation

Please cite CFR and the relevant source datasets/papers, [WildlifeDatasets](https://github.com/WildlifeDatasets/wildlife-datasets), and [SAM3](https://github.com/facebookresearch/sam3).

```bibtex
@inproceedings{upadhyay2027cfr,
  author = {Ankit Kumar Upadhyay and Aleksandr Algasov and Ekaterina Nepovinnykh
            and Fedor Zolotarev and Tuomas Eerola and Heikki Kälviäinen
            and Charles V. Stewart and Lukáš Picek},
  title = {CounterFactual Retrieval for Evidence-Aware Individual Animal Identification},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year = {2027},
  note = {Accepted}
}
```

## License

CFR code is licensed under [MIT](LICENSE). Released masks are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source datasets, SAM3, model weights, and other dependencies retain their respective licenses and terms.

## Acknowledgements

Supported by the [Imageomics Institute](https://imageomics.org), funded by the US National Science Foundation's Harnessing the Data Revolution program under [Award #2118240](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2118240). Opinions, findings, conclusions and recommendations are those of the authors and do not necessarily reflect the views of the National Science Foundation.
