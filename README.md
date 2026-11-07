# Poison-then-Hide

This repository contains the research code for Poison-then-Hide. It is intentionally separate from the lab working directory and contains no datasets, model checkpoints, training outputs, Slurm/SBatch files, notebooks, or manuscript-building files.

The source selection favors the newest working implementations in `src/`. The older `poison-then-hide/` snapshot, files labeled `old`, `Copy`, `debug`, or `work_*`, duplicate loop scripts, and exploratory model-mover branches were not copied.

## Layout

- `src/`: core attack stages, datasets, evaluation, baselines, defenses, PEFT, and the final rebuttal/camera-ready experiments.
- `artifacts/optimized_triggers/`: lightweight optimized-trigger visualizations and metadata available in the local mirror.
- `SOURCE_SELECTION.md`: inclusion and exclusion rationale.

## Running the code

The code currently uses flat local imports, so run entry points from `src/`:

```bash
python -m pip install -r ../requirements.txt
python trigger_optimization.py --help
python finetune_generic_mixed.py --help
python run_sabotage_and_finetune.py --help
python run_finetune.py --help
```

Dataset paths are command-line arguments and default to locations below `./data/`. No datasets are included.

## Planned additions and validation

The repository currently includes the main source code and lightweight trigger visualizations. We plan to add:

1. The final tensor-form optimized triggers (`trigger_patch.pt`) from the remote experiment archive. The local mirror contains the corresponding PNG visualizations and metadata but not the main RESISC and Camelyon17 tensors.
2. Final PACS reproduction commands and results after the camera-ready runs complete.

Before tagging an archival release, we will also test the documented commands on a clean machine and complete the third-party provenance review.

## License

The authors' original code is released under the [MIT License](LICENSE). Third-party software, datasets, pretrained models, and other external assets remain subject to their respective licenses and terms.
