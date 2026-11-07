# Source selection

## Included

- Core model, dataset, training, trigger-optimization, hiding, finetuning, and evaluation utilities.
- Final dataset-specific implementations for CIFAR10/SVHN, ImageNet/RESISC, Camelyon17, UC Merced, and PACS.
- Current FAB, Hibernated, Latent, and NeuBA baseline implementations.
- Current Neural Cleanse, DECREE, Fine-Pruning, MIMIC, DeDe, MARTINI, and CLIP-Inspector implementations.
- PEFT/LoRA finetuning code.
- Camera-ready and rebuttal analyses used by the manuscript: Flip-ASR, paired statistics, PACS semantic shift, overlap, data-scale, proxy-quality, ensemble-size, hyperparameter, and distribution-distance analyses.
- Lightweight optimized-trigger PNGs and their metadata.

`baseline_finetuning_activated_backdoor_v1.py` was the current server entry point for FAB and was copied under the public name `baseline_finetuning_activated_backdoor.py`.

## Excluded

- All model checkpoints and adapters (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, and checkpoint directories).
- All datasets, caches, and downloaded model weights.
- All run directories and intermediate experiment outputs, including raw overlap artifacts.
- Slurm/SBatch scripts, standard output, and error logs.
- Jupyter notebooks, notebook checkpoints, generated figures, manuscript files, and archives.
- The older `poison-then-hide/` source snapshot.
- Files explicitly marked as old, copy, debug, or narrow working variants.
- Duplicate `finetune_generic_mixed_head_ensemble_loop.py` and exploratory mover/partitioned-model branches.
- Earlier PACS trigger-optimization entry points superseded by `pacs_camera_ready_semantic_shift.py`; the older PACS module retained in `src/` is imported as a utility dependency by the final script.

## Deferred remote artifacts

The local mirror does not contain the final tensor-form main triggers. These will be copied from the remote archive after verifying that they correspond to the reported configurations. Raw overlap artifacts are intentionally excluded from this repository.
