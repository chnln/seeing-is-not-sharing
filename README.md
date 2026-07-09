# Seeing Is Not Sharing

Code and prompt templates for the paper:

> **Seeing Is Not Sharing: Some Vision-Language Models Overestimate Common Ground in Asymmetric Dialogue**
> Nan Li, Albert Gatt, Massimo Poesio. SIGDIAL 2026.

We evaluate whether vision-language models (VLMs) can judge *interpretation matching* —
whether two participants in an information-asymmetric MapTask dialogue ground a reference
expression to the *same* landmark — under systematic manipulations of dialogue context and
map information (authentic maps, blank/shuffled controls, and textual map descriptions).

This repository contains **code and prompt templates only**. The evaluation dataset and the
raw MapTask materials are released separately (see [Data](#data)).

## Repository layout

```
config.sh              Central path configuration sourced by every run script
prompts/templates/     Prompt template + per-condition template fillings
scripts/               vLLM inference runners (prediction + calibration/logprobs)
experiments/           Per-experiment run scripts (bash + SLURM) and experiment plans
src/                   Analysis, calibration, and plotting helpers
analysis/paper-targeted/  Notebooks that reproduce the paper figures
```

### Experiments (`experiments/`)

| Folder | Paper section |
|---|---|
| `map-information-modality/`   | Map-information conditions: authentic maps, blank/shuffled controls, textual map descriptions |
| `additional-models/`          | Cross-model comparison (Qwen3-VL 2B/4B/8B, Gemma-3 4B/12B) + architecture-confound baseline |
| `calibration-complete/`       | Calibration analysis (top-20 logprobs) |
| `map-reading-sanity-check/`   | VLM map-reading sanity check (landmark listing + spatial description) |
| `qwen3-vl_on_text-only/`      | Qwen3-VL text-only baseline |

Each folder has a `*_experiment-plan.md` describing the model × condition grid.
`experiments/experiment-list.csv` is the master index.

## Setup

Requires Python ≥ 3.12. Dependencies are pinned in `uv.lock`:

```bash
uv sync
```

Inference additionally requires [vLLM](https://github.com/vllm-project/vllm) and a CUDA GPU
(all reported runs used a single NVIDIA A100 80 GB).

## Reproduction pipeline

This is a code release, so predictions and figures are **not** shipped. Regenerate them:

1. **Get the data** (see [Data](#data)) and place the interpretation-matching dataset and
   MapTask map images where the experiment scripts expect them.
2. **Generate prompts** — run the `create_prompts.ipynb` notebook in the relevant
   `experiments/` folder to expand `prompts/templates/` into per-condition prompt files.
3. **Run inference** — submit the experiment scripts, e.g.
   ```bash
   bash experiments/map-information-modality/exp_map-info-modality.sh \
       --model Qwen/Qwen3-VL-8B-Instruct
   ```
   or via SLURM with the accompanying `slurm_*.sbatch`. The runner is
   `scripts/exp_chat_vllm.py` (greedy decoding, constrained Yes/No output);
   `scripts/exp_chat_vllm_calibration.py` additionally records top-20 logprobs.
4. **Make figures** — run the notebooks under `analysis/paper-targeted/`.

> **Note on paths.** All run scripts source [`config.sh`](config.sh), which defines four
> base directories — `PROMPTS_DIR`, `PREDS_DIR`, `LOGITS_DIR`, `LOGS_DIR` — defaulting to a
> local layout under the repo (`prompts/generated`, `outputs/predictions`, `outputs/logits`,
> `logs`). Edit `config.sh` (or export those variables) to relocate inputs/outputs. The
> `.sbatch` files additionally contain site-specific `#SBATCH --partition=...` and
> environment-activation lines marked with `# EDIT:` — adjust them for your cluster.

## Data

The evaluation dataset is **not** included here. It is the perspectivist annotation release
of the HCRC MapTask corpus, released separately (CC-BY-4.0):

> Nan Li, Albert Gatt, Massimo Poesio. *Grounded Misunderstandings in Asymmetric Dialogue:
> A Perspectivist Annotation Scheme for MapTask.* LREC 2026, pp. 4988–5001.

The underlying MapTask dialogues and map images are from the
[HCRC Map Task Corpus](https://groups.inf.ed.ac.uk/maptask/) (Anderson et al., 1991) and are
subject to its original license; obtain them from the source rather than from this repository.

## License

Code in this repository is released under the [MIT License](LICENSE). The separately released
dataset is under CC-BY-4.0; MapTask source materials retain their original HCRC license.

## Citation

See [`CITATION.cff`](CITATION.cff).
