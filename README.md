# Seeing Is Not Sharing

**Seeing Is Not Sharing (SINS) Binary Common-Ground Judgment Dataset** is a
binary judgment dataset for dialogue common ground. Each instance asks whether
a referring expression is grounded in the dialogue state available to the two
participants.

SINS is a downstream release of
[Grounded Misunderstandings in MapTask (GMMT)](https://huggingface.co/datasets/chnln/grounded-misunderstandings-in-maptask).
It preserves GMMT's transaction-level provenance while exposing a compact
`yes`/`no` task:

- `yes`: the expression is aligned with the available common ground;
- `no`: the expression is pending or misunderstood.

## What Is Released

The Hugging Face dataset contains one Parquet split with 13,077 instances and
these columns:

| Column | Description |
| --- | --- |
| `ref_id` | Stable referring-expression identifier from GMMT. |
| `dialogue_id`, `map_id` | MapTask dialogue and map identifiers. |
| `utt_id`, `transaction_id` | Target utterance and GMMT transaction provenance. |
| `context_transaction_ids`, `end_utt_id_of_context` | Context-window provenance. |
| `timed_unit_ids` | Timed-unit provenance for the target expression. |
| `expression`, `status`, `gold_label` | Target expression, source status, and binary gold label. |

The SINS HF dataset does not contain dialogue context. It does not contain MapTask maps,
images, OCR, or image-derived text. It also excludes generated
prompts, model predictions, token log-probabilities, and external-source
copies. Those omissions are deliberate: MapTask source material is not
redistributed here.

## Local Workflow

Install the lightweight dependencies with:

```bash
uv sync --extra dev
```

Build or verify the released table from a local GMMT checkout:

```bash
uv run python -m scripts.build_dataset \
  --gmmt-dir /path/to/grounded-misunderstandings-in-maptask \
  --out release/hf/data/train-00000-of-00001.parquet
```

To recover evaluation context locally, provide your own authorised MapTask
timed-unit files together with GMMT. The reconstructed file is ignored by Git
and must not be uploaded to Hugging Face:

```bash
uv run python -m scripts.reconstruct_contexts \
  --instances release/hf/data/train-00000-of-00001.parquet \
  --gmmt-dir /path/to/grounded-misunderstandings-in-maptask \
  --maptask-tu-dir /path/to/timed-units_utt_filled \
  --out contexts/ref_contexts.json
```

Render the paper prompt format locally:

```bash
uv run python -m scripts.render_prompts \
  --instances release/hf/data/train-00000-of-00001.parquet \
  --contexts contexts/ref_contexts.json \
  --condition no_maps \
  --out outputs/prompts.jsonl
```

`no_maps` requires no images. For `both_maps`, `giver_only`, or
`follower_only`, pass `--map-dir` pointing to your local MapTask PNG files.

## Minimal Model Template

The optional vLLM extra provides a constrained Yes/No template. It sends the
rendered JSONL prompts to a model and writes only `{ref_id, judge}` records:

```bash
uv sync --extra vllm
uv run python -m scripts.vllm_yes_no_template \
  --prompts outputs/prompts.jsonl \
  --model /path/to/model \
  --out outputs/predictions.jsonl
uv run python -m scripts.evaluate_predictions \
  --instances release/hf/data/train-00000-of-00001.parquet \
  --predictions outputs/predictions.jsonl
```

The template is intentionally a starting point, not a reproduction harness for
all paper experiments.

## Prompts and Paper Context

[`prompts/prompt_template.py`](prompts/prompt_template.py) and
[`prompts/prompt_fillings.json`](prompts/prompt_fillings.json) preserve the
prompt assets used in the paper. [`prompts/textual_map_information.md`](prompts/textual_map_information.md)
records two appendix textual-map examples for documentation only; it is not a
dataset field and this repository does not generate that text from images.

For the complete experimental account, cite the accompanying SIGDIAL paper.
This repository releases the SINS dataset and prompt interfaces, not a full
paper-reproduction pipeline.

## License and Attribution

Code is released under the [MIT License](LICENSE). The SINS dataset is released
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It derives
from GMMT, which is also CC BY 4.0; see [NOTICE](NOTICE) and
[`dataset_card.md`](dataset_card.md) for provenance and restrictions.
