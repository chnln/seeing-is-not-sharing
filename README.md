# Seeing Is Not Sharing

**Seeing Is Not Sharing (SINS) Binary Common-Ground Judgment Dataset** is a
binary common-ground judgment (interpretation matching judgment) dataset. Each
instance asks whether the giver and follower interpret the referring expression
as the same landmark.

SINS is a downstream release of Grounded Misunderstandings in MapTask (GMMT).
It preserves GMMT's transaction-level provenance while exposing a compact
`yes`/`no` task:

- `yes`: GMMT labels the interpretation as `aligned`;
- `no`: GMMT labels it as `pending` or `misunderstood`.

## Related Resources

| Resource | Location |
| --- | --- |
| SINS dataset | [Hugging Face](https://huggingface.co/datasets/chnln/seeing-is-not-sharing) |
| SINS paper | [ACL Anthology](https://aclanthology.org/2026.sigdial-1.49/) · [arXiv:2606.31719](https://arxiv.org/abs/2606.31719), SIGDIAL 2026 |
| GMMT code | [GitHub](https://github.com/chnln/grounded-misunderstandings-in-maptask) |
| GMMT dataset | [Hugging Face](https://huggingface.co/datasets/chnln/grounded-misunderstandings-in-maptask) |
| GMMT paper | [LREC Proceedings](https://lrec.elra.info/lrec2026-main-392) · [arXiv:2511.03718](https://arxiv.org/abs/2511.03718), LREC 2026 |

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

Or, with pip:

```bash
pip install -e ".[dev]"
```

Download the GMMT companion dataset, which supplies the annotation tables and
reference-expression files used by the local scripts:

```bash
git clone https://huggingface.co/datasets/chnln/grounded-misunderstandings-in-maptask
```

Build or verify the released table from a local GMMT checkout:

```bash
uv run python -m scripts.build_dataset \
  --gmmt-dir /path/to/grounded-misunderstandings-in-maptask \
  --out-dir release/hf
```

To recover evaluation context locally, download the timed-unit files from the
[HCRC MapTask corpus](https://groups.inf.ed.ac.uk/maptask/) and use them with
the GMMT checkout above:

```bash
uv run python -m scripts.reconstruct_contexts \
  --instances release/hf/data/train-00000-of-00001.parquet \
  --gmmt-dir /path/to/grounded-misunderstandings-in-maptask \
  --maptask-tu-dir /path/to/maptaskv2-1/Data/timed-units \
  --out contexts/ref_contexts.json
```

Render the paper prompt format locally:

```bash
uv run python -m scripts.render_prompts \
  --instances release/hf/data/train-00000-of-00001.parquet \
  --contexts contexts/ref_contexts.json \
  --text-access startUntilCurLine \
  --map-access no_maps \
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

## Citation

SINS builds on the GMMT data and annotation scheme introduced in our LREC 2026
paper. If you use SINS, please cite both the SINS paper and the GMMT paper.

### SINS: Task and Experiments

Published at SIGDIAL 2026.

```bibtex
@inproceedings{li2026seeing,
  title = {Seeing Is Not Sharing: Some Vision-Language Models Overestimate Common Ground in Asymmetric Dialogue},
  author = {Li, Nan and Gatt, Albert and Poesio, Massimo},
  editor = {Choi, Jinho D. and Chen, Yun-Nung and Funakoshi, Kotaro and Emami, Ali},
  booktitle = {Proceedings of the 27th Annual Meeting of the Special Interest Group on Discourse and Dialogue},
  month = aug,
  year = {2026},
  address = {Atlanta, Georgia, USA},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2026.sigdial-1.49/},
  pages = {694--710}
}
```

Paper: [ACL Anthology](https://aclanthology.org/2026.sigdial-1.49/) ·
[arXiv preprint](https://arxiv.org/abs/2606.31719).

### GMMT: Data and Annotation Scheme

Published at LREC 2026.

```bibtex
@inproceedings{li2026grounded,
  title = {Grounded Misunderstandings in Asymmetric Dialogue: A Perspectivist Annotation Scheme for MapTask},
  author = {Li, Nan and Gatt, Albert and Poesio, Massimo},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {4988--5001},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  url = {https://lrec.elra.info/lrec2026-main-392},
  doi = {10.63317/59anbt78wyj7}
}
```

Paper: [LREC Proceedings](https://lrec.elra.info/lrec2026-main-392) ·
[arXiv preprint](https://arxiv.org/abs/2511.03718).

## License and Attribution

Code is released under the [MIT License](LICENSE). The SINS dataset is released
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It derives
from GMMT, which is also CC BY 4.0; see [NOTICE](NOTICE) and
[`dataset_card.md`](dataset_card.md) for provenance and restrictions.
