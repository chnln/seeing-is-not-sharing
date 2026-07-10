---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
tags:
- dialogue
- common-ground
- maptask
- vision-language
size_categories:
- 10K<n<100K
---

# Seeing Is Not Sharing (SINS) Binary Common-Ground Judgment Dataset

## Dataset Description

SINS is a 13,077-instance binary common-ground judgment dataset for
interpretation matching, derived from the public Grounded Misunderstandings in
MapTask (GMMT) annotations. Each instance asks whether the giver and follower
interpret the referring expression as the same landmark. Rows preserve
identifiers and context-window provenance while mapping GMMT status to
`gold_label`: `aligned` becomes `yes`; `pending` and `misunderstood` become
`no`.

## Related Resources

| Resource | Location |
| --- | --- |
| SINS code and prompts | [GitHub](https://github.com/chnln/seeing-is-not-sharing) |
| SINS paper | [arXiv:2606.31719](https://arxiv.org/abs/2606.31719), to appear in SIGDIAL 2026 |
| GMMT code | [GitHub](https://github.com/chnln/grounded-misunderstandings-in-maptask) |
| GMMT dataset | [Hugging Face](https://huggingface.co/datasets/chnln/grounded-misunderstandings-in-maptask) |
| GMMT paper | [arXiv:2511.03718](https://arxiv.org/abs/2511.03718), LREC 2026 |

## Load with Datasets

```python
from datasets import load_dataset

ds = load_dataset("chnln/seeing-is-not-sharing", split="train")
```

## Data Fields

The release contains `ref_id`, `dialogue_id`, `map_id`, `utt_id`,
`transaction_id`, `context_transaction_ids`, `end_utt_id_of_context`,
`timed_unit_ids`, `expression`, `status`, and `gold_label`.

## Excluded Material

The SINS HF dataset does not contain dialogue context. It does not contain MapTask maps,
images, OCR, or image-derived text. It also does not contain
generated prompts, model predictions, log-probabilities, or copies of the
underlying MapTask source files. Users who download MapTask themselves may
reconstruct context with the SINS code repository and a local GMMT checkout.

## Licensing and Provenance

SINS is released under CC BY 4.0. It is derived from
[GMMT](https://huggingface.co/datasets/chnln/grounded-misunderstandings-in-maptask),
which is also CC BY 4.0. Please cite both the SINS paper and GMMT when using
this dataset. See `NOTICE` in the code repository for details.

## Citation

SINS accompanies the following paper, which will appear in SIGDIAL 2026:

```bibtex
@misc{li2026seeing,
  title = {Seeing Is Not Sharing: Some Vision-Language Models Overestimate Common Ground in Asymmetric Dialogue},
  author = {Li, Nan and Gatt, Albert and Poesio, Massimo},
  year = {2026},
  eprint = {2606.31719},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2606.31719},
  note = {To appear in SIGDIAL 2026}
}
```
